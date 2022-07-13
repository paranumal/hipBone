#include <hip/hip_runtime.h>

/*

The MIT License (MIT)

Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/

#if p_N!=14
#error "Not today"
#endif

#define p_pad 1

#define NPOINTS 15
#define SIZE 16

extern "C" __global__ __launch_bounds__(256) void hipBoneAx_p14_mfma(const dlong Nelements,
                                                                     const dlong  * elementList,
                                                                     const dlong  * GlobalToLocal,
                                                                     const dfloat * ggeo,
                                                                     const dfloat * D,
                                                                     const dfloat lambda,
                                                                     const dfloat * q,
                                                                           dfloat * Aq){

  const int e = blockIdx.x;
  const int r_e = e;

  __shared__ dfloat s_D[SIZE][SIZE+p_pad];
  __shared__ dfloat s_DT[SIZE][SIZE+p_pad];
  __shared__ dfloat s_q[SIZE][SIZE+p_pad];
  __shared__ dfloat s_v[SIZE][SIZE+p_pad];
  __shared__ dfloat s_w[SIZE][SIZE+p_pad];

  dfloat r_GDqt = 0.0;
  dfloat r_Aqk = 0.0;

  // register array to hold u(i,j,0:N) private to thread
  dfloat r_q[SIZE];
  // array for results Au(i,j,0:N)
  dfloat r_Aq[SIZE];

  dlong element;

  dfloat4 r_qr;

  const int i = threadIdx.x;
  const int j = threadIdx.y;

  // Branch the wavefront on the non-padded part.
  // I think I need to do this because full 64-wide wavefronts
  // must enter the mfma builtin (I'm not sure about this).
  if (i < NPOINTS && j < NPOINTS) {
    //load D into local memory
    // s_D[i][j] = d \phi_i at node j
    const dfloat Dji = D[NPOINTS*j+i];// D is column major
    s_D[j][i] = Dji;
    s_DT[i][j] = Dji;
  }
  else {
    s_D[j][i] = 0.0;
    s_DT[i][j] = 0.0;
  }


  element = elementList[e];
  const dlong base = i + j*NPOINTS + element*NPOINTS*NPOINTS*NPOINTS;

  // load pencil of u into register
  if (i < NPOINTS && j < NPOINTS) {
    #pragma unroll NPOINTS
    for (int k=0;k<NPOINTS;k++) {
      const dlong id = GlobalToLocal[base + k*NPOINTS*NPOINTS];
      r_q[k] = (id!=-1) ? q[id] : 0.0;
    }
  }
  // Zero out the padded entry
  r_q[NPOINTS] = 0.0;

  // r_q is uninitialised in the padded portion

  #pragma unroll SIZE
  for (int k=0;k<SIZE;k++) {
    // Yes, SIZE
    r_Aq[k] = 0.0;
  }
  __syncthreads();

  // I think this should be NPOINTS?
  // And we just handle the tail iteration separately?
  for(int k=0;k<NPOINTS;++k){

    if (i < NPOINTS && j < NPOINTS) {
      s_q[j][i] = r_q[k];
    }
    else {
      s_q[j][i] = 0.0;
    }
    __syncthreads();

    dfloat r_G00, r_G01, r_G02, r_G11, r_G12, r_G22, r_GwJ;

    if(r_e<Nelements && i < NPOINTS && j < NPOINTS){
      // prefetch geometric factors
      const dlong gbase = p_Nggeo*(element*NPOINTS*NPOINTS*NPOINTS + k*NPOINTS*NPOINTS + j*NPOINTS + i);

      r_GwJ = ggeo[gbase+p_GWJID];
      r_G00 = ggeo[gbase+p_G00ID];
      r_G01 = ggeo[gbase+p_G01ID];
      r_G11 = ggeo[gbase+p_G11ID];
      r_G12 = ggeo[gbase+p_G12ID];
      r_G02 = ggeo[gbase+p_G02ID];
      r_G22 = ggeo[gbase+p_G22ID];
    }

    dfloat qr = 0.f;
    dfloat qs = 0.f;
    dfloat qt = 0;

    // Don't care about the tail of the wavefront here
    #pragma unroll NPOINTS
    for (int m=0;m<NPOINTS;m++) {
      qt += s_DT[m][k]*r_q[m];  // Don't need the last entry; it would add 0
    }

    #pragma unroll 4
    for (int m=0;m<4;m++) {
      dfloat A, B;
      B = s_DT[(4*m)+(j%4)][i];
      A = s_q[(i%4)+4*(j/4)][(4*m)+(j%4)];

      qr = __builtin_amdgcn_mfma_f64_4x4x4f64(A, B, qr, 0, 0, 0); // x-deriv
    }

    #pragma unroll 4
    for (int m=0;m<4;m++) {
      dfloat A, B, C;
      B = s_q[4*m+j%4][i];
      A = s_DT[4*m+j%4][i%4+4*(j/4)];

      // Ordering of the result is the same as the B matrix
      qs = __builtin_amdgcn_mfma_f64_4x4x4f64(A, B, qs, 0, 0, 0); // y-deriv
    }

    if (i < NPOINTS && j < NPOINTS) {
      s_v[j][i] = (r_G00*qr + r_G01*qs + r_G02*qt);
      s_w[j][i] = (r_G01*qr + r_G11*qs + r_G12*qt);
      r_GDqt    = (r_G02*qr + r_G12*qs + r_G22*qt);

      r_Aqk = r_GwJ*lambda*r_q[k];
    }
    else {
      s_v[j][i] = 0.0;
      s_w[j][i] = 0.0;
      r_GDqt    = 0.0;
      r_Aqk     = 0.0;
    }
    __syncthreads();

    #pragma unroll NPOINTS
    for (int m=0;m<NPOINTS;m++) {
      r_Aq[m] += s_D[k][m]*r_GDqt;  // Do we need to worry about the last entry?
    }

    #pragma unroll 4
    for (int m=0;m<4;m++) {
      dfloat A, B;
      B = s_w[4*m+j%4][i];
      A = s_D[4*m+j%4][i%4+4*(j/4)];

      // Ordering of the result is the same as the B matrix
      r_Aqk = __builtin_amdgcn_mfma_f64_4x4x4f64(A, B, r_Aqk, 0, 0, 0); // y-deriv
    }

    #pragma unroll 4
    for (int m=0;m<4;m++) {
      dfloat A, B, C;
      B = s_D[4*m+j%4][i];
      A = s_v[i%4+4*(j/4)][4*m+j%4];

      r_Aqk = __builtin_amdgcn_mfma_f64_4x4x4f64(A, B, r_Aqk, 0, 0, 0); // x-deriv
    }

    r_Aq[k] += r_Aqk;
    __syncthreads();
  } //end Layer by layer

  // write out
  if(e<Nelements && i < NPOINTS && j < NPOINTS){
    const dlong id = element*NPOINTS*NPOINTS*NPOINTS + j*NPOINTS + i;

    #pragma unroll NPOINTS
    for (int k=0;k<NPOINTS;k++) {
      __builtin_nontemporal_store(r_Aq[k], &(Aq[id+k*NPOINTS*NPOINTS]));
    }
  }
}
