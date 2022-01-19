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

#include "linAlg.hpp"
#include "platform.hpp"


namespace libp {

/*********************/
/* vector operations */
/*********************/

// o_x[n] = alpha
void linAlg_t::set(const dlong N, const dfloat alpha, occa::memory& o_x) {
  setKernel(N, alpha, o_x);
}

// o_y[n] = beta*o_y[n] + alpha*o_x[n]
void linAlg_t::axpy(const dlong N, const dfloat alpha, occa::memory& o_x,
                    const dfloat beta,  occa::memory& o_y) {
  axpyKernel(N, alpha, o_x, beta, o_y);
}

// ||o_a||_2
dfloat linAlg_t::norm2(const dlong N, occa::memory& o_a, MPI_Comm comm) {
  //TODO, maybe complete reduction on device with second kernel?
  int Nblock = (N+blocksize-1)/blocksize;
  Nblock = (Nblock>blocksize) ? blocksize : Nblock; //limit to blocksize entries

  norm2Kernel1(Nblock, N, o_a, o_scratch);
  norm2Kernel2(Nblock, o_scratch);

  o_scratch.copyTo(scratch, 1*sizeof(dfloat), 0, "async: true");
  platform->device.finish();

  dfloat norm = 0;
  MPI_Allreduce(scratch, &norm, 1, MPI_DFLOAT, MPI_SUM, comm);

  return sqrt(norm);
}

// o_x.o_y
dfloat linAlg_t::innerProd(const dlong N, occa::memory& o_x, occa::memory& o_y,
                           MPI_Comm comm) {

  int Nblock = (N+blocksize-1)/blocksize;
  Nblock = (Nblock>blocksize) ? blocksize : Nblock; //limit to blocksize entries

  innerProdKernel1(Nblock, N, o_x, o_y, o_scratch);
  innerProdKernel2(Nblock, o_scratch);

  o_scratch.copyTo(scratch, 1*sizeof(dfloat), 0, "async: true");
  platform->device.finish();

  dfloat dot = 0;
  MPI_Allreduce(scratch, &dot, 1, MPI_DFLOAT, MPI_SUM, comm);

  return dot;
}

} //namespace libp
