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

#include "mesh.hpp"

namespace libp {

// ------------------------------------------------------------------------
// HEX 3D NODES
// ------------------------------------------------------------------------
void mesh_t::NodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]){
  int _Nq = _N+1;

  memory<dfloat> r1D(_Nq);
  JacobiGLL(_N, r1D.ptr()); //Gauss-Legendre-Lobatto nodes

  //Tensor product
  for (int k=0;k<_Nq;k++) {
    for (int j=0;j<_Nq;j++) {
      for (int i=0;i<_Nq;i++) {
        _r[i+j*_Nq+k*_Nq*_Nq] = r1D[i];
        _s[i+j*_Nq+k*_Nq*_Nq] = r1D[j];
        _t[i+j*_Nq+k*_Nq*_Nq] = r1D[k];
      }
    }
  }
}

void mesh_t::FaceNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _faceNodes[]){
  int _Nq = _N+1;
  int _Nfp = _Nq*_Nq;
  int _Np = _Nq*_Nq*_Nq;

  int cnt[6];
  for (int i=0;i<6;i++) cnt[i]=0;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for (int n=0;n<_Np;n++) {
    if(fabs(_t[n]+1)<NODETOL)
      _faceNodes[0*_Nfp+(cnt[0]++)] = n;
    if(fabs(_s[n]+1)<NODETOL)
      _faceNodes[1*_Nfp+(cnt[1]++)] = n;
    if(fabs(_r[n]-1)<NODETOL)
      _faceNodes[2*_Nfp+(cnt[2]++)] = n;
    if(fabs(_s[n]-1)<NODETOL)
      _faceNodes[3*_Nfp+(cnt[3]++)] = n;
    if(fabs(_r[n]+1)<NODETOL)
      _faceNodes[4*_Nfp+(cnt[4]++)] = n;
    if(fabs(_t[n]-1)<NODETOL)
      _faceNodes[5*_Nfp+(cnt[5]++)] = n;
  }
}

void mesh_t::VertexNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]){
  int _Nq = _N+1;
  int _Np = _Nq*_Nq*_Nq;

  dfloat deps = 1.;
  while((1.+deps)>1.)
    deps *= 0.5;

  const dfloat NODETOL = 1000.*deps;

  for(int n=0;n<_Np;++n){
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]+1)*(_s[n]+1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[0] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]+1)*(_s[n]+1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[1] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]-1)*(_s[n]-1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[2] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]-1)*(_s[n]-1)+(_t[n]+1)*(_t[n]+1)<NODETOL)
      _vertexNodes[3] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]+1)*(_s[n]+1)+(_t[n]-1)*(_t[n]-1)<NODETOL)
      _vertexNodes[4] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]+1)*(_s[n]+1)+(_t[n]-1)*(_t[n]-1)<NODETOL)
      _vertexNodes[5] = n;
    if( (_r[n]-1)*(_r[n]-1)+(_s[n]-1)*(_s[n]-1)+(_t[n]-1)*(_t[n]-1)<NODETOL)
      _vertexNodes[6] = n;
    if( (_r[n]+1)*(_r[n]+1)+(_s[n]-1)*(_s[n]-1)+(_t[n]-1)*(_t[n]-1)<NODETOL)
      _vertexNodes[7] = n;
  }
}

/*Find a matching array between nodes on matching faces */
void mesh_t::FaceNodeMatchingHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],
                                   int _faceNodes[], int R[]){

  int _Nq = _N+1;
  int _Nfp = _Nq*_Nq;

  const dfloat NODETOL = 1.0e-5;

  dfloat V0[4][2] = {{-1.0,-1.0},{ 1.0,-1.0},{ 1.0, 1.0},{-1.0, 1.0}};
  dfloat V1[4][2] = {{-1.0,-1.0},{-1.0, 1.0},{ 1.0, 1.0},{ 1.0,-1.0}};

  dfloat EX0[Nverts], EY0[Nverts];
  dfloat EX1[Nverts], EY1[Nverts];

  memory<dfloat> x0(_Nfp);
  memory<dfloat> y0(_Nfp);

  memory<dfloat> x1(_Nfp);
  memory<dfloat> y1(_Nfp);


  for (int fM=0;fM<Nfaces;fM++) {

    for (int v=0;v<Nverts;v++) {
      EX0[v] = 0.0; EY0[v] = 0.0;
    }
    //setup top element with face fM on the bottom
    for (int v=0;v<NfaceVertices;v++) {
      int fv = faceVertices[fM*NfaceVertices + v];
      EX0[fv] = V0[v][0]; EY0[fv] = V0[v][1];
    }

    for(int n=0;n<_Nfp;++n){ /* for each face node */
      const int fn = _faceNodes[fM*_Nfp+n];

      /* (r,s,t) coordinates of interpolation nodes*/
      dfloat rn = _r[fn];
      dfloat sn = _s[fn];
      dfloat tn = _t[fn];

      /* physical coordinate of interpolation node */
      x0[n] =
        +0.125*(1-rn)*(1-sn)*(1-tn)*EX0[0]
        +0.125*(1+rn)*(1-sn)*(1-tn)*EX0[1]
        +0.125*(1+rn)*(1+sn)*(1-tn)*EX0[2]
        +0.125*(1-rn)*(1+sn)*(1-tn)*EX0[3]
        +0.125*(1-rn)*(1-sn)*(1+tn)*EX0[4]
        +0.125*(1+rn)*(1-sn)*(1+tn)*EX0[5]
        +0.125*(1+rn)*(1+sn)*(1+tn)*EX0[6]
        +0.125*(1-rn)*(1+sn)*(1+tn)*EX0[7];

      y0[n] =
        +0.125*(1-rn)*(1-sn)*(1-tn)*EY0[0]
        +0.125*(1+rn)*(1-sn)*(1-tn)*EY0[1]
        +0.125*(1+rn)*(1+sn)*(1-tn)*EY0[2]
        +0.125*(1-rn)*(1+sn)*(1-tn)*EY0[3]
        +0.125*(1-rn)*(1-sn)*(1+tn)*EY0[4]
        +0.125*(1+rn)*(1-sn)*(1+tn)*EY0[5]
        +0.125*(1+rn)*(1+sn)*(1+tn)*EY0[6]
        +0.125*(1-rn)*(1+sn)*(1+tn)*EY0[7];
    }

    for (int fP=0;fP<Nfaces;fP++) { /*For each neighbor face */
      for (int rot=0;rot<NfaceVertices;rot++) { /* For each face rotation */
        // Zero vertices
        for (int v=0;v<Nverts;v++) {
          EX1[v] = 0.0; EY1[v] = 0.0;
        }
        //setup bottom element with face fP on the top
        for (int v=0;v<NfaceVertices;v++) {
          int fv = faceVertices[fP*NfaceVertices + ((v+rot)%NfaceVertices)];
          EX1[fv] = V1[v][0]; EY1[fv] = V1[v][1];
        }

        for(int n=0;n<_Nfp;++n){ /* for each node */
          const int fn = _faceNodes[fP*_Nfp+n];

          /* (r,s,t) coordinates of interpolation nodes*/
          dfloat rn = _r[fn];
          dfloat sn = _s[fn];
          dfloat tn = _t[fn];

          /* physical coordinate of interpolation node */
          x1[n] =  0.125*(1-rn)*(1-sn)*(1-tn)*EX1[0]
                  +0.125*(1+rn)*(1-sn)*(1-tn)*EX1[1]
                  +0.125*(1+rn)*(1+sn)*(1-tn)*EX1[2]
                  +0.125*(1-rn)*(1+sn)*(1-tn)*EX1[3]
                  +0.125*(1-rn)*(1-sn)*(1+tn)*EX1[4]
                  +0.125*(1+rn)*(1-sn)*(1+tn)*EX1[5]
                  +0.125*(1+rn)*(1+sn)*(1+tn)*EX1[6]
                  +0.125*(1-rn)*(1+sn)*(1+tn)*EX1[7];

          y1[n] =  0.125*(1-rn)*(1-sn)*(1-tn)*EY1[0]
                  +0.125*(1+rn)*(1-sn)*(1-tn)*EY1[1]
                  +0.125*(1+rn)*(1+sn)*(1-tn)*EY1[2]
                  +0.125*(1-rn)*(1+sn)*(1-tn)*EY1[3]
                  +0.125*(1-rn)*(1-sn)*(1+tn)*EY1[4]
                  +0.125*(1+rn)*(1-sn)*(1+tn)*EY1[5]
                  +0.125*(1+rn)*(1+sn)*(1+tn)*EY1[6]
                  +0.125*(1-rn)*(1+sn)*(1+tn)*EY1[7];
        }

        /* for each node on this face find the neighbor node */
        for(int n=0;n<_Nfp;++n){
          const dfloat xM = x0[n];
          const dfloat yM = y0[n];

          int m=0;
          for(;m<_Nfp;++m){ /* for each neighbor node */
            const dfloat xP = x1[m];
            const dfloat yP = y1[m];

            /* distance between target and neighbor node */
            const dfloat dist = pow(xM-xP,2) + pow(yM-yP,2);

            /* if neighbor node is close to target, match */
            if(dist<NODETOL){
              R[fM*Nfaces*NfaceVertices*_Nfp
                + fP*NfaceVertices*_Nfp
                + rot*_Nfp + n] = m;
              break;
            }
          }

          /*Check*/
          const dfloat xP = x1[m];
          const dfloat yP = y1[m];

          /* distance between target and neighbor node */
          const dfloat dist = pow(xM-xP,2) + pow(yM-yP,2);
          //This shouldn't happen
          LIBP_ABORT("Unable to match face node, face: " << fM
                     << ", matching face: " << fP
                     << ", rotation: " << rot
                     << ", node: " << n
                     << ". Is the reference node set not symmetric?",
                     dist>NODETOL);
        }
      }
    }
  }
}

} //namespace libp
