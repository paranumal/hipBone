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
void mesh_t::NodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t){
  int _Nq = _N+1;

  dfloat *r1D = (dfloat*) malloc(_Nq*sizeof(dfloat));
  JacobiGLL(_N, r1D); //Gauss-Legendre-Lobatto nodes

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

  free(r1D);
}

void mesh_t::FaceNodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t, int *_faceNodes){
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

void mesh_t::VertexNodesHex3D(int _N, dfloat *_r, dfloat *_s, dfloat *_t, int *_vertexNodes){
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

} //namespace libp