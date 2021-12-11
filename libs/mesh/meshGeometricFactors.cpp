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

void mesh_t::GeometricFactors(){

  /* number of second order geometric factors */
  Nggeo = 7;
  ggeo.malloc(Nelements*Nggeo*Np);

  for(dlong e=0;e<Nelements;++e){ /* for each element */

    dfloat xr = 0, xs = 0, xt = 0;
    dfloat yr = 0, ys = 0, yt = 0;
    dfloat zr = 0, zs = 0, zt = 0;

    //fake the geofactors (assumes all elements are bi-unit cubes)
    xr = 1.0;
    ys = 1.0;
    zt = 1.0;

    /* compute geometric factors for affine coordinate transform*/
    dfloat J = xr*(ys*zt-zs*yt) - yr*(xs*zt-zs*xt) + zr*(xs*yt-ys*xt);
    dfloat rx =  (ys*zt - zs*yt)/J, ry = -(xs*zt - zs*xt)/J, rz =  (xs*yt - ys*xt)/J;
    dfloat sx = -(yr*zt - zr*yt)/J, sy =  (xr*zt - zr*xt)/J, sz = -(xr*yt - yr*xt)/J;
    dfloat tx =  (yr*zs - zr*ys)/J, ty = -(xr*zs - zr*xs)/J, tz =  (xr*ys - yr*xs)/J;

    for(int k=0;k<Nq;++k){
      for(int j=0;j<Nq;++j){
        for(int i=0;i<Nq;++i){

          int n = i + j*Nq + k*Nq*Nq;
          dfloat JW = J*gllw[i]*gllw[j]*gllw[k];

          /* store second order geometric factors */
          ggeo[Nggeo*Np*e + n + Np*G00ID] = JW*(rx*rx + ry*ry + rz*rz);
          ggeo[Nggeo*Np*e + n + Np*G01ID] = JW*(rx*sx + ry*sy + rz*sz);
          ggeo[Nggeo*Np*e + n + Np*G02ID] = JW*(rx*tx + ry*ty + rz*tz);
          ggeo[Nggeo*Np*e + n + Np*G11ID] = JW*(sx*sx + sy*sy + sz*sz);
          ggeo[Nggeo*Np*e + n + Np*G12ID] = JW*(sx*tx + sy*ty + sz*tz);
          ggeo[Nggeo*Np*e + n + Np*G22ID] = JW*(tx*tx + ty*ty + tz*tz);
          // ggeo[Nggeo*Np*e + n + Np*GWJID] = JW;
          ggeo[Nggeo*Np*e + n + Np*GWJID] = weight[Np*e + n]; //inverse counting weights
        }
      }
    }
  }
}

} //namespace libp