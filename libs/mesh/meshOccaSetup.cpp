/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

void mesh_t::OccaSetup(){

  o_internalElementIds = platform.malloc<dlong>(internalElementIds);
  o_haloElementIds = platform.malloc<dlong>(haloElementIds);
  o_globalGatherElementList = platform.malloc<dlong>(globalGatherElementList);
  o_localGatherElementList = platform.malloc<dlong>(localGatherElementList);

  props = platform.props(); //copy platform props

  props["defines/" "p_dim"]= dim;
  props["defines/" "p_N"]= N;
  props["defines/" "p_Nq"]= N+1;
  props["defines/" "p_Np"]= Np;
  props["defines/" "p_Nfp"]= Nfp;
  props["defines/" "p_Nfaces"]= Nfaces;
  props["defines/" "p_NfacesNfp"]= Nfp*Nfaces;
  props["defines/" "p_Nggeo"]= Nggeo;

  props["defines/" "p_G00ID"]= G00ID;
  props["defines/" "p_G01ID"]= G01ID;
  props["defines/" "p_G02ID"]= G02ID;
  props["defines/" "p_G11ID"]= G11ID;
  props["defines/" "p_G12ID"]= G12ID;
  props["defines/" "p_G22ID"]= G22ID;
  props["defines/" "p_GWJID"]= GWJID;

  o_D = platform.malloc<dfloat>(D);
  o_ggeo = platform.malloc<dfloat>(ggeo);
}

} //namespace libp
