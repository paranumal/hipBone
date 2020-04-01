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

#include "hipBone.hpp"

hipBone_t& hipBone_t::Setup(mesh_t& mesh, linAlg_t& linAlg){

  hipBone_t* hipBone = new hipBone_t(mesh, linAlg);

  hipBone->lambda = 0.1; //hard code

  //setup linear algebra module
  hipBone->linAlg.InitKernels({"set", "axpy", "innerProd", "norm2"},
                                mesh.comm);

  //tmp local storage buffer for Ax op
  hipBone->o_AqL = mesh.device.malloc(mesh.Np*mesh.Nelements*sizeof(dfloat));

  // OCCA build stuff
  occa::properties kernelInfo = hipBone->props; //copy base occa properties

  // Ax kernel
  hipBone->operatorKernel = buildKernel(mesh.device,
                                   DHIPBONE "/okl/hipBoneAx.okl",
                                   "hipBoneAx",
                                   kernelInfo, mesh.comm);

  hipBone->forcingKernel = buildKernel(mesh.device,
                                   DHIPBONE "/okl/hipBoneRhs.okl",
                                   "hipBoneRhs",
                                   kernelInfo, mesh.comm);

  return *hipBone;
}

hipBone_t::~hipBone_t() {
  o_AqL.free();
  operatorKernel.free();
  forcingKernel.free();
}
