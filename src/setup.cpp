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

void hipBone_t::Setup(platform_t& _platform, mesh_t& _mesh){

  platform = _platform;
  mesh = _mesh;

  lambda = 0.1; //hard code

  //setup linear algebra module
  platform.linAlg().InitKernels({"set", "axpy", "innerProd", "norm2"},
                                mesh.comm);

  //Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  //tmp local storage buffer for Ax op
  o_AqL = platform.malloc(mesh.Np*mesh.Nelements*sizeof(dfloat));

  // OCCA build stuff
  occa::properties kernelInfo = mesh.props; //copy mesh occa properties

  forcingKernel = platform.buildKernel(DHIPBONE "/okl/hipBoneRhs.okl",
                                   "hipBoneRhs", kernelInfo);

  // Ax kernels
  // Use the non-MFMA operator kernel for all orders except 15 and use the MFMA
  // kernels at only order 15
  if (mesh.Nq == 16) {
    kernelInfo["okl/enabled"] = false;
    operatorKernel = platform.buildKernel(DHIPBONE "/okl/hipBoneAx_mfma.cpp",
                                     "hipBoneAx_mfma", kernelInfo);
  }
  else {
    operatorKernel = platform.buildKernel(DHIPBONE "/okl/hipBoneAx.okl",
                                     "hipBoneAx", kernelInfo);
  }
}
