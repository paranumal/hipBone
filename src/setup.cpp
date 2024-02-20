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

#include "hipBone.hpp"
#include "parameters.hpp"

void hipBone_t::Setup(platform_t& _platform, mesh_t& _mesh){

  platform = _platform;
  mesh = _mesh;

  lambda = 0.1; //hard code

  //setup linear algebra module
  platform.linAlg().InitKernels({"set", "axpy", "innerProd", "norm2"});

  //Trigger JIT kernel builds
  ogs::InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  //tmp local storage buffer for Ax op
  o_AqL = platform.malloc<dfloat>(mesh.Np*mesh.Nelements);

  // OCCA build stuff
  properties_t kernelInfo = mesh.props; //copy mesh occa properties

  forcingKernel = platform.buildKernel("okl/hipBoneRhs.okl",
                                       "hipBoneRhs", kernelInfo);

  parameters_t tuningParameters;
  std::string filename = platform.exePath() + "/json/hipBoneAx.json";

  properties_t keys;
  keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
  keys["N"] = mesh.N;
  keys["mode"] = platform.device.mode();

  std::string arch = platform.device.arch();
  if (platform.device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  std::string name = "hipBoneAx.okl";

  if (platform.settings().compareSetting("VERBOSE", "TRUE") && mesh.rank==0) {
    std::cout << "Loading Tuning Parameters, looking for match for Name:'" << name << "', keys:" << tuningParameters.toString(keys) << std::endl;
  }

  tuningParameters.load(filename, mesh.comm);
  properties_t matchProps = tuningParameters.findProperties(name, keys);

  if (platform.settings().compareSetting("VERBOSE", "TRUE") && mesh.rank==0) {
    std::cout << "Found best match = " << tuningParameters.toString(matchProps) << std::endl;
  }

  //add defines
  kernelInfo["defines"] += matchProps["props"];

  // Ax kernel
  operatorKernel = platform.buildKernel("okl/hipBoneAx.okl",
                                        "hipBoneAx", kernelInfo);
}

