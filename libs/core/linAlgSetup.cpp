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

#include "core.hpp"
#include "linAlg.hpp"
#include "platform.hpp"
#include "parameters.hpp"

namespace libp {

using std::string;
using std::stringstream;

linAlg_t::linAlg_t(platform_t *_platform) {

  platform = _platform;
  kernelInfo = platform->props();

  parameters_t tuningParameters;
  std::string filename = platform->exePath() + "/json/linAlg.json";

  properties_t keys;
  keys["dfloat"] = (sizeof(dfloat)==4) ? "float" : "double";
  keys["mode"] = platform->device.mode();

  std::string arch = platform->device.arch();
  if (platform->device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  std::string name = "linAlg.okl";
  tuningParameters.load(filename, platform->comm);
  properties_t matchProps = tuningParameters.findProperties(name, keys);

  //add defines
  kernelInfo["defines"] += matchProps["props"];

  /*Read blocksizes from properties*/
  normBlockSize = static_cast<int>(matchProps["props/NORM_BLOCKSIZE"]);
  innerProdBlockSize = static_cast<int>(matchProps["props/DOT_BLOCKSIZE"]);

  //pinned scratch buffer
  const int maxBlockSize = 1024;
  h_scratch = platform->hostMalloc<dfloat>(maxBlockSize);
  o_scratch = platform->malloc<dfloat>(maxBlockSize);
}

//initialize list of kernels
void linAlg_t::InitKernels(std::vector<string> kernels) {

  for (size_t i=0;i<kernels.size();i++) {
    string name = kernels[i];
    if (name=="set") {
      if (setKernel.isInitialized()==false)
        setKernel = platform->buildKernel("libs/core/okl/"
                                        "linAlgSet.okl",
                                        "set",
                                        kernelInfo);
    } else if (name=="axpy") {
      if (axpyKernel.isInitialized()==false)
        axpyKernel = platform->buildKernel("libs/core/okl/"
                                        "linAlgAXPY.okl",
                                        "axpy",
                                        kernelInfo);
    } else if (name=="norm2") {
      if (norm2Kernel.isInitialized()==false)
        norm2Kernel = platform->buildKernel("libs/core/okl/"
                                        "linAlgNorm2.okl",
                                        "norm2",
                                        kernelInfo);
    } else if (name=="innerProd") {
      if (innerProdKernel.isInitialized()==false)
        innerProdKernel = platform->buildKernel("libs/core/okl/"
                                        "linAlgInnerProd.okl",
                                        "innerProd",
                                        kernelInfo);
    } else {
      LIBP_FORCE_ABORT("Requested linAlg routine \"" << name << "\" not found");
    }
  }
}

} //namespace libp
