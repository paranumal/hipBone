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

namespace libp {

constexpr int LINALG_BLOCKSIZE = 256;

using std::string;
using std::stringstream;

linAlg_t::linAlg_t(platform_t *_platform): blocksize(LINALG_BLOCKSIZE) {

  platform = _platform;
  kernelInfo = platform->props();

  //add defines
  kernelInfo["defines/" "p_blockSize"] = (int)LINALG_BLOCKSIZE;

  //pinned scratch buffer
  h_scratch = platform->hostMalloc<dfloat>(LINALG_BLOCKSIZE);
  o_scratch = platform->malloc<dfloat>(LINALG_BLOCKSIZE);
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
      if (norm2Kernel1.isInitialized()==false)
        norm2Kernel1 = platform->buildKernel("libs/core/okl/"
                                        "linAlgNorm2.okl",
                                        "norm2_1",
                                        kernelInfo);

      if (norm2Kernel2.isInitialized()==false)
        norm2Kernel2 = platform->buildKernel("libs/core/okl/"
                                        "linAlgNorm2.okl",
                                        "norm2_2",
                                        kernelInfo);
    } else if (name=="innerProd") {
      if (innerProdKernel1.isInitialized()==false)
        innerProdKernel1 = platform->buildKernel("libs/core/okl/"
                                        "linAlgInnerProd.okl",
                                        "innerProd_1",
                                        kernelInfo);

      if (innerProdKernel2.isInitialized()==false)
        innerProdKernel2 = platform->buildKernel("libs/core/okl/"
                                        "linAlgInnerProd.okl",
                                        "innerProd_2",
                                        kernelInfo);
    } else {
      LIBP_FORCE_ABORT("Requested linAlg routine \"" << name << "\" not found");
    }
  }
}

} //namespace libp
