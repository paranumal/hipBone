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

#include "core.hpp"
#include "linAlg.hpp"

#define LINALG_BLOCKSIZE 256
#define AXPY_BLOCKSIZE 1024
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

linAlg_t::linAlg_t(occa::device& device_,
         settings_t& settings_, occa::properties& props_):
  device(device_), settings(settings_), props(props_), blocksize(LINALG_BLOCKSIZE) {};

//named cosntructor
linAlg_t& linAlg_t::Setup(occa::device& device_,
         settings_t& settings_, occa::properties& props_) {

  linAlg_t *linAlg = new linAlg_t(device_, settings_, props_);

  //pinned scratch buffer
  occa::properties mprops;
  mprops["host"] = true;
  linAlg->h_scratch = linAlg->device.malloc(LINALG_BLOCKSIZE*sizeof(dfloat), mprops);
  linAlg->scratch = (dfloat*) linAlg->h_scratch.ptr();

  linAlg->o_scratch = linAlg->device.malloc(LINALG_BLOCKSIZE*sizeof(dfloat));

  return *linAlg;
}

//initialize list of kernels
void linAlg_t::InitKernels(vector<string> kernels, MPI_Comm comm) {

  occa::properties kernelInfo = props; //copy base properties

  //add defines
  kernelInfo["defines/" "p_blockSize"] = (int)LINALG_BLOCKSIZE;

  for (size_t i=0;i<kernels.size();i++) {
    string name = kernels[i];
    if (name=="set") {
      if (setKernel.isInitialized()==false)
        setKernel = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgSet.okl",
                                        "set",
                                        kernelInfo, comm);
    } else if (name=="axpy") {
      occa::properties axpyKernelInfo = kernelInfo;
      if (device.mode()=="HIP") {
        axpyKernelInfo["compiler_flags"] += " --gpu-max-threads-per-block=" + std::string(TOSTRING(AXPY_BLOCKSIZE));
      }
      if (axpyKernel.isInitialized()==false)
        axpyKernel = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgAXPY.okl",
                                        "axpy",
                                        axpyKernelInfo, comm);
    } else if (name=="norm2") {
      if (norm2Kernel1.isInitialized()==false)
        norm2Kernel1 = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgNorm2.okl",
                                        "norm2_1",
                                        kernelInfo, comm);

      if (norm2Kernel2.isInitialized()==false)
        norm2Kernel2 = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgNorm2.okl",
                                        "norm2_2",
                                        kernelInfo, comm);
    } else if (name=="innerProd") {
      if (innerProdKernel1.isInitialized()==false)
        innerProdKernel1 = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgInnerProd.okl",
                                        "innerProd_1",
                                        kernelInfo, comm);

      if (innerProdKernel2.isInitialized()==false)
        innerProdKernel2 = buildKernel(device, HIPBONE_DIR "/core/okl/"
                                        "linAlgInnerProd.okl",
                                        "innerProd_2",
                                        kernelInfo, comm);
    } else {
      stringstream ss;
      ss << "Requested linAlg routine \"" << name << "\" not found";
      HIPBONE_ABORT(ss.str());
    }
  }
}

linAlg_t::~linAlg_t() {
  setKernel.free();
  axpyKernel.free();
  norm2Kernel1.free();
  norm2Kernel2.free();
  innerProdKernel1.free();
  innerProdKernel2.free();
}
