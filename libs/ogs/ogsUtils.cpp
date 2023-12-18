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

#include <limits>
#include "ogs.hpp"
#include "ogs/ogsOperator.hpp"
#include "ogs/ogsExchange.hpp"
#include "ogs/ogsUtils.hpp"
#include "parameters.hpp"

namespace libp {

namespace ogs {

stream_t ogsBase_t::dataStream;

kernel_t ogsOperator_t::gatherScatterKernel[4][4];
kernel_t ogsOperator_t::gatherKernel[4][4];
kernel_t ogsOperator_t::scatterKernel[4];

kernel_t ogsExchange_t::extractKernel[4];

//defaults
int gsblockSize = 256;
int gblockSize = 256;
int sblockSize = 256;

int gsNodesPerBlock = 512;
int gNodesPerBlock = 512;
int sNodesPerBlock = 512;

void InitializeParams(platform_t& platform,
                      comm_t& comm,
                      const bool verbose) {
  static bool isInitialized = false;

  if (isInitialized) return;

  parameters_t tuningParameters;

  std::string filename = platform.exePath() + "/json/ogs.json";

  properties_t keys;
  keys["mode"] = platform.device.mode();

  std::string arch = platform.device.arch();
  if (platform.device.mode()=="HIP") {
    arch = arch.substr(0,arch.find(":")); //For HIP mode, remove the stuff after the :
  }
  keys["arch"] = arch;

  std::string name = "ogsKernels.okl";

  if (verbose && comm.rank()==0) {
    std::cout << "Loading Tuning Parameters, looking for match for Name:'" << name << "', keys:" << tuningParameters.toString(keys) << std::endl;
  }

  tuningParameters.load(filename, comm);

  properties_t matchProps = tuningParameters.findProperties(name, keys);

  if (verbose && comm.rank()==0) {
    std::cout << "Found best match = " << tuningParameters.toString(matchProps) << std::endl;
  }

  /*Read blocksizes from properties*/
  gblockSize  = static_cast<int>(matchProps["props/G_BLOCKSIZE"]);
  sblockSize  = static_cast<int>(matchProps["props/S_BLOCKSIZE"]);
  gsblockSize = static_cast<int>(matchProps["props/GS_BLOCKSIZE"]);

  gNodesPerBlock  = static_cast<int>(matchProps["props/G_NODESPERBLOCK"]);
  sNodesPerBlock  = static_cast<int>(matchProps["props/S_NODESPERBLOCK"]);
  gsNodesPerBlock = static_cast<int>(matchProps["props/GS_NODESPERBLOCK"]);

  isInitialized = true;
}

void InitializeKernels(platform_t& platform, const Type type, const Op op) {

  //check if the gather kernel is initialized
  if (!ogsOperator_t::gatherKernel[type][op].isInitialized()) {

    properties_t kernelInfo = platform.props();

    kernelInfo["defines/GS_BLOCKSIZE"] = ogs::gsblockSize;
    kernelInfo["defines/G_BLOCKSIZE"]  = ogs::gblockSize;
    kernelInfo["defines/S_BLOCKSIZE"]  = ogs::sblockSize;

    kernelInfo["defines/G_NODESPERBLOCK"]  = ogs::gNodesPerBlock;
    kernelInfo["defines/S_NODESPERBLOCK"]  = ogs::sNodesPerBlock;
    kernelInfo["defines/GS_NODESPERBLOCK"] = ogs::gsNodesPerBlock;

    switch (type) {
      case Float:  kernelInfo["defines/T"] =  "float"; break;
      case Double: kernelInfo["defines/T"] =  "double"; break;
      case Int32:  kernelInfo["defines/T"] =  "int"; break;
      case Int64:  kernelInfo["defines/T"] =  "long long int"; break;
    }

    switch (type) {
      case Float:
        switch (op) {
          case Add: kernelInfo["defines/OGS_OP_INIT"] =  float{0}; break;
          case Mul: kernelInfo["defines/OGS_OP_INIT"] =  float{1}; break;
          case Min: kernelInfo["defines/OGS_OP_INIT"] =  std::numeric_limits<float>::max(); break;
          case Max: kernelInfo["defines/OGS_OP_INIT"] = -std::numeric_limits<float>::max(); break;
        }
        break;
      case Double:
        switch (op) {
          case Add: kernelInfo["defines/OGS_OP_INIT"] =  double{0}; break;
          case Mul: kernelInfo["defines/OGS_OP_INIT"] =  double{1}; break;
          case Min: kernelInfo["defines/OGS_OP_INIT"] =  std::numeric_limits<double>::max(); break;
          case Max: kernelInfo["defines/OGS_OP_INIT"] = -std::numeric_limits<double>::max(); break;
        }
        break;
      case Int32:
        switch (op) {
          case Add: kernelInfo["defines/OGS_OP_INIT"] =  int32_t{0}; break;
          case Mul: kernelInfo["defines/OGS_OP_INIT"] =  int32_t{1}; break;
          case Min: kernelInfo["defines/OGS_OP_INIT"] =  std::numeric_limits<int32_t>::max(); break;
          case Max: kernelInfo["defines/OGS_OP_INIT"] = -std::numeric_limits<int32_t>::max(); break;
        }
        break;
      case Int64:
        switch (op) {
          case Add: kernelInfo["defines/OGS_OP_INIT"] =  int64_t{0}; break;
          case Mul: kernelInfo["defines/OGS_OP_INIT"] =  int64_t{1}; break;
          case Min: kernelInfo["defines/OGS_OP_INIT"] =  std::numeric_limits<int64_t>::max(); break;
          case Max: kernelInfo["defines/OGS_OP_INIT"] = -std::numeric_limits<int64_t>::max(); break;
        }
        break;
    }

    switch (op) {
      case Add: kernelInfo["defines/OGS_OP(a,b)"] = "a+=b"; break;
      case Mul: kernelInfo["defines/OGS_OP(a,b)"] = "a*=b"; break;
      case Min: kernelInfo["defines/OGS_OP(a,b)"] = "if(b<a) a=b"; break;
      case Max: kernelInfo["defines/OGS_OP(a,b)"] = "if(b>a) a=b"; break;
    }

    ogsOperator_t::gatherScatterKernel[type][op] = platform.buildKernel("libs/ogs/okl/ogsKernels.okl",
                                                         "gatherScatter",
                                                         kernelInfo);


    ogsOperator_t::gatherKernel[type][op] = platform.buildKernel("libs/ogs/okl/ogsKernels.okl",
                                                "gather",
                                                kernelInfo);

    if (!ogsOperator_t::scatterKernel[type].isInitialized()) {
      ogsOperator_t::scatterKernel[type] = platform.buildKernel("libs/ogs/okl/ogsKernels.okl",
                                                 "scatter",
                                                 kernelInfo);

      ogsExchange_t::extractKernel[type] = platform.buildKernel("libs/ogs/okl/ogsKernels.okl",
                                                "extract", kernelInfo);
    }
  }
}

} //namespace ogs

} //namespace libp
