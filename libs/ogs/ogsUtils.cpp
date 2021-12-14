/*

The MIT License (MIT)

Copyright (c) 2017-2021 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

namespace libp {

namespace ogs {

MPI_Datatype MPI_PARALLELNODE_T;

void InitMPIType() {
  // Make the MPI_PARALLELNODE_T data type
  parallelNode_t node{};
  MPI_Datatype dtype[6] = {MPI_DLONG, MPI_HLONG,
                           MPI_DLONG, MPI_INT,
                           MPI_INT, MPI_INT};
  int blength[6] = {1, 1, 1, 1, 1, 1};
  MPI_Aint addr[6], displ[6];
  MPI_Get_address ( &(node.localId), addr+0);
  MPI_Get_address ( &(node.baseId), addr+1);
  MPI_Get_address ( &(node.newId), addr+2);
  MPI_Get_address ( &(node.sign), addr+3);
  MPI_Get_address ( &(node.rank), addr+4);
  MPI_Get_address ( &(node.destRank), addr+5);
  displ[0] = 0;
  displ[1] = addr[1] - addr[0];
  displ[2] = addr[2] - addr[0];
  displ[3] = addr[3] - addr[0];
  displ[4] = addr[4] - addr[0];
  displ[5] = addr[5] - addr[0];
  MPI_Type_create_struct (6, blength, displ, dtype, &MPI_PARALLELNODE_T);
  MPI_Type_commit (&MPI_PARALLELNODE_T);
}

void DestroyMPIType() {
  MPI_Type_free(&MPI_PARALLELNODE_T);
}

occa::kernel ogsOperator_t::gatherScatterKernel[4][4];
occa::kernel ogsOperator_t::gatherKernel[4][4];
occa::kernel ogsOperator_t::scatterKernel[4];

occa::kernel ogsExchange_t::extractKernel[4];
occa::stream ogsExchange_t::dataStream;


void InitializeKernels(platform_t& platform, const Type type, const Op op) {

  //check if the gather kernel is initialized
  if (!ogsOperator_t::gatherKernel[type][op].isInitialized()) {

    occa::properties kernelInfo = platform.props();

    kernelInfo["defines/p_blockSize"] = ogsOperator_t::blockSize;
    kernelInfo["defines/p_gatherNodesPerBlock"] = ogsOperator_t::gatherNodesPerBlock;

    switch (type) {
      case Float:  kernelInfo["defines/T"] =  "float"; break;
      case Double: kernelInfo["defines/T"] =  "double"; break;
      case Int32:  kernelInfo["defines/T"] =  "int32_t"; break;
      case Int64:  kernelInfo["defines/T"] =  "int64_t"; break;
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

    ogsOperator_t::gatherScatterKernel[type][op] = platform.buildKernel(OGS_DIR "/okl/ogsKernels.okl",
                                                         "gatherScatter",
                                                         kernelInfo);


    ogsOperator_t::gatherKernel[type][op] = platform.buildKernel(OGS_DIR "/okl/ogsKernels.okl",
                                                "gather",
                                                kernelInfo);

    if (!ogsOperator_t::scatterKernel[type].isInitialized()) {
      ogsOperator_t::scatterKernel[type] = platform.buildKernel(OGS_DIR "/okl/ogsKernels.okl",
                                                 "scatter",
                                                 kernelInfo);

      ogsExchange_t::extractKernel[type] = platform.buildKernel(OGS_DIR "/okl/ogsKernels.okl",
                                                "extract", kernelInfo);\
    }
  }
}

size_t Sizeof(const Type type) {
  switch(type) {
    case  Float: return sizeof(float);
    case Double: return sizeof(double);
    case  Int32: return sizeof(int32_t);
    case  Int64: return sizeof(int64_t);
  }
  return 0;
}

MPI_Datatype MPI_Type(const Type type) {
  switch(type) {
    case  Float: return MPI_FLOAT;
    case Double: return MPI_DOUBLE;
    case  Int32: return MPI_INT32_T;
    case  Int64: return MPI_INT64_T;
  }
  return 0;
}

} //namespace ogs

} //namespace libp
