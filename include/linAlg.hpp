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

#ifndef LINALG_HPP
#define LINALG_HPP

#include "core.hpp"
#include "memory.hpp"

namespace libp {

class platform_t;

//launcher for basic linear algebra OCCA kernels
class linAlg_t {

public:
  platform_t *platform;
  properties_t kernelInfo;

  int blocksize;

  //scratch space for reductions
  deviceMemory<dfloat> o_scratch;
  pinnedMemory<dfloat> h_scratch;

  linAlg_t(platform_t *_platform);

  //initialize list of kernels
  void InitKernels(std::vector<std::string> kernels);

  /*********************/
  /* vector operations */
  /*********************/

  // o_x[n] = alpha
  void set(const dlong N, const dfloat alpha, deviceMemory<dfloat> o_x);

  // o_y[n] = beta*o_y[n] + alpha*o_x[n]
  void axpy(const dlong N, const dfloat alpha, deviceMemory<dfloat> o_x,
                           const dfloat beta,  deviceMemory<dfloat> o_y);

  // ||o_a||_2
  dfloat norm2(const dlong N, deviceMemory<dfloat> o_a, comm_t comm);

  // o_x.o_y
  dfloat innerProd(const dlong N,
                   deviceMemory<dfloat> o_x,
                   deviceMemory<dfloat> o_y,
                   comm_t comm);

  kernel_t setKernel;
  kernel_t axpyKernel;
  kernel_t norm2Kernel1;
  kernel_t norm2Kernel2;
  kernel_t innerProdKernel1;
  kernel_t innerProdKernel2;

  static void matrixRightSolve(int NrowsA, int NcolsA, double *A, int NrowsB, int NcolsB, double *B, double *C);
  static void matrixRightSolve(int NrowsA, int NcolsA, float *A, int NrowsB, int NcolsB, float *B, float *C);

  static void matrixEigenVectors(int N, double *A, double *VR, double *WR, double *WI);
  static void matrixEigenVectors(int N, float *A, float *VR, float *WR, float *WI);

  static void matrixEigenValues(int N, double *A, double *WR, double *WI);
  static void matrixEigenValues(int N, float *A, float *WR, float *WI);

  static void matrixInverse(int N, double *A);
  static void matrixInverse(int N, float *A);
};

} //namespace libp

#endif
