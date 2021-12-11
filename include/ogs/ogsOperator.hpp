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

#ifndef OGS_OPERATOR_HPP
#define OGS_OPERATOR_HPP

#include "ogs.hpp"

namespace libp {

namespace ogs {

// The Z operator class is essentially a sparse CSR matrix,
// with no vals stored. By construction, the sparse
// matrix will have at most 1 non-zero per column.
class ogsOperator_t {
public:
  platform_t& platform;

  dlong Ncols=0;
  dlong NrowsN=0;
  dlong NrowsT=0;
  dlong nnzN=0;
  dlong nnzT=0;
  dlong *rowStartsN=nullptr;
  dlong *rowStartsT=nullptr;
  dlong *colIdsN=nullptr;
  dlong *colIdsT=nullptr;

  occa::memory o_rowStartsN;
  occa::memory o_rowStartsT;
  occa::memory o_colIdsN;
  occa::memory o_colIdsT;

  dlong NrowBlocksN=0;
  dlong NrowBlocksT=0;
  dlong *blockRowStartsN=nullptr;
  dlong *blockRowStartsT=nullptr;
  occa::memory o_blockRowStartsN;
  occa::memory o_blockRowStartsT;

  Kind kind;

  ogsOperator_t(platform_t& _platform)
   : platform(_platform) {};

  void Free();
  ~ogsOperator_t() {Free();}

  void setupRowBlocks();

  //Apply Z operator
  void Gather(occa::memory&  o_gv,
              occa::memory&  o_v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);
  void Gather(void* gv,
              const void* v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);

  //Apply Z^T transpose operator
  void Scatter(occa::memory&  o_v,
               occa::memory&  o_gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);
  void Scatter(void* v,
               const void* gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);

  //Apply Z^T*Z operator
  void GatherScatter(occa::memory&  o_v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  void GatherScatter(void* v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);

private:
  template <typename T, template<typename> class Op>
  void Gather(T* gv, const T* v,
              const int K, const Transpose trans);
  template <typename T>
  void Scatter(T* v, const T* gv,
              const int K, const Transpose trans);
  template <typename T, template<typename> class Op>
  void GatherScatter(T* v,
              const int K, const Transpose trans);
};

template<typename T>
void extract(const dlong N,
             const int K,
             const dlong *ids,
             const T *q,
             T *gatherq);

void extract(const dlong N,
             const int K,
             const Type type,
             const dlong *ids,
             const void *q,
             void *gatherq);

} //namespace ogs

} //namespace libp

#endif
