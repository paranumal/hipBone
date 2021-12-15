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
#include "ogs/ogsUtils.hpp"
#include "ogs/ogsOperator.hpp"


namespace ogs {

template<typename T>
struct Op_Add {
  const T init(){ return T{0}; }
  void operator()(T& gv, const T v) { gv += v; }
};
template<typename T>
struct Op_Mul {
  const T init(){ return T{1}; }
  void operator()(T& gv, const T v) { gv *= v; }
};
template<typename T>
struct Op_Max {
  const T init(){ return -std::numeric_limits<T>::max(); }
  void operator()(T& gv, const T v) { gv = (v>gv) ? v : gv; }
};
template<typename T>
struct Op_Min {
  const T init() {return  std::numeric_limits<T>::max(); }
  void operator()(T& gv, const T v) { gv = (v<gv) ? v : gv; }
};

/********************************
 * Gather Operation
 ********************************/
template <typename T, template<typename> class Op>
void ogsOperator_t::Gather(T* gv,
                           const T* v,
                           const int K,
                           const Transpose trans) {

  dlong Nrows;
  dlong *rowStarts, *colIds;
  if (trans==NoTrans) {
    Nrows = NrowsN;
    rowStarts = rowStartsN;
    colIds = colIdsN;
  } else {
    Nrows = NrowsT;
    rowStarts = rowStartsT;
    colIds = colIdsT;
  }

  #pragma omp parallel for
  for(dlong n=0;n<Nrows;++n){
    const dlong start = rowStarts[n];
    const dlong end   = rowStarts[n+1];

    for (int k=0;k<K;++k) {
      T val = Op<T>().init();
      for(dlong g=start;g<end;++g){
        Op<T>()(val, v[k+colIds[g]*K]);
      }
      gv[k+n*K] = val;
    }
  }
}

void ogsOperator_t::Gather(void* gv,
                            const void* v,
                            const int k,
                            const Type type,
                            const Op op,
                            const Transpose trans) {
  switch (op){
    case Add:
    switch (type){
      case Float:  Gather<float  , Op_Add>
                    (static_cast<float  *>(gv),
                     static_cast<const float  *>( v),
                     k, trans); break;
      case Double: Gather<double , Op_Add>
                    (static_cast<double  *>(gv),
                     static_cast<const double  *>( v),
                     k, trans); break;
      case Int32:  Gather<int32_t, Op_Add>
                    (static_cast<int32_t  *>(gv),
                     static_cast<const int32_t  *>( v),
                     k, trans); break;
      case Int64:  Gather<int64_t, Op_Add>
                    (static_cast<int64_t  *>(gv),
                     static_cast<const int64_t  *>( v),
                     k, trans); break;
    }
    break;
    case Mul:
    switch (type){
      case Float:  Gather<float  , Op_Mul>
                    (static_cast<float  *>(gv),
                     static_cast<const float  *>( v),
                     k, trans); break;
      case Double: Gather<double , Op_Mul>
                    (static_cast<double  *>(gv),
                     static_cast<const double  *>( v),
                     k, trans); break;
      case Int32:  Gather<int32_t, Op_Mul>
                    (static_cast<int32_t  *>(gv),
                     static_cast<const int32_t  *>( v),
                     k, trans); break;
      case Int64:  Gather<int64_t, Op_Mul>
                    (static_cast<int64_t  *>(gv),
                     static_cast<const int64_t  *>( v),
                     k, trans); break;
    }
    break;
    case Max:
    switch (type){
      case Float:  Gather<float  , Op_Max>
                    (static_cast<float  *>(gv),
                     static_cast<const float  *>( v),
                     k, trans); break;
      case Double: Gather<double , Op_Max>
                    (static_cast<double  *>(gv),
                     static_cast<const double  *>( v),
                     k, trans); break;
      case Int32:  Gather<int32_t, Op_Max>
                    (static_cast<int32_t  *>(gv),
                     static_cast<const int32_t  *>( v),
                     k, trans); break;
      case Int64:  Gather<int64_t, Op_Max>
                    (static_cast<int64_t  *>(gv),
                     static_cast<const int64_t  *>( v),
                     k, trans); break;
    }
    break;
    case Min:
    switch (type){
      case Float:  Gather<float  , Op_Min>
                    (static_cast<float  *>(gv),
                     static_cast<const float  *>( v),
                     k, trans); break;
      case Double: Gather<double , Op_Min>
                    (static_cast<double  *>(gv),
                     static_cast<const double  *>( v),
                     k, trans); break;
      case Int32:  Gather<int32_t, Op_Min>
                    (static_cast<int32_t  *>(gv),
                     static_cast<const int32_t  *>( v),
                     k, trans); break;
      case Int64:  Gather<int64_t, Op_Min>
                    (static_cast<int64_t  *>(gv),
                     static_cast<const int64_t  *>( v),
                     k, trans); break;
    }
    break;
  }
}

void ogsOperator_t::Gather(occa::memory&  o_gv,
                            occa::memory&  o_v,
                            const int k,
                            const Type type,
                            const Op op,
                            const Transpose trans) {
  InitializeKernels(platform, type, op);

  if (trans==NoTrans) {
    if (NrowBlocksN)
      gatherKernel[type][op](NrowBlocksN,
                              k,
                              o_blockRowStartsN,
                              o_rowStartsN,
                              o_colIdsN,
                              o_v,
                              o_gv);
  } else {
    if (NrowBlocksT)
      gatherKernel[type][op](NrowBlocksT,
                              k,
                              o_blockRowStartsT,
                              o_rowStartsT,
                              o_colIdsT,
                              o_v,
                              o_gv);
  }
}

/********************************
 * Scatter Operation
 ********************************/
template <typename T>
void ogsOperator_t::Scatter(T* v,
                            const T* gv,
                            const int K,
                            const Transpose trans) {

  dlong Nrows;
  dlong *rowStarts, *colIds;
  if (trans==Trans) {
    Nrows = NrowsN;
    rowStarts = rowStartsN;
    colIds = colIdsN;
  } else {
    Nrows = NrowsT;
    rowStarts = rowStartsT;
    colIds = colIdsT;
  }

  #pragma omp parallel for
  for(dlong n=0;n<Nrows;++n){
    const dlong start = rowStarts[n];
    const dlong end   = rowStarts[n+1];

    for(dlong g=start;g<end;++g){
      for (int k=0;k<K;++k) {
        v[k+colIds[g]*K] = gv[k+n*K];
      }
    }
  }
}

void ogsOperator_t::Scatter(void* v,
                             const void* gv,
                             const int k,
                             const Type type,
                             const Op op,
                             const Transpose trans) {
  switch (type){
    case Float:  Scatter<float  >(static_cast<float  *>(v),
                                  static_cast<const float  *>(gv),
                                  k, trans); break;
    case Double: Scatter<double >(static_cast<double *>(v),
                                  static_cast<const double *>(gv),
                                  k, trans); break;
    case Int32:  Scatter<int32_t>(static_cast<int32_t*>(v),
                                  static_cast<const int32_t*>(gv),
                                  k, trans); break;
    case Int64:  Scatter<int64_t>(static_cast<int64_t*>(v),
                                  static_cast<const int64_t*>(gv),
                                  k, trans); break;
  }
}

void ogsOperator_t::Scatter(occa::memory&  o_v,
                             occa::memory&  o_gv,
                             const int k,
                             const Type type,
                             const Op op,
                             const Transpose trans) {
  InitializeKernels(platform, type, Add);

  if (trans==Trans) {
    if (NrowBlocksN)
      scatterKernel[type](NrowBlocksN,
                          k,
                          o_blockRowStartsN,
                          o_rowStartsN,
                          o_colIdsN,
                          o_gv,
                          o_v);
  } else {
    if (NrowBlocksT)
      scatterKernel[type](NrowBlocksT,
                          k,
                          o_blockRowStartsT,
                          o_rowStartsT,
                          o_colIdsT,
                          o_gv,
                          o_v);
  }
}

/********************************
 * GatherScatter Operation
 ********************************/
template <typename T, template<typename> class Op>
void ogsOperator_t::GatherScatter(T* v,
                                  const int K,
                                  const Transpose trans) {

  dlong Nrows;
  dlong *gRowStarts, *gColIds;
  dlong *sRowStarts, *sColIds;

  if (trans==Trans) {
    Nrows = NrowsN;
    gRowStarts = rowStartsT;
    gColIds    = colIdsT;
    sRowStarts = rowStartsN;
    sColIds    = colIdsN;
  } else if (trans==Sym) {
    Nrows = NrowsT;
    gRowStarts = rowStartsT;
    gColIds    = colIdsT;
    sRowStarts = rowStartsT;
    sColIds    = colIdsT;
  } else {
    Nrows = NrowsT;
    gRowStarts = rowStartsN;
    gColIds    = colIdsN;
    sRowStarts = rowStartsT;
    sColIds    = colIdsT;
  }

  #pragma omp parallel for
  for(dlong n=0;n<Nrows;++n){
    const dlong gstart = gRowStarts[n];
    const dlong gend   = gRowStarts[n+1];
    const dlong sstart = sRowStarts[n];
    const dlong send   = sRowStarts[n+1];

    for (int k=0;k<K;++k) {
      T val = Op<T>().init();
      for(dlong g=gstart;g<gend;++g){
        Op<T>()(val, v[k+gColIds[g]*K]);
      }
      for(dlong s=sstart;s<send;++s){
        v[k+sColIds[s]*K] = val;
      }
    }
  }
}

void ogsOperator_t::GatherScatter(void* v,
                                  const int k,
                                  const Type type,
                                  const Op op,
                                  const Transpose trans) {
  switch (op){
    case Add:
    switch (type){
      case Float:  GatherScatter<float  , Op_Add>
                    (static_cast<float  *>(v), k, trans); break;
      case Double: GatherScatter<double , Op_Add>
                    (static_cast<double *>(v), k, trans); break;
      case Int32:  GatherScatter<int32_t, Op_Add>
                    (static_cast<int32_t*>(v), k, trans); break;
      case Int64:  GatherScatter<int64_t, Op_Add>
                    (static_cast<int64_t*>(v), k, trans); break;
    }
    break;
    case Mul:
    switch (type){
      case Float:  GatherScatter<float  , Op_Mul>
                    (static_cast<float  *>(v), k, trans); break;
      case Double: GatherScatter<double , Op_Mul>
                    (static_cast<double *>(v), k, trans); break;
      case Int32:  GatherScatter<int32_t, Op_Mul>
                    (static_cast<int32_t*>(v), k, trans); break;
      case Int64:  GatherScatter<int64_t, Op_Mul>
                    (static_cast<int64_t*>(v), k, trans); break;
    }
    break;
    case Max:
    switch (type){
      case Float:  GatherScatter<float  , Op_Max>
                    (static_cast<float  *>(v), k, trans); break;
      case Double: GatherScatter<double , Op_Max>
                    (static_cast<double *>(v), k, trans); break;
      case Int32:  GatherScatter<int32_t, Op_Max>
                    (static_cast<int32_t*>(v), k, trans); break;
      case Int64:  GatherScatter<int64_t, Op_Max>
                    (static_cast<int64_t*>(v), k, trans); break;
    }
    break;
    case Min:
    switch (type){
      case Float:  GatherScatter<float  , Op_Min>
                    (static_cast<float  *>(v), k, trans); break;
      case Double: GatherScatter<double , Op_Min>
                    (static_cast<double *>(v), k, trans); break;
      case Int32:  GatherScatter<int32_t, Op_Min>
                    (static_cast<int32_t*>(v), k, trans); break;
      case Int64:  GatherScatter<int64_t, Op_Min>
                    (static_cast<int64_t*>(v), k, trans); break;
    }
    break;
  }
}

void ogsOperator_t::GatherScatter(occa::memory&  o_v,
                                  const int k,
                                  const Type type,
                                  const Op op,
                                  const Transpose trans) {
  InitializeKernels(platform, type, Add);

  if (trans==Trans) {
    if (NrowBlocksT)
      gatherScatterKernel[type][Add](NrowBlocksT,
                                     k,
                                     o_blockRowStartsT,
                                     o_rowStartsT,
                                     o_colIdsT,
                                     o_rowStartsN,
                                     o_colIdsN,
                                     o_v);
  } else if (trans==Sym) {
    if (NrowBlocksT)
      gatherScatterKernel[type][Add](NrowBlocksT,
                                     k,
                                     o_blockRowStartsT,
                                     o_rowStartsT,
                                     o_colIdsT,
                                     o_rowStartsT,
                                     o_colIdsT,
                                     o_v);
  } else {
    if (NrowBlocksT)
      gatherScatterKernel[type][Add](NrowBlocksT,
                                     k,
                                     o_blockRowStartsT,
                                     o_rowStartsN,
                                     o_colIdsN,
                                     o_rowStartsT,
                                     o_colIdsT,
                                     o_v);
  }
}

void ogsOperator_t::setupRowBlocks() {

  dlong blockSumN=0, blockSumT=0;
  NrowBlocksN=0, NrowBlocksT=0;

  if (NrowsN) NrowBlocksN++;
  if (NrowsT) NrowBlocksT++;

  for (dlong i=0;i<NrowsT;i++) {
    const dlong rowSizeN  = rowStartsN[i+1]-rowStartsN[i];

    if (rowSizeN > ogs::gatherNodesPerBlock) {
      //this row is pathalogically big. We can't currently run this
      stringstream ss;
      ss << "Multiplicity of global node id: " << i
         << " in ogsOperator_t::setupRowBlocks is too large.";
      HIPBONE_ABORT(ss.str())
    }

    const dlong rowSizeT  = rowStartsT[i+1]-rowStartsT[i];

    if (rowSizeT > ogs::gatherNodesPerBlock) {
      //this row is pathalogically big. We can't currently run this
      stringstream ss;
      ss << "Multiplicity of global node id: " << i
         << " in ogsOperator_t::setupRowBlocks is too large.";
      HIPBONE_ABORT(ss.str())
    }

    if (blockSumN+rowSizeN > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      NrowBlocksN++; //count the previous block
      blockSumN=rowSizeN; //start a new row block
    } else {
      blockSumN+=rowSizeN; //add this row to the block
    }

    if (blockSumT+rowSizeT > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      NrowBlocksT++; //count the previous block
      blockSumT=rowSizeT; //start a new row block
    } else {
      blockSumT+=rowSizeT; //add this row to the block
    }
  }

  blockRowStartsN  = (dlong*) calloc(NrowBlocksN+1,sizeof(dlong));
  blockRowStartsT  = (dlong*) calloc(NrowBlocksT+1,sizeof(dlong));

  blockSumN=0, blockSumT=0;
  NrowBlocksN=0, NrowBlocksT=0;
  if (NrowsN) NrowBlocksN++;
  if (NrowsT) NrowBlocksT++;

  for (dlong i=0;i<NrowsT;i++) {
    const dlong rowSizeN  = rowStartsN[i+1]-rowStartsN[i];
    const dlong rowSizeT  = rowStartsT[i+1]-rowStartsT[i];

    if (blockSumN+rowSizeN > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      blockRowStartsN[NrowBlocksN++] = i; //mark the previous block
      blockSumN=rowSizeN; //start a new row block
    } else {
      blockSumN+=rowSizeN; //add this row to the block
    }
    if (blockSumT+rowSizeT > ogs::gatherNodesPerBlock) { //adding this row will exceed the nnz per block
      blockRowStartsT[NrowBlocksT++] = i; //mark the previous block
      blockSumT=rowSizeT; //start a new row block
    } else {
      blockSumT+=rowSizeT; //add this row to the block
    }
  }
  blockRowStartsN[NrowBlocksN] = NrowsT;
  blockRowStartsT[NrowBlocksT] = NrowsT;

  o_blockRowStartsN = platform.malloc((NrowBlocksN+1)*sizeof(dlong), blockRowStartsN);
  o_blockRowStartsT = platform.malloc((NrowBlocksT+1)*sizeof(dlong), blockRowStartsT);
}

void ogsOperator_t::Free() {
  if(rowStartsT) {free(rowStartsT); rowStartsT=nullptr;}
  if(colIdsT) {free(colIdsT); colIdsT=nullptr;}

  if(o_rowStartsT.size()) o_rowStartsT.free();
  if(o_colIdsT.size()) o_colIdsT.free();

  if (kind==Signed) {
    if(rowStartsN) {free(rowStartsN); rowStartsN=nullptr;}
    if(colIdsN) {free(colIdsN); colIdsN=nullptr;}

    if(o_rowStartsN.size()) o_rowStartsN.free();
    if(o_colIdsN.size()) o_colIdsN.free();
  } else {
    rowStartsN=nullptr;
    colIdsN=nullptr;
  }

  if(blockRowStartsT) {free(blockRowStartsT); blockRowStartsT=nullptr;}
  if(blockRowStartsN) {free(blockRowStartsN); blockRowStartsN=nullptr;}
  if(o_blockRowStartsN.size()) o_blockRowStartsN.free();
  if(o_blockRowStartsT.size()) o_blockRowStartsT.free();

  nnzN=0;
  nnzT=0;
  NrowsN=0;
  NrowsT=0;
  Ncols=0;
  NrowBlocksN=0;
  NrowBlocksT=0;
}


template<typename T>
void extract(const dlong N,
             const int K,
             const dlong *ids,
             const T *q,
             T *gatherq) {

  for(dlong n=0;n<N;++n){
    const dlong gid = ids[n];

    for (int k=0;k<K;++k) {
      gatherq[k+n*K] = q[k+gid*K];
    }
  }
}

void extract(const dlong N,
             const int K,
             const Type type,
             const dlong *ids,
             const void *q,
             void *gatherq) {
  switch (type){
    case Float:  extract<float  >(N, K, ids,
                                 static_cast<const float  *>(q),
                                 static_cast<float  *>(gatherq));
                 break;
    case Double: extract<double >(N, K, ids,
                                 static_cast<const double*>(q),
                                 static_cast<double*>(gatherq));
                 break;
    case Int32:  extract<int32_t>(N, K, ids,
                                 static_cast<const int32_t*>(q),
                                 static_cast<int32_t*>(gatherq));
                 break;
    case Int64:  extract<int64_t>(N, K, ids,
                                 static_cast<const int64_t*>(q),
                                 static_cast<int64_t*>(gatherq));
                 break;
  }
}

} //namespace ogs
