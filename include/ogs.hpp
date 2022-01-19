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

/*
  OCCA Gather/Scatter Library

  The code

    MPI_Comm comm;
  	dlong N;
    hlong id[N];    // the hlong and dlong types are defined in "types.h"
    bool verbose;
    bool unique;
    ogs_t ogs(platform);
    ...
    ogs.Setup(N, id, comm, ogs::Signed,
              ogs::Auto, unique, verbose);

  defines a partition of the set of (processor, local index) pairs,
    (p,i) \in S_j  iff   abs(id[i]) == j  on processor p
  That is, all (p,i) pairs are grouped together (in group S_j) that have the
    same id (=j).
  S_0 is treated specially --- it is ignored completely
    (i.e., when id[i] == 0, local index i does not participate in any
    gather/scatter operation)
  If id[i] on proc p is negative then the pair (p,i) is "flagged". This
  determines the non-symmetric behavior. For the simpler, symmetric case,
  ogs::Unsigned can be passed to the 'Kind' parameter, which
  treats all id's as positive.

  When "ogs" is no longer needed, free it with

    ogs.Free();

  A basic gatherScatter operation is, e.g.,

    occa::memory o_v;
    ...
    ogs.GatherScatter(o_v, 1, ogs::Double, ogs::Add, ogs::Sym);

  This gs call has the effect,

    o_v[i] <--  \sum_{ (p,j) \in S_{id[i]} } o_v_(p) [j]

  where o_v_(p) [j] means o_v[j] on proc p. In other words, every o_v[i] is replaced
  by the sum of all o_v[j]'s with the same id, given by id[i]. This accomplishes
  "direct stiffness summation" corresponding to the action of QQ^T, where
  "Q" is a boolean matrix that copies from a global vector (indexed by id)
  to the local vectors indexed by (p,i) pairs.

  Summation on doubles is not the only operation and datatype supported. Support
  includes the operations
    ogs::Add, ogs::Mul, ogs::Max, ogs::Min
  and datatypes
    ogs::Dfloat, ogs::Double, ogs::Float, ogs::Int32, ogs::Int64, ogs::Dlong, ogs::Hlong.
  (The int32 and int64 types are the normal C++ types, whereas dfloat, dlong, and hlong
   are defined in "types.h").

  For the nonsymmetric behavior, the "Transpose" parameter is important:

    ogs.GatherScatter(o_v, 1, ogs::Double, ogs::Add, ogs::[NoTrans/Trans/Sym]);

  When transpose == ogs::NoTrans, any "flagged" (p,i) pairs (id[i] negative on p)
  do not participate in the sum, but *do* still receive the sum on output.
  As a special case, when only one (p,i) pair is unflagged per group this
  corresponds to the rectangular "Q" matrix referred to above.

  When transpose == ogs::Trans, the "flagged" (p,i) pairs *do* participate in the sum,
  but do *not* get set on output. In the special case of only one unflagged
  (p,i) pair, this corresponds to the transpose of "Q" referred to above.

  When transpose == ogs::Sym, all ids are considered "unflagged". That is,
  the "flagged" (p,i) pairs *do* participate in the sum, and *do* get set
  on output.

  When the 'unique' parameter is passed as 'true', the setup call modifies ids,
  "flagging" (by negating id[i]) all (p,i) pairs in each group except one.
  The sole "unflagged" member of the group is chosen in an arbitrary but
  consistent way. When all groups of (p,i) pairs have a single "unflagged"
  pair in this mannor, an additional nonsymmetric operation is available:

    ogs.Gather(o_Gv, o_v, 1, ogs::Double, ogs::Add, ogs::Trans);

  this has the effect of "assembling" the vector o_Gv. That is

    o_Gv[gid[j]] <--  \sum_{ (p,j) \in S_{id[i]} } o_v_(p) [j]

  for some ordering gid. As with the GatherScatter operation, when
  Transpose == ogs::NoTrans, any "flagged" (p,i) pairs (id[i] negative on p)
  do not participate in the sum, otherwise the "flagged" (p,i) pairs *do*
  participate in the sum.

  The inverse of this operation is

    ogs.Scatter(o_v, o_Gv, 1, ogs::Double, ogs::Add, ogs::Trans);

  which has the effect of scattering in the assembled entries in o_Gv back to the
  orginal ordering. When Transpose == ogs::Trans, "flagged" (p,i) pairs (id[i]
  negative on p) do *not* recieve their corresponding entry from o_Gv, otherwise
  the "flagged" (p,i) pairs recieve an entry.

  For operating on contiguously packed vectors, the K parameter is used, e.g.,

    ogs.GatherScatter(o_v, 3, ogs::Double, ogs::Add, ogs::Sym);

  which is like "GatherScatter" operating on the datatype double[3],
  with summation here being vector summation. Number of messages sent
  is independent of k.

  Asynchronous versions of the various GatherScatter functions are provided by

    ogs.GatherScatterStart(o_v, k, ogs::Double, ogs::Add, ogs::Sym);
    ...
    ogs.GatherScatterFinish(o_v, k, ogs::Double, ogs::Add, ogs::Sym);

  MPI communication is not initiated in GatherScatterStart, rather some initial
  message packing and host<->device transfers are queued. The user can then queue
  their own local kernels to the device which overlapps with this work before
  calling GatherScatterFinish. The MPI communication will then take place while the
  user's local kernels execute to maximize the amount of communication hiding.

  Finally, a specialized communcation object, named halo_t is provided. This
  object is analogous to an ogs_t object, where each group S_j has a sole
  "unflagged" (p,i) pair, as discussed above regarding the 'unique' parameter,
  and furthermore each "unflagged" (p,i) pair has a unique label ids[i] on its
  process. That is, for each "unflagged" (p,i), there are no other, flagged or
  unflagged, pairs (p,j) on process p with the label ids[i].

  With this particular flagging of (p,i) pairs, simple exchange routines are
  defined:

    halo_t halo(platofrm);
    halo.Setup(N, ids, comm, ogs::Auto, verbose);
    halo.Exchange(o_v, k, ogs::Double);

  which has the effect of filling all "flagged" pairs (p,i) on all processes with
  the corresponding value from the unique "unflagged" pair in S_j.

  An additional untility operation available in the halo_t object is

    halo.Combine(o_v, k, ogs::Double);

  which has the effect of summing the entries in S_j and writing the result to
  the sole "unflagged" pair in S_j.

*/

#ifndef OGS_HPP
#define OGS_HPP

#include "core.hpp"
#include "platform.hpp"

namespace libp {

namespace ogs {

/* type enum */
typedef enum { Float, Double, Int32, Int64} Type;

constexpr Type Dfloat = (std::is_same<double, dfloat>::value)
                          ? Double : Float;
// constexpr Type Pfloat = (std::is_same<double, pfloat>::value)
//                           ? Double : Float;
constexpr Type Dlong  = (std::is_same<int32_t, dlong>::value)
                          ? Int32 : Int64;
constexpr Type Hlong  = (std::is_same<int32_t, hlong>::value)
                          ? Int32 : Int64;

/* operation enum */
typedef enum { Add, Mul, Max, Min} Op;

/* transpose switch */
typedef enum { Sym, NoTrans, Trans } Transpose;

/* method switch */
typedef enum { Auto, Pairwise, CrystalRouter, AllToAll} Method;

/* kind enum */
typedef enum { Unsigned, Signed, Halo} Kind;

} //namespace ogs

} //namespace libp

#include "ogs/ogsBase.hpp"

namespace libp {

namespace ogs {

//pre-build kernels
void InitializeKernels(platform_t& platform, const Type type, const Op op);

// OCCA Gather Scatter
class ogs_t : public ogsBase_t {
public:
  ogs_t()=default;
  ~ogs_t()=default;

  void Setup(const dlong _N,
             hlong *ids,
             MPI_Comm _comm,
             const Kind _kind,
             const Method method,
             const bool _unique,
             const bool verbose,
             platform_t& _platform);

  void SetupGlobalToLocalMapping(dlong *GlobalToLocal);

  // host versions
  void GatherScatter(void* v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  // Synchronous device buffer versions
  void GatherScatter(occa::memory&  o_v,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  // Asynchronous device buffer versions
  void GatherScatterStart (occa::memory&  o_v,
                           const int k,
                           const Type type,
                           const Op op,
                           const Transpose trans);
  void GatherScatterFinish(occa::memory&  o_v,
                           const int k,
                           const Type type,
                           const Op op,
                           const Transpose trans);

  // host versions
  void Gather(void* gv,
              const void* v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);
  // Synchronous device buffer versions
  void Gather(occa::memory&  o_gv,
              occa::memory&  o_v,
              const int k,
              const Type type,
              const Op op,
              const Transpose trans);
  // Asynchronous device buffer versions
  void GatherStart (occa::memory&  o_gv,
                    occa::memory&  o_v,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans);
  void GatherFinish(occa::memory&  o_gv,
                    occa::memory&  o_v,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans);

  // host versions
  void Scatter(void* v,
               const void* gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);
  // Synchronous device buffer versions
  void Scatter(occa::memory&  o_v,
               occa::memory&  o_gv,
               const int k,
               const Type type,
               const Op op,
               const Transpose trans);
  // Asynchronous device buffer versions
  void ScatterStart (occa::memory&  o_v,
                     occa::memory&  o_gv,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);
  void ScatterFinish(occa::memory&  o_v,
                     occa::memory&  o_gv,
                     const int k,
                     const Type type,
                     const Op op,
                     const Transpose trans);

  friend class halo_t;
};

// OCCA Halo
class halo_t : public ogsBase_t {
public:
  halo_t()=default;
  ~halo_t()=default;

  bool gathered_halo=false;
  dlong Nhalo=0;

  void Setup(const dlong _N,
             hlong *ids,
             MPI_Comm _comm,
             const Method method,
             const bool verbose,
             platform_t& _platform);

  void SetupFromGather(ogs_t& ogs);

  // Host version
  void Exchange(void  *v, const int k, const Type type);
  // Synchronous device buffer version
  void Exchange(occa::memory &o_v, const int k, const Type type);
  // Asynchronous device buffer version
  void ExchangeStart (occa::memory &o_v, const int k, const Type type);
  void ExchangeFinish(occa::memory &o_v, const int k, const Type type);

  // Host version
  void Combine(void  *v, const int k, const Type type);
  // Synchronous device buffer version
  void Combine(occa::memory &o_v, const int k, const Type type);
  // Asynchronous device buffer version
  void CombineStart (occa::memory &o_v, const int k, const Type type);
  void CombineFinish(occa::memory &o_v, const int k, const Type type);
};

} //namespace ogs
} //namespace libp
#endif
