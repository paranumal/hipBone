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

#include "ogs.hpp"
#include "ogs/ogsUtils.hpp"
#include "ogs/ogsOperator.hpp"
#include "ogs/ogsExchange.hpp"

namespace libp {

namespace ogs {

/********************************
 * Scalar GatherScatter
 ********************************/
void ogs_t::GatherScatter(occa::memory&  o_v,
                          const int k,
                          const Type type,
                          const Op op,
                          const Transpose trans){
  GatherScatterStart (o_v, k, type, op, trans);
  GatherScatterFinish(o_v, k, type, op, trans);
}

void ogs_t::GatherScatterStart(occa::memory& o_v,
                               const int k,
                               const Type type,
                               const Op op,
                               const Transpose trans){
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  gatherHalo->Gather(exchange->o_haloBuf, o_v,
                     k, type, op, trans);

  //prepare MPI exchange
  exchange->Start(k, type, op, trans);
}

void ogs_t::GatherScatterFinish(occa::memory& o_v,
                                const int k,
                                const Type type,
                                const Op op,
                                const Transpose trans){

  //queue local gs operation
  gatherLocal->GatherScatter(o_v, k, type, op, trans);

  //finish MPI exchange
  exchange->Finish(k, type, op, trans);

  //write exchanged halo buffer back to vector
  gatherHalo->Scatter(o_v, exchange->o_haloBuf,
                      k, type, op, trans);
}

void ogs_t::GatherScatter(void* v,
                          const int k,
                          const Type type,
                          const Op op,
                          const Transpose trans){
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  gatherHalo->Gather(exchange->haloBuf, v,
                     k, type, op, trans);

  //prepare MPI exchange
  exchange->Start(k, type, op, trans, true);

  //queue local gs operation
  gatherLocal->GatherScatter(v, k, type, op, trans);

  //finish MPI exchange
  exchange->Finish(k, type, op, trans, true);

  //write exchanged halo buffer back to vector
  gatherHalo->Scatter(v, exchange->haloBuf,
                      k, type, op, trans);
}

/********************************
 * Scalar Gather
 ********************************/
void ogs_t::Gather(occa::memory&  o_gv,
                   occa::memory&  o_v,
                   const int k,
                   const Type type,
                   const Op op,
                   const Transpose trans){
  GatherStart (o_gv, o_v, k, type, op, trans);
  GatherFinish(o_gv, o_v, k, type, op, trans);
}

void ogs_t::GatherStart(occa::memory&  o_gv,
                        occa::memory&  o_v,
                        const int k,
                        const Type type,
                        const Op op,
                        const Transpose trans){
  AssertGatherDefined();

  if (trans==Trans) { //if trans!=ogs::Trans theres no comms required
    exchange->AllocBuffer(k*Sizeof(type));

    //collect halo buffer
    gatherHalo->Gather(exchange->o_haloBuf, o_v,
                       k, type, op, Trans);

    //prepare MPI exchange
    exchange->Start(k, type, op, Trans);
  } else {
    //gather halo
    occa::memory o_gvHalo = o_gv + k*NlocalT*Sizeof(type);
    gatherHalo->Gather(o_gvHalo, o_v,
                       k, type, op, trans);
  }
}

void ogs_t::GatherFinish(occa::memory&  o_gv,
                         occa::memory&  o_v,
                         const int k,
                         const Type type,
                         const Op op,
                         const Transpose trans){
  AssertGatherDefined();

  //queue local g operation
  gatherLocal->Gather(o_gv, o_v,
                      k, type, op, trans);

  if (trans==Trans) { //if trans!=ogs::Trans theres no comms required
    //finish MPI exchange
    exchange->Finish(k, type, op, Trans);

    //put the result at the end of o_gv
    if (NhaloP)
      exchange->o_haloBuf.copyTo(o_gv + k*NlocalT*Sizeof(type),
                                 k*NhaloP*Sizeof(type), 0, 0, "async: true");
  }
}

//host versions
void ogs_t::Gather(void*  gv,
                   const void*  v,
                   const int k,
                   const Type type,
                   const Op op,
                   const Transpose trans){
  AssertGatherDefined();

  if (trans==Trans) { //if trans!=ogs::Trans theres no comms required
    exchange->AllocBuffer(k*Sizeof(type));

    //collect halo buffer
    gatherHalo->Gather(exchange->haloBuf, v,
                       k, type, op, Trans);

    //prepare MPI exchange
    exchange->Start(k, type, op, Trans, true);

    //local gather
    gatherLocal->Gather(gv, v, k, type, op, Trans);

    //finish MPI exchange
    exchange->Finish(k, type, op, Trans, true);

    //put the result at the end of o_gv
    std::memcpy(static_cast<char*>(gv) + k*NlocalT*Sizeof(type),
                exchange->haloBuf,
                k*NhaloP*Sizeof(type));
  } else {
    //local gather
    gatherLocal->Gather(gv, v, k, type, op, trans);

    //gather halo
    gatherHalo->Gather(static_cast<char*>(gv) + k*NlocalT*Sizeof(type),
                       v, k, type, op, trans);
  }
}


/********************************
 * Scalar Scatter
 ********************************/
void ogs_t::Scatter(occa::memory&  o_v,
                    occa::memory&  o_gv,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans){
  ScatterStart (o_v, o_gv, k, type, op, trans);
  ScatterFinish(o_v, o_gv, k, type, op, trans);
}

void ogs_t::ScatterStart(occa::memory&  o_v,
                         occa::memory&  o_gv,
                         const int k,
                         const Type type,
                         const Op op,
                         const Transpose trans){
  AssertGatherDefined();

  if (trans==NoTrans) { //if trans!=ogs::NoTrans theres no comms required
    exchange->AllocBuffer(k*Sizeof(type));

    //collect halo buffer
    if (NhaloP)
      exchange->o_haloBuf.copyFrom(o_gv + k*NlocalT*Sizeof(type),
                                   k*NhaloP*Sizeof(type), 0, 0, "async: true");

    //prepare MPI exchange
    exchange->Start(k, type, op, NoTrans);
  }
}


void ogs_t::ScatterFinish(occa::memory&  o_v,
                          occa::memory&  o_gv,
                          const int k,
                          const Type type,
                          const Op op,
                          const Transpose trans){
  AssertGatherDefined();

  //queue local s operation
  gatherLocal->Scatter(o_v, o_gv, k, type, op, trans);

  if (trans==NoTrans) { //if trans!=ogs::NoTrans theres no comms required
    //finish MPI exchange (and put the result at the end of o_gv)
    exchange->Finish(k, type, op, NoTrans);

    //scatter halo buffer
    gatherHalo->Scatter(o_v, exchange->o_haloBuf,
                        k, type, op, NoTrans);
  } else {
    //scatter halo
    occa::memory o_gvHalo = o_gv + k*NlocalT*Sizeof(type);
    gatherHalo->Scatter(o_v, o_gvHalo, k, type, op, trans);
  }
}

//host versions
void ogs_t::Scatter(void*  v,
                    const void*  gv,
                    const int k,
                    const Type type,
                    const Op op,
                    const Transpose trans){
  AssertGatherDefined();

  if (trans==NoTrans) { //if trans!=ogs::NoTrans theres no comms required
    exchange->AllocBuffer(k*Sizeof(type));

    //collect halo buffer
    std::memcpy(exchange->haloBuf,
                static_cast<const char*>(gv) + k*NlocalT*Sizeof(type),
                k*NhaloP*Sizeof(type));

    //prepare MPI exchange
    exchange->Start(k, type, op, NoTrans, true);

    //local scatter
    gatherLocal->Scatter(v, gv, k, type, op, NoTrans);

    //finish MPI exchange (and put the result at the end of o_gv)
    exchange->Finish(k, type, op, NoTrans, true);

    //scatter halo buffer
    gatherHalo->Scatter(v, exchange->haloBuf,
                        k, type, op, NoTrans);
  } else {
    //local scatter
    gatherLocal->Scatter(v, gv, k, type, op, trans);

    //scatter halo
    gatherHalo->Scatter(v,
                        static_cast<const char*>(gv)
                            + k*NlocalT*Sizeof(type),
                        k, type, op, trans);
  }
}

} //namespace ogs

} //namespace libp
