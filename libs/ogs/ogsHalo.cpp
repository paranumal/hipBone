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

namespace ogs {

/********************************
 * Exchange
 ********************************/
void halo_t::Exchange(occa::memory& o_v,
                      const int k,
                      const Type type) {
  ExchangeStart (o_v, k, type);
  ExchangeFinish(o_v, k, type);
}

void halo_t::ExchangeStart(occa::memory& o_v,
                           const int k,
                           const Type type){
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    if (NhaloP)
      exchange->o_haloBuf.copyFrom(o_v + k*NlocalT*Sizeof(type),
                                   k*NhaloP*Sizeof(type), 0, 0, "async: true");
  } else {
    gatherHalo->Gather(exchange->o_haloBuf, o_v,
                        k, type, Add, NoTrans);
  }

  //prepare MPI exchange
  exchange->Start(k, type, Add, NoTrans);
}

void halo_t::ExchangeFinish(occa::memory& o_v,
                            const int k,
                            const Type type){

  //finish MPI exchange
  exchange->Finish(k, type, Add, NoTrans);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    if (NhaloP)
      exchange->o_haloBuf.copyTo(o_v + k*(NlocalT+NhaloP)*Sizeof(type),
                                 k*Nhalo*Sizeof(type),
                                 0, k*NhaloP*Sizeof(type), "async: true");
  } else {
    gatherHalo->Scatter(o_v, exchange->o_haloBuf,
                        k, type, Add, NoTrans);
  }
}

//host version
void halo_t::Exchange(void* v,
                      const int k,
                      const Type type) {
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    std::memcpy(exchange->haloBuf,
                static_cast<char*>(v) + k*NlocalT*Sizeof(type),
                k*NhaloP*Sizeof(type));
  } else {
    gatherHalo->Gather(exchange->haloBuf, v,
                       k, type, Add, NoTrans);
  }

  //MPI exchange
  exchange->Start (k, type, Add, NoTrans, true);
  exchange->Finish(k, type, Add, NoTrans, true);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    std::memcpy(static_cast<char*>(v) + k*(NlocalT+NhaloP)*Sizeof(type),
                static_cast<char*>(exchange->haloBuf)
                                + k*NhaloP*Sizeof(type),
                k*Nhalo*Sizeof(type));
  } else {
    gatherHalo->Scatter(v, exchange->haloBuf,
                        k, type, Add, NoTrans);
  }
}

/********************************
 * Combine
 ********************************/
void halo_t::Combine(occa::memory& o_v,
                     const int k,
                     const Type type) {
  CombineStart (o_v, k, type);
  CombineFinish(o_v, k, type);
}

void halo_t::CombineStart(occa::memory& o_v,
                          const int k,
                          const Type type){
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    if (NhaloT)
      exchange->o_haloBuf.copyFrom(o_v + k*NlocalT*Sizeof(type),
                                   k*NhaloT*Sizeof(type), 0, 0, "async: true");
  } else {
    gatherHalo->Gather(exchange->o_haloBuf, o_v,
                       k, type, Add, Trans);
  }

  //prepare MPI exchange
  exchange->Start(k, type, Add, Trans);
}


void halo_t::CombineFinish(occa::memory& o_v,
                           const int k,
                           const Type type){

  //finish MPI exchange
  exchange->Finish(k, type, Add, Trans);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    if (NhaloP)
      exchange->o_haloBuf.copyTo(o_v + k*NlocalT*Sizeof(type),
                                 k*NhaloP*Sizeof(type),
                                 0, 0, "async: true");
  } else {
    gatherHalo->Scatter(o_v, exchange->o_haloBuf,
                        k, type, Add, Trans);
  }
}

//host version
void halo_t::Combine(void* v,
                     const int k,
                     const Type type) {
  exchange->AllocBuffer(k*Sizeof(type));

  //collect halo buffer
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    std::memcpy(exchange->haloBuf,
                static_cast<char*>(v) + k*NlocalT*Sizeof(type),
                k*NhaloT*Sizeof(type));
  } else {
    gatherHalo->Gather(exchange->haloBuf, v,
                       k, type, Add, Trans);
  }

  //MPI exchange
  exchange->Start (k, type, Add, Trans, true);
  exchange->Finish(k, type, Add, Trans, true);

  //write exchanged halo buffer back to vector
  if (gathered_halo) {
    //if this halo was build from a gathered ogs the halo nodes are at the end
    std::memcpy(static_cast<char*>(v) + k*NlocalT*Sizeof(type),
                exchange->haloBuf,
                k*NhaloP*Sizeof(type));
  } else {
    gatherHalo->Scatter(v, exchange->haloBuf,
                        k, type, Add, Trans);
  }
}

} //namespace ogs