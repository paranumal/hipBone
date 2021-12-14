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
#include "ogs/ogsExchange.hpp"

namespace libp {

namespace ogs {

void ogsPairwise_t::Start(const int k,
                          const Type type,
                          const Op op,
                          const Transpose trans,
                          const bool host){

  occa::device &device = platform.device;

  //get current stream
  occa::stream currentStream = device.getStream();

  const dlong Nsend = (trans == NoTrans) ? NsendN : NsendT;
  const dlong N     = (trans == NoTrans) ? NhaloP : Nhalo;

  if (Nsend) {
    if (gpu_aware && !host) {
      //if using gpu-aware mpi and exchanging device buffers,
      //  assemble the send buffer on device
      if (trans == NoTrans) {
        extractKernel[type](NsendN, k, o_sendIdsN, o_haloBuf, o_sendBuf);
      } else {
        extractKernel[type](NsendT, k, o_sendIdsT, o_haloBuf, o_sendBuf);
      }
      //if not overlapping, wait for kernel to finish on default stream
      device.finish();
    } else if (!host) {
      //if not using gpu-aware mpi and exchanging device buffers,
      // move the halo buffer to the host
      device.setStream(dataStream);
      const size_t Nbytes = k*Sizeof(type);
      o_haloBuf.copyTo(haloBuf, N*Nbytes, 0, "async: true");
      device.setStream(currentStream);
    }
  }
}


void ogsPairwise_t::Finish(const int k,
                           const Type type,
                           const Op op,
                           const Transpose trans,
                           const bool host){

  const size_t Nbytes = k*Sizeof(type);
  occa::device &device = platform.device;

  //get current stream
  occa::stream currentStream = device.getStream();

  const dlong Nsend = (trans == NoTrans) ? NsendN : NsendT;

  if (Nsend && !gpu_aware && !host) {
    //synchronize data stream to ensure the host buffer is on the host
    device.setStream(dataStream);
    device.finish();
    device.setStream(dataStream);
  }


  char *sendPtr, *recvPtr;
  if (gpu_aware && !host) { //device pointer
    sendPtr = (char*)o_sendBuf.ptr();
    recvPtr = (char*)o_haloBuf.ptr() + Nhalo*Nbytes;
  } else { //host pointer
    sendPtr = sendBuf;
    recvPtr = haloBuf + Nhalo*Nbytes;
  }

  const int NranksSend  = (trans==NoTrans) ? NranksSendN  : NranksSendT;
  const int NranksRecv  = (trans==NoTrans) ? NranksRecvN  : NranksRecvT;
  const int *sendRanks  = (trans==NoTrans) ? sendRanksN.ptr()   : sendRanksT.ptr();
  const int *recvRanks  = (trans==NoTrans) ? recvRanksN.ptr()   : recvRanksT.ptr();
  const int *sendCounts = (trans==NoTrans) ? sendCountsN.ptr()  : sendCountsT.ptr();
  const int *recvCounts = (trans==NoTrans) ? recvCountsN.ptr()  : recvCountsT.ptr();
  const int *sendOffsets= (trans==NoTrans) ? sendOffsetsN.ptr() : sendOffsetsT.ptr();
  const int *recvOffsets= (trans==NoTrans) ? recvOffsetsN.ptr() : recvOffsetsT.ptr();

  //post recvs
  for (int r=0;r<NranksRecv;r++) {
    MPI_Irecv(recvPtr+recvOffsets[r]*Nbytes,
              k*recvCounts[r], MPI_Type(type), recvRanks[r],
              recvRanks[r], comm, requests.ptr()+r);
  }

  //if the halo data is on the host, extract the send buffer
  if (host || !gpu_aware) {
    if (trans == NoTrans)
      extract(NsendN, k, type, sendIdsN.ptr(), haloBuf, sendBuf);
    else
      extract(NsendT, k, type, sendIdsT.ptr(), haloBuf, sendBuf);
  }

  //post sends
  for (int r=0;r<NranksSend;r++) {
    MPI_Isend(sendPtr+sendOffsets[r]*Nbytes,
              k*sendCounts[r], MPI_Type(type), sendRanks[r],
              rank, comm, requests.ptr()+NranksRecv+r);
  }
  MPI_Waitall(NranksRecv+NranksSend, requests.ptr(), statuses.ptr());

  //if we recvieved anything via MPI, gather the recv buffer and scatter
  // it back to to original vector
  dlong Nrecv = recvOffsets[NranksRecv];
  if (Nrecv) {
    if (!gpu_aware || host) {
      //if not gpu-aware or recieved data is on the host,
      // gather the recieved nodes
      postmpi.Gather(haloBuf, haloBuf,
                      k, type, op, trans);

      if (!host) {
        // copy recv back to device
        device.setStream(dataStream);
        const dlong N = (trans == Trans) ? NhaloP : Nhalo;
        o_haloBuf.copyFrom(haloBuf, N*Nbytes, 0, "async: true");
        device.finish(); //wait for transfer to finish
        device.setStream(currentStream);
      }
    } else {
      // gather the recieved nodes on device
      postmpi.Gather(o_haloBuf, o_haloBuf,
                      k, type, op, trans);
    }
  }
}

ogsPairwise_t::ogsPairwise_t(dlong Nshared,
                             libp::memory<parallelNode_t> &sharedNodes,
                             ogsOperator_t& gatherHalo,
                             MPI_Comm _comm,
                             platform_t &_platform):
  ogsExchange_t(_platform,_comm) {

  Nhalo  = gatherHalo.NrowsT;
  NhaloP = gatherHalo.NrowsN;

  // sort the list by rank to the order where they will be sent by MPI_Allgatherv
  std::sort(sharedNodes.ptr(), sharedNodes.ptr()+Nshared,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(a.rank < b.rank) return true; //group by rank
              if(a.rank > b.rank) return false;

              return a.newId < b.newId; //then order by the localId relative to this rank
            });

  //make mpi allgatherv counts and offsets
  libp::memory<int> mpiSendCountsT(size,0);
  libp::memory<int> mpiSendCountsN(size,0);
  libp::memory<int> mpiRecvCountsT(size);
  libp::memory<int> mpiRecvCountsN(size);
  libp::memory<int> mpiSendOffsetsT(size+1);
  libp::memory<int> mpiSendOffsetsN(size+1);
  libp::memory<int> mpiRecvOffsetsT(size+1);
  libp::memory<int> mpiRecvOffsetsN(size+1);

  for (dlong n=0;n<Nshared;n++) { //loop through nodes we need to send
    const int r = sharedNodes[n].rank;
    if (sharedNodes[n].sign>0) mpiSendCountsN[r]++;
    mpiSendCountsT[r]++;
  }

  //shared counts
  MPI_Alltoall(mpiSendCountsT.ptr(), 1, MPI_INT,
               mpiRecvCountsT.ptr(), 1, MPI_INT, comm);
  MPI_Alltoall(mpiSendCountsN.ptr(), 1, MPI_INT,
               mpiRecvCountsN.ptr(), 1, MPI_INT, comm);

  //cumulative sum
  mpiSendOffsetsN[0] = 0;
  mpiSendOffsetsT[0] = 0;
  mpiRecvOffsetsN[0] = 0;
  mpiRecvOffsetsT[0] = 0;
  for (int r=0;r<size;r++) {
    mpiSendOffsetsN[r+1] = mpiSendOffsetsN[r]+mpiSendCountsN[r];
    mpiSendOffsetsT[r+1] = mpiSendOffsetsT[r]+mpiSendCountsT[r];
    mpiRecvOffsetsN[r+1] = mpiRecvOffsetsN[r]+mpiRecvCountsN[r];
    mpiRecvOffsetsT[r+1] = mpiRecvOffsetsT[r]+mpiRecvCountsT[r];
  }

  //make ops for scattering halo nodes before sending
  NsendN=mpiSendOffsetsN[size];
  NsendT=mpiSendOffsetsT[size];

  sendIdsN.calloc(NsendN);
  sendIdsT.calloc(NsendT);

  NsendN=0; //positive node count
  NsendT=0; //all node count

  for (dlong n=0;n<Nshared;n++) { //loop through nodes we need to send
    dlong id = sharedNodes[n].newId; //coalesced index for this baseId on this rank
    if (sharedNodes[n].sign==2) {
      sendIdsN[NsendN++] = id;
    }
    sendIdsT[NsendT++] = id;
  }
  o_sendIdsT = platform.malloc(NsendT*sizeof(dlong), sendIdsT.ptr());
  o_sendIdsN = platform.malloc(NsendN*sizeof(dlong), sendIdsN.ptr());

  //send the node lists so we know what we'll receive
  dlong Nrecv = mpiRecvOffsetsT[size];
  libp::memory<parallelNode_t> recvNodes(Nrecv);

  //Send list of nodes to each rank
  MPI_Alltoallv(sharedNodes.ptr(), mpiSendCountsT.ptr(), mpiSendOffsetsT.ptr(), MPI_PARALLELNODE_T,
                  recvNodes.ptr(), mpiRecvCountsT.ptr(), mpiRecvOffsetsT.ptr(), MPI_PARALLELNODE_T,
                comm);
  MPI_Barrier(comm);

  //make ops for gathering halo nodes after an MPI_Allgatherv
  postmpi.platform = platform;
  postmpi.kind = Signed;

  postmpi.NrowsN = Nhalo;
  postmpi.NrowsT = Nhalo;
  postmpi.rowStartsN.calloc(Nhalo+1);
  postmpi.rowStartsT.calloc(Nhalo+1);

  //make array of counters
  libp::memory<dlong> haloGatherTCounts(Nhalo);
  libp::memory<dlong> haloGatherNCounts(Nhalo);

  //count the data that will already be in haloBuf
  for (dlong n=0;n<Nhalo;n++) {
    haloGatherNCounts[n] = (n<NhaloP) ? 1 : 0;
    haloGatherTCounts[n] = 1;
  }

  for (dlong n=0;n<Nrecv;n++) { //loop through nodes needed for gathering halo nodes
    dlong id = recvNodes[n].localId; //coalesced index for this baseId on this rank
    if (recvNodes[n].sign==2) haloGatherNCounts[id]++;  //tally
    haloGatherTCounts[id]++;  //tally
  }

  for (dlong i=0;i<Nhalo;i++) {
    postmpi.rowStartsN[i+1] = postmpi.rowStartsN[i] + haloGatherNCounts[i];
    postmpi.rowStartsT[i+1] = postmpi.rowStartsT[i] + haloGatherTCounts[i];
    haloGatherNCounts[i] = 0;
    haloGatherTCounts[i] = 0;
  }
  postmpi.nnzN = postmpi.rowStartsN[Nhalo];
  postmpi.nnzT = postmpi.rowStartsT[Nhalo];
  postmpi.colIdsN.calloc(postmpi.nnzN);
  postmpi.colIdsT.calloc(postmpi.nnzT);

  for (dlong n=0;n<NhaloP;n++) {
    const dlong soffset = postmpi.rowStartsN[n];
    const int sindex  = haloGatherNCounts[n];
    postmpi.colIdsN[soffset+sindex] = n; //record id
    haloGatherNCounts[n]++;
  }
  for (dlong n=0;n<Nhalo;n++) {
    const dlong soffset = postmpi.rowStartsT[n];
    const int sindex  = haloGatherTCounts[n];
    postmpi.colIdsT[soffset+sindex] = n; //record id
    haloGatherTCounts[n]++;
  }

  dlong cnt=Nhalo; //positive node count
  for (dlong n=0;n<Nrecv;n++) { //loop through nodes we need to send
    dlong id = recvNodes[n].localId; //coalesced index for this baseId on this rank
    if (recvNodes[n].sign==2) {
      const dlong soffset = postmpi.rowStartsN[id];
      const int sindex  = haloGatherNCounts[id];
      postmpi.colIdsN[soffset+sindex] = cnt++; //record id
      haloGatherNCounts[id]++;
    }
    const dlong soffset = postmpi.rowStartsT[id];
    const int sindex  = haloGatherTCounts[id];
    postmpi.colIdsT[soffset+sindex] = n + Nhalo; //record id
    haloGatherTCounts[id]++;
  }

  postmpi.o_rowStartsN = platform.malloc((postmpi.NrowsT+1)*sizeof(dlong), postmpi.rowStartsN.ptr());
  postmpi.o_rowStartsT = platform.malloc((postmpi.NrowsT+1)*sizeof(dlong), postmpi.rowStartsT.ptr());
  postmpi.o_colIdsN = platform.malloc((postmpi.nnzN)*sizeof(dlong), postmpi.colIdsN.ptr());
  postmpi.o_colIdsT = platform.malloc((postmpi.nnzT)*sizeof(dlong), postmpi.colIdsT.ptr());

  //free up space
  recvNodes.free();
  haloGatherNCounts.free();
  haloGatherTCounts.free();

  postmpi.setupRowBlocks();

  //compress the send/recv counts to pairwise exchanges
  NranksSendN=0;
  NranksSendT=0;
  NranksRecvN=0;
  NranksRecvT=0;
  for (int r=0;r<size;r++) {
    NranksSendN += (mpiSendCountsN[r]>0) ? 1 : 0;
    NranksSendT += (mpiSendCountsT[r]>0) ? 1 : 0;
    NranksRecvN += (mpiRecvCountsN[r]>0) ? 1 : 0;
    NranksRecvT += (mpiRecvCountsT[r]>0) ? 1 : 0;
  }

  sendRanksN.calloc(NranksSendN);
  sendRanksT.calloc(NranksSendT);
  recvRanksN.calloc(NranksRecvN);
  recvRanksT.calloc(NranksRecvT);
  sendCountsN.calloc(NranksSendN);
  sendCountsT.calloc(NranksSendT);
  recvCountsN.calloc(NranksRecvN);
  recvCountsT.calloc(NranksRecvT);
  sendOffsetsN.calloc(NranksSendN+1);
  sendOffsetsT.calloc(NranksSendT+1);
  recvOffsetsN.calloc(NranksRecvN+1);
  recvOffsetsT.calloc(NranksRecvT+1);

  //reset
  NranksSendN=0;
  NranksSendT=0;
  NranksRecvN=0;
  NranksRecvT=0;
  for (int r=0;r<size;r++) {
    if (mpiSendCountsN[r]>0) {
      sendRanksN[NranksSendN]  = r;
      sendCountsN[NranksSendN] = mpiSendCountsN[r];
      sendOffsetsN[NranksSendN] = mpiSendOffsetsN[r];
      NranksSendN++;
    }
    if (mpiSendCountsT[r]>0) {
      sendRanksT[NranksSendT]  = r;
      sendCountsT[NranksSendT] = mpiSendCountsT[r];
      sendOffsetsT[NranksSendT] = mpiSendOffsetsT[r];
      NranksSendT++;
    }
    if (mpiRecvCountsN[r]>0) {
      recvRanksN[NranksRecvN]   = r;
      recvCountsN[NranksRecvN]  = mpiRecvCountsN[r];
      recvOffsetsN[NranksRecvN] = mpiRecvOffsetsN[r];
      NranksRecvN++;
    }
    if (mpiRecvCountsT[r]>0) {
      recvRanksT[NranksRecvT]   = r;
      recvCountsT[NranksRecvT]  = mpiRecvCountsT[r];
      recvOffsetsT[NranksRecvT] = mpiRecvOffsetsT[r];
      NranksRecvT++;
    }
  }
  sendOffsetsN[NranksSendN] = mpiSendOffsetsN[size];
  sendOffsetsT[NranksSendT] = mpiSendOffsetsT[size];
  recvOffsetsN[NranksRecvN] = mpiRecvOffsetsN[size];
  recvOffsetsT[NranksRecvT] = mpiRecvOffsetsT[size];

  requests.malloc(NranksSendT+NranksRecvT);
  statuses.malloc(NranksSendT+NranksRecvT);

  //make scratch space
  AllocBuffer(Sizeof(Dfloat));
}

void ogsPairwise_t::AllocBuffer(size_t Nbytes) {
  if (o_haloBuf.size() < postmpi.nnzT*Nbytes) {
    if (o_haloBuf.size()) o_haloBuf.free();
    haloBuf = static_cast<char*>(platform.hostMalloc(postmpi.nnzT*Nbytes,  nullptr, h_haloBuf));
    o_haloBuf = platform.malloc(postmpi.nnzT*Nbytes);
  }
  if (o_sendBuf.size() < NsendT*Nbytes) {
    if (o_sendBuf.size()) o_sendBuf.free();
    sendBuf = static_cast<char*>(platform.hostMalloc(NsendT*Nbytes,  nullptr, h_sendBuf));
    o_sendBuf = platform.malloc(NsendT*Nbytes);
  }
}

} //namespace ogs

} //namespace libp
