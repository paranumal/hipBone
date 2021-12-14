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

void ogs_t::Setup(const dlong _N,
                  hlong *ids,
                  MPI_Comm _comm,
                  const Kind _kind,
                  const Method method,
                  const bool _unique,
                  const bool verbose,
                  platform_t& _platform){
  ogsBase_t::Setup(_N, ids, _comm, _kind, method, _unique, verbose, _platform);
}

void halo_t::Setup(const dlong _N,
                  hlong *ids,
                  MPI_Comm _comm,
                  const Method method,
                  const bool verbose,
                  platform_t& _platform){
  ogsBase_t::Setup(_N, ids, _comm, Halo, method, false, verbose, _platform);

  Nhalo = NhaloT - NhaloP; //number of extra recieved nodes
}

/********************************
 * Setup
 ********************************/
void ogsBase_t::Setup(const dlong _N,
                      hlong *ids,
                      MPI_Comm _comm,
                      const Kind _kind,
                      const Method method,
                      const bool _unique,
                      const bool verbose,
                      platform_t& _platform){

  //release resources if this ogs was setup before
  Free();

  platform = _platform;

  N = _N;
  comm = _comm;
  kind = _kind;
  unique = _unique;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  //sanity check options
  if (   (kind==Unsigned && unique==true)
      || (kind==Halo && unique==true) )
    HIPBONE_ABORT("Invalid ogs setup requested");

  //count how many ids are non-zero
  dlong Nids=0;
  for (dlong n=0;n<N;n++)
    if (ids[n]!=0) Nids++;

  // make list of nodes
  parallelNode_t *nodes = new parallelNode_t[Nids];

  //fill the data (squeezing out zero ids)
  Nids=0;
  for (dlong n=0;n<N;n++) {
    if (ids[n]!=0) {
      nodes[Nids].localId = Nids; //record a compressed id first (useful for ordering)
      nodes[Nids].baseId = (kind==Unsigned) ?
                            abs(ids[n]) : ids[n]; //record global id
      nodes[Nids].rank = rank;
      nodes[Nids].destRank = abs(ids[n]) % size;
      Nids++;
    }
  }

  /*Register MPI_PARALLELNODE_T type*/
  InitMPIType();

  //flag which nodes are shared via MPI
  FindSharedNodes(Nids, nodes, verbose);

  //Index the local and halo baseIds on this rank and
  // construct sharedNodes which contains all the info
  // we need to setup the MPI exchange.
  dlong Nshared=0;
  parallelNode_t *sharedNodes=nullptr;
  ConstructSharedNodes(Nids, nodes, Nshared, sharedNodes);

  Nids=0;
  for (dlong n=0;n<N;n++) {
    if (ids[n]!=0) {
      nodes[Nids].localId = n; //record the real id now

      //if we altered the signs of ids, write them back
      if (unique)
        ids[n] = nodes[Nids].baseId;

      Nids++;
    }
  }

  //setup local gather operators
  if (kind==Signed)
    LocalSignedSetup(Nids, nodes);
  else if (kind==Unsigned)
    LocalUnsignedSetup(Nids, nodes);
  else
    LocalHaloSetup(Nids, nodes);

  //with that, we're done with the local nodes list
  delete[] nodes;

  // At this point, we've setup gs operators to gather/scatter the purely local nodes,
  // and gather/scatter the shared halo nodes to/from a coalesced ordering. We now
  // need gs operators to scatter/gather the coalesced halo nodes to/from the expected
  // orderings for MPI communications.

  if (method == AllToAll) {
    exchange = std::shared_ptr<ogsExchange_t>(
                  new ogsAllToAll_t(Nshared, sharedNodes,
                                    *gatherHalo, comm, platform));
  } else if (method == Pairwise) {
    exchange = std::shared_ptr<ogsExchange_t>(
                  new ogsPairwise_t(Nshared, sharedNodes,
                                    *gatherHalo, comm, platform));
  } else if (method == CrystalRouter) {
    exchange = std::shared_ptr<ogsExchange_t>(
                  new ogsCrystalRouter_t(Nshared, sharedNodes,
                                         *gatherHalo, comm, platform));
  } else { //Auto
    exchange = std::shared_ptr<ogsExchange_t>(
                  AutoSetup(Nshared, sharedNodes,
                            *gatherHalo, comm,
                            platform, verbose));
  }

  /*Free the MPI_PARALLELNODE_T type*/
  DestroyMPIType();

  //we're now done with the sharedNodes list
  delete[] sharedNodes;
}

void ogsBase_t::FindSharedNodes(const dlong Nids,
                                parallelNode_t nodes[],
                                const int verbose){

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int *sendCounts =  new int[size];
  int *recvCounts =  new int[size];
  int *sendOffsets = new int[size+1];
  int *recvOffsets = new int[size+1];

  for (int r=0;r<size;r++) {
    sendCounts[r] = 0;
  }

  //count number of ids we're sending
  for (dlong n=0;n<Nids;n++) {
    sendCounts[nodes[n].destRank]++;
  }

  MPI_Alltoall(sendCounts, 1, MPI_INT,
               recvCounts, 1, MPI_INT, comm);

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for (int r=0;r<size;r++) {
    sendOffsets[r+1] = sendOffsets[r]+sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r]+recvCounts[r];

    //reset counter
    sendCounts[r] = 0;
  }

  //write a send ordering into newIds
  for (dlong n=0;n<Nids;n++) {
    const int r = nodes[n].destRank;
    nodes[n].newId = sendOffsets[r]+sendCounts[r]++;
  }

  // permute the list to send ordering
  permute(Nids, nodes, [](const parallelNode_t& a) { return a.newId; } );

  dlong recvN = recvOffsets[size]; //total ids to recv

  parallelNode_t *recvNodes = new parallelNode_t[recvN];

  //Send all the nodes to their destination rank.
  MPI_Alltoallv(    nodes, sendCounts, sendOffsets, MPI_PARALLELNODE_T,
                recvNodes, recvCounts, recvOffsets, MPI_PARALLELNODE_T,
                comm);

  //remember this ordering
  for (dlong n=0;n<recvN;n++) {
    recvNodes[n].newId = n;
  }

  // sort based on base ids
  std::sort(recvNodes, recvNodes+recvN,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              return abs(a.baseId) < abs(b.baseId);
            });

  // We now have a collection of nodes associated with some subset of all global Ids
  // Our list is sorted by baseId to group nodes with the same globalId together
  // We now want to flag which nodes are shared via MPI

  int locally_unique=1;

  dlong Nshared=0;

  dlong start=0;
  for (dlong n=0;n<recvN;n++) {
    if (n==recvN-1 || abs(recvNodes[n].baseId)!=abs(recvNodes[n+1].baseId)) {
      dlong end = n+1;

      int positiveCount=0;
      if (unique) {
        //Make a single node from each baseId group the sole positive node
        const hlong baseId = abs(recvNodes[start].baseId);

        //pick a random node in this group
        const int m = (rand() % (end-start));

        for (int i=start;i<end;i++)
          recvNodes[i].baseId = -baseId;

        recvNodes[start+m].baseId = baseId;
        positiveCount=1;
      } else {
        //count how many postive baseIds there are in this group
        for (int i=start;i<end;i++)
          if (recvNodes[i].baseId>0) positiveCount++;

        //if we didnt find a sole positive baseId, the gather is not well-defined
        if (positiveCount!=1) locally_unique=0;
      }

      // When making a halo excahnge, check that we have a leading positive id
      if (kind==Halo && positiveCount!=1) {
        std::stringstream ss;
        ss << "Found " << positiveCount << " positive Ids for baseId: "
           << abs(recvNodes[start].baseId)<< ".";
        HIPBONE_ABORT(ss.str());
      }

      //determine if this node is shared via MPI,
      int shared=1;
      const int r = recvNodes[start].rank;
      for (int i=start+1;i<end;i++) {
        if (recvNodes[i].rank != r) {
          shared=2;
          Nshared++;
          break;
        }
      }

      //set shared flag.
      for (int i=start;i<end;i++) {
        recvNodes[i].sign = shared;
      }

      //set new baseId group start point
      start=n+1;
    }
  }

  //shared the unique node check so we know if the gather operation is well-defined
  int globally_unique=1;
  MPI_Allreduce(&locally_unique, &globally_unique, 1, MPI_INT, MPI_MIN, comm);
  gather_defined = (globally_unique==1);

  hlong Nshared_local = Nshared;
  hlong Nshared_global = Nshared;
  MPI_Reduce(&Nshared_local, &Nshared_global, 1, MPI_HLONG, MPI_SUM, 0, comm);
  if (!rank && verbose) {
    std::cout << "ogs Setup: " << Nshared_global << " unique labels shared." << std::endl;
  }

  //at this point each collection of baseIds either has all nodes have
  // sign = 1, meaning all the nodes with this baseId are on the
  // same rank, or have sign=2, meaning that baseId must be communicated

  // permute recv nodes back to recv'd ordering
  permute(recvN, recvNodes, [](const parallelNode_t& a) { return a.newId; } );

  //Return all the nodes to their origin rank.
  MPI_Alltoallv(recvNodes, recvCounts, recvOffsets, MPI_PARALLELNODE_T,
                    nodes, sendCounts, sendOffsets, MPI_PARALLELNODE_T,
                comm);
  //free up some space
  MPI_Barrier(comm);
  delete[] recvNodes;
  delete[] sendCounts;
  delete[] recvCounts;
  delete[] sendOffsets;
  delete[] recvOffsets;
}

void ogsBase_t::ConstructSharedNodes(const dlong Nids,
                                     parallelNode_t nodes[],
                                     dlong &Nshared,
                                     parallelNode_t* &sharedNodes) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // sort based on abs(baseId)
  std::sort(nodes, nodes+Nids,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              if(abs(a.baseId) < abs(b.baseId)) return true; //group by abs(baseId)
              if(abs(a.baseId) > abs(b.baseId)) return false;

              return a.baseId > b.baseId; //positive ids on a rank first
            });

  //count how many unique global Ids we have on this rank
  // and flag baseId groups that have a positive baseId somewhere on this rank
  dlong NbaseIds=0;
  NlocalT=0; NlocalP=0;
  NhaloT=0; NhaloP=0;
  dlong start=0;
  for (dlong n=0;n<Nids;n++) {
    if (n==Nids-1 || abs(nodes[n].baseId)!=abs(nodes[n+1].baseId)) {
      dlong end = n+1;

      //if there's no leading postive id, flag this baseId group as negative
      int sign = abs(nodes[start].sign);
      if (nodes[start].baseId<0) {
        sign = -sign;
        for (int i=start;i<end;i++) {
          nodes[i].sign = sign;
        }
      }

      //count the positive/negative local and halo gather nodes
      if (abs(sign)==1) {
        NlocalT++;
        if (sign==1) NlocalP++;
      } else {
        NhaloT++;
        if (sign==2) NhaloP++;
      }

      //record the new ordering
      for (int i=start;i<end;i++) {
        nodes[i].newId=NbaseIds;
      }

      NbaseIds++;
      start = end;
    }
  }

  //total number of positive owned gathered nodes
  Ngather = NlocalP+NhaloP;

  //global total
  hlong NgatherLocal = (hlong) Ngather;
  MPI_Allreduce(&NgatherLocal, &(NgatherGlobal), 1, MPI_HLONG, MPI_SUM, comm);

  //extract the leading node from each shared baseId
  parallelNode_t *sendSharedNodes = new parallelNode_t[NhaloT];

  NhaloT=0;
  for (dlong n=0;n<Nids;n++) {
    if (n==0 || abs(nodes[n].baseId)!=abs(nodes[n-1].baseId)) {
      if (abs(nodes[n].sign)==2) {
        sendSharedNodes[NhaloT++] = nodes[n];
      }
    }
  }

  // permute the list back to local id ordering
  permute(Nids, nodes, [](const parallelNode_t& a) { return a.localId; } );

  // Use the newId index to reorder the baseId groups based on
  // the order we encouter them in their original ordering.
  dlong* indexMap = new dlong[NbaseIds];
  for (dlong i=0;i<NbaseIds;i++) indexMap[i] = -1; //initialize map

  dlong localCntN = 0, localCntT = NlocalP;  //start point for local gather nodes
  dlong haloCntN  = 0, haloCntT  = NhaloP;   //start point for halo gather nodes
  for (dlong n=0;n<Nids;n++) {
    const dlong newId = nodes[n].newId; //get the new baseId group id

    //record a new index if we've not encoutered this baseId group before
    if (indexMap[newId]==-1) {
      if        (nodes[n].sign== 1) {
        indexMap[newId] = localCntN++;
      } else if (nodes[n].sign==-1) {
        indexMap[newId] = localCntT++;
      } else if (nodes[n].sign== 2) {
        indexMap[newId] = haloCntN++;
      } else { //nodes[n].sign==-2
        indexMap[newId] = haloCntT++;
      }
    }

    const dlong gid = indexMap[newId];
    nodes[n].newId = gid; //reorder
  }

  //re-order the shared node list
  for (dlong n=0;n<NhaloT;n++) {
    const dlong newId = sendSharedNodes[n].newId; //get the new baseId group id
    const dlong gid = indexMap[newId];
    sendSharedNodes[n].localId = gid; //reorder the localId to the compressed order
  }

  delete[] indexMap;

  int *sendCounts = new int[size];
  int *recvCounts = new int[size];
  int *sendOffsets = new int[size+1];
  int *recvOffsets = new int[size+1];

  // sort based on destination rank
  std::sort(sendSharedNodes, sendSharedNodes+NhaloT,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              return a.destRank < b.destRank;
            });

  for (int r=0;r<size;r++) {
    sendCounts[r] = 0;
  }

  //count number of ids we're sending
  for (dlong n=0;n<NhaloT;n++) {
    sendCounts[sendSharedNodes[n].destRank]++;
  }

  MPI_Alltoall(sendCounts, 1, MPI_INT,
               recvCounts, 1, MPI_INT, comm);

  sendOffsets[0] = 0;
  recvOffsets[0] = 0;
  for (int r=0;r<size;r++) {
    sendOffsets[r+1] = sendOffsets[r]+sendCounts[r];
    recvOffsets[r+1] = recvOffsets[r]+recvCounts[r];
  }
  dlong recvN = recvOffsets[size]; //total ids to recv

  parallelNode_t *recvSharedNodes = new parallelNode_t[recvN];

  //Send all the nodes to their destination rank.
  MPI_Alltoallv(sendSharedNodes, sendCounts, sendOffsets, MPI_PARALLELNODE_T,
                recvSharedNodes, recvCounts, recvOffsets, MPI_PARALLELNODE_T,
                comm);

  //free up some space
  MPI_Barrier(comm);
  delete[] sendSharedNodes;
  delete[] sendCounts;
  delete[] recvCounts;
  delete[] sendOffsets;
  delete[] recvOffsets;

  // sort based on base ids
  std::sort(recvSharedNodes, recvSharedNodes+recvN,
            [](const parallelNode_t& a, const parallelNode_t& b) {
              return abs(a.baseId) < abs(b.baseId);
            });

  //count number of shared nodes we will be sending
  int *sharedSendCounts  = new int[size];
  int *sharedRecvCounts  = new int[size];
  int *sharedSendOffsets = new int[size+1];
  int *sharedRecvOffsets = new int[size+1];

  for (int r=0;r<size;r++) {
    sharedSendCounts[r] = 0;
  }

  start=0;
  for (dlong n=0;n<recvN;n++) {
    if (n==recvN-1 || abs(recvSharedNodes[n].baseId)!=abs(recvSharedNodes[n+1].baseId)) {
      dlong end = n+1;

      for (int i=start;i<end;i++) {
        //We'll be sending all the shared nodes to each rank involved
        sharedSendCounts[recvSharedNodes[i].rank] += end-start-1;
      }

      //set new baseId group start point
      start=n+1;
    }
  }

  // Each rank has a set of shared global Ids and for each global id that
  // rank knows what MPI ranks participate in gathering. We now send this
  // information to the involved ranks.

  //share counts
  MPI_Alltoall(sharedSendCounts, 1, MPI_INT,
               sharedRecvCounts, 1, MPI_INT, comm);

  //cumulative sum
  sharedSendOffsets[0] = 0;
  sharedRecvOffsets[0] = 0;
  for (int r=0;r<size;r++) {
    sharedSendOffsets[r+1] = sharedSendOffsets[r]+sharedSendCounts[r];
    sharedRecvOffsets[r+1] = sharedRecvOffsets[r]+sharedRecvCounts[r];
  }

  //make a send buffer
  parallelNode_t *sharedSendNodes = new parallelNode_t[sharedSendOffsets[size]];

  //reset sendCounts
  for (int r=0;r<size;r++) sharedSendCounts[r]=0;

  start=0;
  for (dlong n=0;n<recvN;n++) {
    if (n==recvN-1 || abs(recvSharedNodes[n].baseId)!=abs(recvSharedNodes[n+1].baseId)) {
      dlong end = n+1;

      //build the node list to send
      for (int i=start;i<end;i++) {
        const int r = recvSharedNodes[i].rank;
        const dlong id = recvSharedNodes[i].localId;
        const int sign = recvSharedNodes[i].sign;

        int sid = sharedSendCounts[r]+sharedSendOffsets[r];
        for (int j=start;j<end;j++) {
          if (j==i) continue; //dont bother sending this rank's own node
          sharedSendNodes[sid] = recvSharedNodes[j];
          sharedSendNodes[sid].newId = id;
          sharedSendNodes[sid].sign = sign;
          sid++;
        }
        sharedSendCounts[r] += end-start-1;
      }

      //set new baseId group start point
      start=n+1;
    }
  }
  delete[] recvSharedNodes;

  //make sharedNodes to hold the exchange data we recv
  Nshared = sharedRecvOffsets[size];
  sharedNodes = new parallelNode_t[Nshared];

  //Share all the gathering info
  MPI_Alltoallv(sharedSendNodes, sharedSendCounts, sharedSendOffsets, MPI_PARALLELNODE_T,
                    sharedNodes, sharedRecvCounts, sharedRecvOffsets, MPI_PARALLELNODE_T,
                comm);

  //free up space
  MPI_Barrier(comm);
  delete[] sharedSendNodes;
  delete[] sharedSendCounts;
  delete[] sharedRecvCounts;
  delete[] sharedSendOffsets;
  delete[] sharedRecvOffsets;
}

//Make local and halo gather operators using nodes list
void ogsBase_t::LocalSignedSetup(const dlong Nids, parallelNode_t* nodes){

int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);

  gatherLocal = std::make_shared<ogsOperator_t>(platform);
  gatherHalo  = std::make_shared<ogsOperator_t>(platform);

  gatherLocal->kind = Signed;
  gatherHalo->kind = Signed;

  gatherLocal->Ncols = N;
  gatherHalo->Ncols = N;

  gatherLocal->NrowsN = NlocalP;
  gatherLocal->NrowsT = NlocalT;
  gatherHalo->NrowsN = NhaloP;
  gatherHalo->NrowsT = NhaloT;

  //tally up how many nodes are being gathered to each gatherNode and
  //  map to a local ordering
  dlong *localGatherNCounts = new dlong[gatherLocal->NrowsT];
  dlong *localGatherTCounts = new dlong[gatherLocal->NrowsT];
  dlong *haloGatherNCounts  = new dlong[gatherHalo->NrowsT];
  dlong *haloGatherTCounts  = new dlong[gatherHalo->NrowsT];

  for (dlong i=0;i<gatherLocal->NrowsT;++i) {
    localGatherNCounts[i]=0;
    localGatherTCounts[i]=0;
  }

  for (dlong i=0;i<gatherHalo->NrowsT;++i) {
    haloGatherNCounts[i]=0;
    haloGatherTCounts[i]=0;
  }

  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId; //re-mapped baseId on this rank

    if (abs(nodes[i].sign)==1) { //local
      if (nodes[i].baseId>0) localGatherNCounts[gid]++;  //tally
      localGatherTCounts[gid]++;  //tally
    } else { //halo
      if (nodes[i].baseId>0) haloGatherNCounts[gid]++;  //tally
      haloGatherTCounts[gid]++;  //tally
    }
  }

  //make local row offsets
  gatherLocal->rowStartsN.malloc(gatherLocal->NrowsT+1);
  gatherLocal->rowStartsT.malloc(gatherLocal->NrowsT+1);
  gatherLocal->rowStartsN[0] = 0;
  gatherLocal->rowStartsT[0] = 0;
  for (dlong i=0;i<gatherLocal->NrowsT;i++) {
    gatherLocal->rowStartsN[i+1] = gatherLocal->rowStartsN[i] + localGatherNCounts[i];
    gatherLocal->rowStartsT[i+1] = gatherLocal->rowStartsT[i] + localGatherTCounts[i];
    localGatherNCounts[i] = 0; //reset counters
    localGatherTCounts[i] = 0; //reset counters
  }
  gatherLocal->nnzN = gatherLocal->rowStartsN[gatherLocal->NrowsT];
  gatherLocal->nnzT = gatherLocal->rowStartsT[gatherLocal->NrowsT];
  gatherLocal->colIdsN.malloc(gatherLocal->nnzN);
  gatherLocal->colIdsT.malloc(gatherLocal->nnzT);

  //make halo row offsets
  gatherHalo->rowStartsN.malloc(gatherHalo->NrowsT+1);
  gatherHalo->rowStartsT.malloc(gatherHalo->NrowsT+1);
  gatherHalo->rowStartsN[0] = 0;
  gatherHalo->rowStartsT[0] = 0;
  for (dlong i=0;i<gatherHalo->NrowsT;i++) {
    gatherHalo->rowStartsN[i+1] = gatherHalo->rowStartsN[i] + haloGatherNCounts[i];
    gatherHalo->rowStartsT[i+1] = gatherHalo->rowStartsT[i] + haloGatherTCounts[i];
    haloGatherNCounts[i] = 0;
    haloGatherTCounts[i] = 0;
  }
  gatherHalo->nnzN = gatherHalo->rowStartsN[gatherHalo->NrowsT];
  gatherHalo->nnzT = gatherHalo->rowStartsT[gatherHalo->NrowsT];
  gatherHalo->colIdsN.malloc(gatherHalo->nnzN);
  gatherHalo->colIdsT.malloc(gatherHalo->nnzT);


  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId;

    if (abs(nodes[i].sign)==1) { //local gather group
      if (nodes[i].baseId>0) {
        const dlong soffset = gatherLocal->rowStartsN[gid];
        const int sindex  = localGatherNCounts[gid];
        gatherLocal->colIdsN[soffset+sindex] = nodes[i].localId;
        localGatherNCounts[gid]++;
      }
      const dlong soffset = gatherLocal->rowStartsT[gid];
      const int sindex  = localGatherTCounts[gid];
      gatherLocal->colIdsT[soffset+sindex] = nodes[i].localId;
      localGatherTCounts[gid]++;
    } else {
      if (nodes[i].baseId>0) {
        const dlong soffset = gatherHalo->rowStartsN[gid];
        const int sindex  = haloGatherNCounts[gid];
        gatherHalo->colIdsN[soffset+sindex] = nodes[i].localId;
        haloGatherNCounts[gid]++;
      }
      const dlong soffset = gatherHalo->rowStartsT[gid];
      const int sindex  = haloGatherTCounts[gid];
      gatherHalo->colIdsT[soffset+sindex] = nodes[i].localId;
      haloGatherTCounts[gid]++;
    }
  }
  delete[] localGatherNCounts;
  delete[] localGatherTCounts;
  delete[] haloGatherNCounts;
  delete[] haloGatherTCounts;

  gatherLocal->o_rowStartsN = platform.malloc((gatherLocal->NrowsT+1)*sizeof(dlong), gatherLocal->rowStartsN.ptr());
  gatherLocal->o_rowStartsT = platform.malloc((gatherLocal->NrowsT+1)*sizeof(dlong), gatherLocal->rowStartsT.ptr());
  gatherLocal->o_colIdsN = platform.malloc((gatherLocal->nnzN)*sizeof(dlong), gatherLocal->colIdsN.ptr());
  gatherLocal->o_colIdsT = platform.malloc((gatherLocal->nnzT)*sizeof(dlong), gatherLocal->colIdsT.ptr());

  gatherHalo->o_rowStartsN = platform.malloc((gatherHalo->NrowsT+1)*sizeof(dlong), gatherHalo->rowStartsN.ptr());
  gatherHalo->o_rowStartsT = platform.malloc((gatherHalo->NrowsT+1)*sizeof(dlong), gatherHalo->rowStartsT.ptr());
  gatherHalo->o_colIdsN = platform.malloc((gatherHalo->nnzN)*sizeof(dlong), gatherHalo->colIdsN.ptr());
  gatherHalo->o_colIdsT = platform.malloc((gatherHalo->nnzT)*sizeof(dlong), gatherHalo->colIdsT.ptr());

  //divide the list of colIds into roughly equal sized blocks so that each
  // threadblock loads approximately an equal amount of data
  gatherLocal->setupRowBlocks();
  gatherHalo->setupRowBlocks();
}

//Make local and halo gather operators using nodes list
void ogsBase_t::LocalUnsignedSetup(const dlong Nids, parallelNode_t* nodes){

  gatherLocal = std::make_shared<ogsOperator_t>(platform);
  gatherHalo  = std::make_shared<ogsOperator_t>(platform);

  gatherLocal->kind = Unsigned;
  gatherHalo->kind = Unsigned;

  gatherLocal->Ncols = N;
  gatherHalo->Ncols = N;

  gatherLocal->NrowsN = NlocalP;
  gatherLocal->NrowsT = NlocalT;
  gatherHalo->NrowsN = NhaloP;
  gatherHalo->NrowsT = NhaloT;

  //tally up how many nodes are being gathered to each gatherNode and
  //  map to a local ordering
  dlong *localGatherTCounts = new dlong[gatherLocal->NrowsT];
  dlong *haloGatherTCounts  = new dlong[gatherHalo->NrowsT];

  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId; //re-mapped baseId on this rank

    if (abs(nodes[i].sign)==1) { //local
      localGatherTCounts[gid]++;  //tally
    } else { //halo
      haloGatherTCounts[gid]++;  //tally
    }
  }

  //make local row offsets
  gatherLocal->rowStartsT.malloc(gatherLocal->NrowsT+1);
  gatherLocal->rowStartsN = gatherLocal->rowStartsT;
  gatherLocal->rowStartsT[0] = 0;
  for (dlong i=0;i<gatherLocal->NrowsT;i++) {
    gatherLocal->rowStartsT[i+1] = gatherLocal->rowStartsT[i] + localGatherTCounts[i];
    localGatherTCounts[i] = 0; //reset counters
  }
  gatherLocal->nnzT = gatherLocal->rowStartsT[gatherLocal->NrowsT];
  gatherLocal->nnzN = gatherLocal->nnzT;
  gatherLocal->colIdsT.malloc(gatherLocal->nnzT);
  gatherLocal->colIdsN = gatherLocal->colIdsT;

  //make halo row offsets
  gatherHalo->rowStartsT.malloc(gatherHalo->NrowsT+1);
  gatherHalo->rowStartsN = gatherHalo->rowStartsT;
  gatherHalo->rowStartsT[0] = 0;
  for (dlong i=0;i<gatherHalo->NrowsT;i++) {
    gatherHalo->rowStartsT[i+1] = gatherHalo->rowStartsT[i] + haloGatherTCounts[i];
    haloGatherTCounts[i] = 0;
  }
  gatherHalo->nnzT = gatherHalo->rowStartsT[gatherHalo->NrowsT];
  gatherHalo->nnzN = gatherHalo->nnzT;
  gatherHalo->colIdsT.malloc(gatherHalo->nnzT);
  gatherHalo->colIdsN = gatherHalo->colIdsT;


  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId;

    if (abs(nodes[i].sign)==1) { //local gather group
      const dlong soffset = gatherLocal->rowStartsT[gid];
      const int sindex  = localGatherTCounts[gid];
      gatherLocal->colIdsT[soffset+sindex] = nodes[i].localId;
      localGatherTCounts[gid]++;
    } else {
      const dlong soffset = gatherHalo->rowStartsT[gid];
      const int sindex  = haloGatherTCounts[gid];
      gatherHalo->colIdsT[soffset+sindex] = nodes[i].localId;
      haloGatherTCounts[gid]++;
    }
  }
  delete[] localGatherTCounts;
  delete[] haloGatherTCounts;

  gatherLocal->o_rowStartsT = platform.malloc((gatherLocal->NrowsT+1)*sizeof(dlong), gatherLocal->rowStartsT.ptr());
  gatherLocal->o_rowStartsN = gatherLocal->o_rowStartsT;
  gatherLocal->o_colIdsT = platform.malloc((gatherLocal->nnzT)*sizeof(dlong), gatherLocal->colIdsT.ptr());
  gatherLocal->o_colIdsN = gatherLocal->o_colIdsT;

  gatherHalo->o_rowStartsT = platform.malloc((gatherHalo->NrowsT+1)*sizeof(dlong), gatherHalo->rowStartsT.ptr());
  gatherHalo->o_rowStartsN = gatherHalo->o_rowStartsT;
  gatherHalo->o_colIdsT = platform.malloc((gatherHalo->nnzT)*sizeof(dlong), gatherHalo->colIdsT.ptr());
  gatherHalo->o_colIdsN = gatherHalo->o_colIdsT;

  //divide the list of colIds into roughly equal sized blocks so that each
  // threadblock loads approximately an equal amount of data
  gatherLocal->setupRowBlocks();
  gatherHalo->setupRowBlocks();
}

//Make local and halo gather operators using nodes list
void ogsBase_t::LocalHaloSetup(const dlong Nids, parallelNode_t* nodes){

  gatherHalo  = std::make_shared<ogsOperator_t>(platform);
  gatherHalo->kind = Signed;

  gatherHalo->Ncols = N;

  gatherHalo->NrowsN = NhaloP;
  gatherHalo->NrowsT = NhaloT;

  //tally up how many nodes are being gathered to each gatherNode and
  //  map to a local ordering
  dlong *haloGatherNCounts = new dlong[gatherHalo->NrowsT];
  dlong *haloGatherTCounts = new dlong[gatherHalo->NrowsT];

  for (dlong i=0;i<gatherHalo->NrowsT;i++) {
    haloGatherNCounts[i] = 0;
    haloGatherTCounts[i] = 0;
  }

  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId; //re-mapped baseId on this rank

    if (abs(nodes[i].sign)==2) {//halo
      if (nodes[i].sign==2) haloGatherNCounts[gid]++;  //tally
      haloGatherTCounts[gid]++;  //tally
    }
  }

  //make halo row offsets
  gatherHalo->rowStartsN.malloc(gatherHalo->NrowsT+1);
  gatherHalo->rowStartsT.malloc(gatherHalo->NrowsT+1);
  gatherHalo->rowStartsN[0]=0;
  gatherHalo->rowStartsT[0]=0;
  for (dlong i=0;i<gatherHalo->NrowsT;i++) {
    gatherHalo->rowStartsN[i+1] = gatherHalo->rowStartsN[i] + haloGatherNCounts[i];
    gatherHalo->rowStartsT[i+1] = gatherHalo->rowStartsT[i] + haloGatherTCounts[i];
    haloGatherNCounts[i] = 0;
    haloGatherTCounts[i] = 0;
  }
  gatherHalo->nnzN = gatherHalo->rowStartsN[gatherHalo->NrowsT];
  gatherHalo->nnzT = gatherHalo->rowStartsT[gatherHalo->NrowsT];
  gatherHalo->colIdsN.malloc(gatherHalo->nnzN);
  gatherHalo->colIdsT.malloc(gatherHalo->nnzT);


  for (dlong i=0;i<Nids;i++) {
    const dlong gid = nodes[i].newId;

    if (abs(nodes[i].sign)==2) {
      if (nodes[i].sign==2) {
        const dlong soffset = gatherHalo->rowStartsN[gid];
        const int sindex  = haloGatherNCounts[gid];
        gatherHalo->colIdsN[soffset+sindex] = nodes[i].localId;
        haloGatherNCounts[gid]++;
      }
      const dlong soffset = gatherHalo->rowStartsT[gid];
      const int sindex  = haloGatherTCounts[gid];
      gatherHalo->colIdsT[soffset+sindex] = nodes[i].localId;
      haloGatherTCounts[gid]++;
    }
  }
  delete[] haloGatherNCounts;
  delete[] haloGatherTCounts;

  gatherHalo->o_rowStartsN = platform.malloc((gatherHalo->NrowsT+1)*sizeof(dlong), gatherHalo->rowStartsN.ptr());
  gatherHalo->o_rowStartsT = platform.malloc((gatherHalo->NrowsT+1)*sizeof(dlong), gatherHalo->rowStartsT.ptr());
  gatherHalo->o_colIdsN = platform.malloc((gatherHalo->nnzN)*sizeof(dlong), gatherHalo->colIdsN.ptr());
  gatherHalo->o_colIdsT = platform.malloc((gatherHalo->nnzT)*sizeof(dlong), gatherHalo->colIdsT.ptr());

  //divide the list of colIds into roughly equal sized blocks so that each
  // threadblock loads approximately an equal amount of data
  gatherHalo->setupRowBlocks();
}

void ogsBase_t::Free() {
  gatherLocal = nullptr;
  gatherHalo = nullptr;
  exchange = nullptr;
  N=0;
  NlocalT=0;
  NhaloT=0;
  Ngather=0;
  NgatherGlobal=0;
}

void ogsBase_t::AssertGatherDefined() {
  if (!gather_defined) {
    HIPBONE_ABORT("Gather operation not well-defined.");
  }
}

//Populate the local mapping of the original ids and the gathered ordering
void ogs_t::SetupGlobalToLocalMapping(dlong *GlobalToLocal) {

  //Note: Must have GlobalToLocal have N entries.

  dlong *ids = (dlong*) malloc((NlocalT+NhaloT)*sizeof(dlong));

  for (dlong n=0;n<NlocalT+NhaloT;n++)
    ids[n] = n;

  for (dlong n=0;n<N;n++)
    GlobalToLocal[n] = -1;

  gatherLocal->Scatter(GlobalToLocal, ids,
                       1, Dlong, Add, NoTrans);
  gatherHalo->Scatter(GlobalToLocal, ids+NlocalT,
                       1, Dlong, Add, NoTrans);

  free(ids);
}

void halo_t::SetupFromGather(ogs_t& ogs) {

  ogs.AssertGatherDefined();

  N = ogs.NlocalT + ogs.NhaloT;

  Ngather = Ngather;
  Nhalo = ogs.NhaloT - ogs.NhaloP;

  NgatherGlobal = ogs.NgatherGlobal;
  comm = ogs.comm;

  kind = Halo;
  unique = ogs.unique;

  NlocalP = ogs.NlocalP;
  NlocalT  = ogs.NlocalT;

  NhaloP = ogs.NhaloP;
  NhaloT  = ogs.NhaloT;

  gather_defined=false;

  gathered_halo=true;

  exchange = ogs.exchange;
}

} //namespace ogs

} //namespace libp
