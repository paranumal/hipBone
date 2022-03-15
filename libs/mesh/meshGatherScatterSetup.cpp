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

#include "mesh.hpp"

namespace libp {

void mesh_t::GatherScatterSetup() {

  dlong Ntotal = Nverts*(Nelements+totalHaloPairs);

  memory<int> minRank(Ntotal);
  memory<int> maxRank(Ntotal);

  for (dlong i=0;i<Ntotal;i++) {
    minRank[i] = rank;
    maxRank[i] = rank;
  }

  hlong gatherChange = 1;

  // keep comparing numbers on positive and negative traces until convergence
  while(gatherChange>0){

    // reset change counter
    gatherChange = 0;

    // send halo data and recv into extension of buffer
    halo.Exchange(minRank, Nverts);
    halo.Exchange(maxRank, Nverts);

    // compare trace vertices
    #pragma omp parallel for collapse(2)
    for(dlong e=0;e<Nelements;++e){
      for(int n=0;n<Nfaces*NfaceVertices;++n){
        dlong id  = e*Nfaces*NfaceVertices + n;
        dlong idM = VmapM[id];
        dlong idP = VmapP[id];

        int minRankM = minRank[idM];
        int minRankP = minRank[idP];

        int maxRankM = maxRank[idM];
        int maxRankP = maxRank[idP];

        if(minRankP<minRankM){
          gatherChange=1;
          minRank[idM] = minRankP;
        }

        if(maxRankP>maxRankM){
          gatherChange=1;
          maxRank[idM] = maxRankP;
        }
      }
    }

    // sum up changes
    comm.Allreduce(gatherChange);
  }

  // count elements that contribute to global C0 gather-scatter
  dlong globalCount = 0;
  dlong localCount = 0;
  for(dlong e=0;e<Nelements;++e){
    int isHalo = 0;
    for(int n=0;n<Nverts;++n){
      dlong id = e*Nverts+n;
      if ((minRank[id]!=rank)||(maxRank[id]!=rank)) {
        isHalo = 1;
        break;
      }
    }
    globalCount += isHalo;
    localCount += 1-isHalo;
  }

  globalGatherElementList.malloc(globalCount);
  localGatherElementList.malloc(localCount);

  globalCount = 0;
  localCount = 0;

  for(dlong e=0;e<Nelements;++e){
    int isHalo = 0;
    for(int n=0;n<Nverts;++n){
      dlong id = e*Nverts+n;
      if ((minRank[id]!=rank)||(maxRank[id]!=rank)) {
        isHalo = 1;
        break;
      }
    }
    if(isHalo){
      globalGatherElementList[globalCount++] = e;
    } else{
      localGatherElementList[localCount++] = e;
    }
  }

  NglobalGatherElements = globalCount;
  NlocalGatherElements = localCount;

  minRank.free(); maxRank.free();


  //make a masked version of the global id numbering
  maskedGlobalIds.malloc(Nelements*Np);
  for (dlong n=0;n<Nelements*Np;++n)
    maskedGlobalIds[n] = globalIds[n];

  //The mesh is just a structured brick, so we don't have to worry about
  // singleton corners or edges that are on the boundary. Just
  // check the faces through EToE
  for (dlong e=0;e<Nelements;e++) {
    for (int f=0;f<Nfaces;f++) {
      if (EToE[f+e*Nfaces]<0) { //unconnected faces are boundary
        for (int n=0;n<Nfp;n++) {
          const int fid = faceNodes[n+f*Nfp];
          maskedGlobalIds[fid+e*Np] = 0; //mask
        }
      }
    }
  }

  Nmasked=0;
  for (dlong n=0;n<Nelements*Np;++n)
    if (maskedGlobalIds[n]==0) Nmasked++;

  //use the masked ids to make another gs handle (signed so the gather is defined)
  bool verbose = platform.settings().compareSetting("VERBOSE", "TRUE");
  bool unique = true; //flag a unique node in every gather node
  ogsMasked.Setup(Nelements*Np, maskedGlobalIds,
                  comm, ogs::Signed, ogs::Auto,
                  unique, verbose, platform);

  gHalo.SetupFromGather(ogsMasked);

  GlobalToLocal.malloc(Nelements*Np);
  ogsMasked.SetupGlobalToLocalMapping(GlobalToLocal);

  o_GlobalToLocal = platform.malloc(GlobalToLocal);

  /* use the masked gs handle to define a global ordering */
  hlong Ngather = ogsMasked.Ngather;     // number of degrees of freedom on this rank (after gathering)

  // build inverse degree vectors
  // used for the weight in linear solvers (used in C0)
  Ntotal = Np*Nelements;
  weight.malloc(Ntotal);
  weightG.malloc(ogsMasked.Ngather);

  #pragma omp parallel for
  for(dlong n=0;n<Ntotal;++n) weight[n] = 1.0;

  ogsMasked.Gather(weightG, weight, 1, ogs::Add, ogs::Trans);

  #pragma omp parallel for
  for(dlong n=0;n<ogsMasked.Ngather;++n)
    if (weightG[n]) weightG[n] = 1./weightG[n];

  ogsMasked.Scatter(weight, weightG, 1, ogs::NoTrans);

  // o_weight  = device.malloc(Ntotal*sizeof(dfloat), weight);
  // o_weightG = device.malloc(ogsMasked.Ngather*sizeof(dfloat), weightG);

  // create a global numbering system
  memory<hlong> newglobalIds(Ngather);

  // every gathered degree of freedom has its own global id
  hlong localNgather = Ngather;
  hlong gatherOffset = localNgather;
  comm.Scan(localNgather, gatherOffset);
  gatherOffset -= localNgather;

  //use the offsets to set a consecutive global numbering
  #pragma omp parallel for
  for (dlong n =0;n<ogsMasked.Ngather;n++) {
    newglobalIds[n] = n + gatherOffset;
  }

  //scatter this numbering to the original nodes
  maskedGlobalNumbering.malloc(Ntotal);

  #pragma omp parallel for
  for (dlong n=0;n<Ntotal;n++) maskedGlobalNumbering[n] = -1;

  ogsMasked.Scatter(maskedGlobalNumbering, newglobalIds, 1, ogs::NoTrans);
}

} //namespace libp
