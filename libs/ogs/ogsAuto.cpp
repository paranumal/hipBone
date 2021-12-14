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

static void ExchangeTest(ogsExchange_t* exchange, double time[3], bool host=false) {
  const int Ncold = 10;
  const int Nhot  = 10;
  double start, end;
  double localTime, sumTime, minTime, maxTime;

  int rank, size;
  MPI_Comm_rank(exchange->comm, &rank);
  MPI_Comm_size(exchange->comm, &size);

  //dry run
  for (int n=0;n<Ncold;++n) {
    exchange->Start(1, Dfloat, Add, Sym, host);
    exchange->Finish(1, Dfloat, Add, Sym, host);
  }

  //hot runs
  if (!host) exchange->platform.device.finish();
  start = MPI_Wtime();
  for (int n=0;n<Nhot;++n) {
    exchange->Start(1, Dfloat, Add, Sym, host);
    exchange->Finish(1, Dfloat, Add, Sym, host);
  }
  if (!host) exchange->platform.device.finish();
  end = MPI_Wtime();

  localTime = (end-start)/Nhot;
  MPI_Allreduce(&localTime, &sumTime, 1, MPI_DOUBLE, MPI_SUM, exchange->comm);
  MPI_Allreduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, exchange->comm);
  MPI_Allreduce(&localTime, &minTime, 1, MPI_DOUBLE, MPI_MIN, exchange->comm);

  time[0] = sumTime/size; //avg
  time[1] = minTime;      //min
  time[2] = maxTime;      //max
}

ogsExchange_t* ogsBase_t::AutoSetup(dlong Nshared,
                                    libp::memory<parallelNode_t> &sharedNodes,
                                    ogsOperator_t& _gatherHalo,
                                    MPI_Comm _comm,
                                    platform_t &_platform,
                                    const int verbose) {

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  ogsExchange_t* bestExchange;
  Method method;
  double bestTime;

#ifdef GPU_AWARE_MPI
  if (rank==0 && verbose)
    printf("   Method         Device Exchange (avg, min, max)  Device Exchange (GPU-aware)      Host Exchange \n");
#else
  if (rank==0 && verbose)
    printf("   Method         Device Exchange (avg, min, max)  Host Exchange \n");
#endif

  //Trigger JIT kernel builds
  InitializeKernels(platform, ogs::Dfloat, ogs::Add);

  /********************************
   * Pairwise
   ********************************/
  ogsExchange_t* pairwise = new ogsPairwise_t(Nshared, sharedNodes,
                                           _gatherHalo, comm, platform);

  //standard copy to host - exchange - copy back to device
  pairwise->gpu_aware=false;

  double pairwiseTime[3];
  ExchangeTest(pairwise, pairwiseTime, false);
  double pairwiseAvg = pairwiseTime[0];

#ifdef GPU_AWARE_MPI
  //test GPU-aware exchange
  pairwise->gpu_aware=true;

  double pairwiseGATime[3];
  ExchangeTest(pairwise, pairwiseGATime, false);

  if (pairwiseGATime[0] < pairwiseAvg)
    pairwiseAvg = pairwiseGATime[0];
  else
    pairwise->gpu_aware=false;

#endif

  //test exchange from host memory (just for reporting)
  double pairwiseHostTime[3];
  ExchangeTest(pairwise, pairwiseHostTime, true);

  bestExchange = pairwise;
  method = Pairwise;
  bestTime = pairwiseAvg;

#ifdef GPU_AWARE_MPI
  if (rank==0 && verbose)
    printf("   Pairwise       %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            pairwiseTime[0],     pairwiseTime[1],     pairwiseTime[2],
            pairwiseGATime[0],   pairwiseGATime[1],   pairwiseGATime[2],
            pairwiseHostTime[0], pairwiseHostTime[1], pairwiseHostTime[2]);
#else
  if (rank==0 && verbose)
    printf("   Pairwise       %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            pairwiseTime[0],     pairwiseTime[1],     pairwiseTime[2],
            pairwiseHostTime[0], pairwiseHostTime[1], pairwiseHostTime[2]);
#endif

  /********************************
   * All-to-All
   ********************************/
  ogsExchange_t* alltoall = new ogsAllToAll_t(Nshared, sharedNodes,
                                           _gatherHalo, comm, platform);

  //standard copy to host - exchange - copy back to device
  alltoall->gpu_aware=false;

  double alltoallTime[3];
  ExchangeTest(alltoall, alltoallTime, false);
  double alltoallAvg = alltoallTime[0];

#ifdef GPU_AWARE_MPI
  //test GPU-aware exchange
  alltoall->gpu_aware=true;

  double alltoallGATime[3];
  ExchangeTest(alltoall, alltoallGATime, false);

  if (alltoallGATime[0] < alltoallAvg)
    alltoallAvg = alltoallGATime[0];
  else
    alltoall->gpu_aware=false;

#endif

  //test exchange from host memory (just for reporting)
  double alltoallHostTime[3];
  ExchangeTest(alltoall, alltoallHostTime, true);

  if (alltoallAvg < bestTime) {
    delete bestExchange;
    bestExchange = alltoall;
    method = AllToAll;
    bestTime = alltoallAvg;
  } else {
    delete alltoall;
  }

#ifdef GPU_AWARE_MPI
  if (rank==0 && verbose)
    printf("   AllToAll       %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            alltoallTime[0],     alltoallTime[1],     alltoallTime[2],
            alltoallGATime[0],   alltoallGATime[1],   alltoallGATime[2],
            alltoallHostTime[0], alltoallHostTime[1], alltoallHostTime[2]);
#else
  if (rank==0 && verbose)
    printf("   AllToAll       %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            alltoallTime[0],     alltoallTime[1],     alltoallTime[2],
            alltoallHostTime[0], alltoallHostTime[1], alltoallHostTime[2]);
#endif

  /********************************
   * Crystal Router
   ********************************/
  ogsExchange_t* crystal = new ogsAllToAll_t(Nshared, sharedNodes,
                                           _gatherHalo, comm, platform);

  //standard copy to host - exchange - copy back to device
  crystal->gpu_aware=false;

  double crystalTime[3];
  ExchangeTest(crystal, crystalTime, false);
  double crystalAvg = crystalTime[0];

#ifdef GPU_AWARE_MPI
  //test GPU-aware exchange
  crystal->gpu_aware=true;

  double crystalGATime[3];
  ExchangeTest(crystal, crystalGATime, false);

  if (crystalGATime[0] < crystalAvg)
    crystalAvg = crystalGATime[0];
  else
    crystal->gpu_aware=false;

#endif

  //test exchange from host memory (just for reporting)
  double crystalHostTime[3];
  ExchangeTest(crystal, crystalHostTime, true);

  if (crystalAvg < bestTime) {
    delete bestExchange;
    bestExchange = crystal;
    method = CrystalRouter;
    bestTime = crystalAvg;
  } else {
    delete crystal;
  }

#ifdef GPU_AWARE_MPI
  if (rank==0 && verbose)
    printf("   CrystalRouter  %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            crystalTime[0],     crystalTime[1],     crystalTime[2],
            crystalGATime[0],   crystalGATime[1],   crystalGATime[2],
            crystalHostTime[0], crystalHostTime[1], crystalHostTime[2]);
#else
  if (rank==0 && verbose)
    printf("   CrystalRouter  %5.3e %5.3e %5.3e    %5.3e %5.3e %5.3e \n",
            crystalTime[0],     crystalTime[1],     crystalTime[2],
            crystalHostTime[0], crystalHostTime[1], crystalHostTime[2]);
#endif

  if (rank==0 && verbose) {
    switch (method) {
      case AllToAll:
        printf("   Exchange method selected: AllToAll"); break;
      case Pairwise:
        printf("   Exchange method selected: Pairwise"); break;
      case CrystalRouter:
        printf("   Exchange method selected: CrystalRouter"); break;
      default:
        break;
    }
    if (bestExchange->gpu_aware) printf(" (GPU-aware)");
    printf("\n");
  }

  return bestExchange;
}


} //namespace ogs

} //namespace libp
