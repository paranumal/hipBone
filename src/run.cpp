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

#include "hipBone.hpp"

void hipBone_t::Run(){

  //setup linear solver
  dlong N = mesh.ogsMasked.Ngather;
  dlong Nhalo = mesh.gHalo.Nhalo;
  // linearSolver_t *linearSolver = new cg(platform, N, Nhalo);

  hlong NGlobal = mesh.ogsMasked.NgatherGlobal;
  dlong NLocal = mesh.Np*mesh.Nelements;

  //create occa buffers
  dlong Nall = N+Nhalo;
  occa::memory o_r = platform.malloc(Nall*sizeof(dfloat));
  occa::memory o_x = platform.malloc(Nall*sizeof(dfloat));
  occa::memory o_rL = platform.malloc(NLocal*sizeof(dfloat));
  occa::memory o_xL = platform.malloc(NLocal*sizeof(dfloat));

  //set x =0
  // platform.linAlg().set(Nall, 0.0, o_x);

  //NekBone-like RHS
  forcingKernel(N, o_r);

  int maxIter = 100;
  int verbose = platform.settings().compareSetting("VERBOSE", "TRUE") ? 1 : 0;
  int isLocal = platform.settings().compareSetting("OPERATOR", "LOCAL") ? 1 : 0;
  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double startTime = MPI_Wtime();

  //call the solver
  dfloat tol = 0.0;
  // int Niter = linearSolver->Solve(*this, o_x, o_r, tol, maxIter, verbose);

  int Niter=50;
  for (int i=0;i<Niter;++i) {
    if(isLocal)
      LocalOperator(o_rL, o_xL);
    else
      Operator(o_r, o_xL);
  }

  platform.device.finish();
  MPI_Barrier(mesh.comm);
  double endTime = MPI_Wtime();
  double elapsedTime = endTime - startTime;

  int Np = mesh.Np, Nq = mesh.Nq;

  hlong NunMasked = NLocal - mesh.Nmasked;
  hlong NunMaskedGlobal;
  MPI_Allreduce(&NunMasked, &NunMaskedGlobal, 1, MPI_HLONG, MPI_SUM, mesh.comm);

  hlong Ndofs = NGlobal;

  size_t NbytesAx;

  if(!isLocal)
    NbytesAx = NGlobal*sizeof(dfloat) //q
      +  (Np*7*sizeof(dfloat) // ggeo
	  +  sizeof(dlong) // localGatherElementList
	  +  Np*sizeof(dlong) // GlobalToLocal
	  +  Np*sizeof(dfloat) /*Aq*/ )*mesh.NelementsGlobal;
  else
    NbytesAx = 
      (Np*7*sizeof(dfloat) // ggeo
       +  sizeof(dlong) // localGatherElementList
       +  2*Np*sizeof(dfloat) /*q,Aq*/ )*mesh.NelementsGlobal;
    
  size_t NbytesGather =  (NGlobal+1)*sizeof(dlong) //row starts
                       + NunMaskedGlobal*sizeof(dlong) //local Ids
                       + NunMaskedGlobal*sizeof(dfloat) //AqL
                       + NGlobal*sizeof(dfloat);

  // size_t Nbytes = ( 4*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather) //first iteration
  //               + (11*Ndofs*sizeof(dfloat) + NbytesAx + NbytesGather)*Niter; //bytes per CG iteration
  size_t Nbytes = (NbytesAx)*Niter; //bytes per CG iteration

  size_t NflopsAx=( 12*Nq*Nq*Nq*Nq
                   +18*Nq*Nq*Nq)*mesh.NelementsGlobal;

  size_t NflopsGather = NunMaskedGlobal;

  // size_t Nflops =   ( 5*Ndofs + NflopsAx + NflopsGather) //first iteration
  //                 + (11*Ndofs + NflopsAx + NflopsGather)*Niter; //flops per CG iteration
  size_t Nflops =   (NflopsAx)*Niter; //flops per CG iteration

  size_t NflopsNekbone =   (15*Np  //CG flops
			    + 19*Np+12*Nq*Nq*Nq*Nq )*mesh.NelementsGlobal*Niter; //flops per CG iteration
  
  if (mesh.rank==0){
    printf("hipBone: %d, " hlongFormat ", %4.4f, %1.2e, %4.1f, %4.1f; N, DOFs, elapsed, time per DOF, BW (GB/s), GFLOPs \n",
           mesh.N,
           Ndofs,
           elapsedTime,
           elapsedTime/(Ndofs),
           Nbytes/(1.0e9 * elapsedTime),
           Nflops/(1.0e9 * elapsedTime));
  }

  o_r.free(); o_x.free();
  // delete linearSolver;
}
