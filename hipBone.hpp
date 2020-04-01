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

#ifndef HIPBONE_HPP
#define HIPBONE_HPP 1

#include "core.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "linAlg.hpp"
#include "linearSolver.hpp"

#define DHIPBONE HIPBONE_DIR

class hipBoneSettings_t: public settings_t {
public:
  hipBoneSettings_t(const int argc, char** argv, MPI_Comm& _comm);
  void report();
};

class hipBone_t: public solver_t {
public:
  dfloat lambda;

  occa::memory o_AqL;

  occa::kernel operatorKernel;
  occa::kernel forcingKernel;

  hipBone_t() = delete;
  hipBone_t(mesh_t& _mesh, linAlg_t& _linAlg):
    solver_t(_mesh, _linAlg) {}

  ~hipBone_t();

  //setup
  static hipBone_t& Setup(mesh_t& mesh, linAlg_t& linAlg);

  void Run();

  void PlotFields(dfloat* Q, char *fileName);

  void Operator(occa::memory& o_q, occa::memory& o_Aq);
};


#endif

