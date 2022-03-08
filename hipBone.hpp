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

#include "platform.hpp"
#include "mesh.hpp"
#include "solver.hpp"
#include "linAlg.hpp"
#include "linearSolver.hpp"

#define DHIPBONE HIPBONE_DIR

using namespace libp;

class hipBoneSettings_t: public settings_t {
public:
  hipBoneSettings_t(const int argc, char** argv, comm_t _comm);
  void report();
};

class hipBone_t: public solver_t {

 public:
  mesh_t mesh;

  dfloat lambda;

  deviceMemory<dfloat> o_AqL;

  kernel_t operatorKernel;
  kernel_t forcingKernel;

  hipBone_t() = default;
  hipBone_t(platform_t& _platform, mesh_t &_mesh) {
    Setup(_platform, _mesh);
  }

  //setup
  void Setup(platform_t& _platform, mesh_t& _mesh);

  void Run();

  void Operator(deviceMemory<dfloat>& o_q, deviceMemory<dfloat>& o_Aq);
};


#endif

