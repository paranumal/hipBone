/*

The MIT License (MIT)

Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

//settings for hipBone solver
hipBoneSettings_t::hipBoneSettings_t(const int argc, char** argv, comm_t _comm):
  settings_t(_comm) {

  platformAddSettings(*this);
  meshAddSettings(*this);

  newToggle("-v", "--verbose",
            "VERBOSE",
            "FALSE",
            "Enable verbose output");

  newToggle("-ga", "--gpu-aware-mpi",
            "GPU-AWARE MPI",
            "FALSE",
            "Enable direct access of GPU memory in MPI");

  newToggle("-V", "--version",
            "VERSION",
            "FALSE",
            "Hipbone Version string");

  parseSettings(argc, argv);
}

void hipBoneSettings_t::report() {

  if (comm.rank()==0) {
    std::cout << "Settings:\n\n";
    platformReportSettings(*this);
    meshReportSettings(*this);

    reportSetting("GPU-AWARE MPI");
  }
}
