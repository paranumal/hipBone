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

int main(int argc, char **argv){

  // start up MPI
  comm_t::Init(argc, argv);

  { /*Scope so everything is destructed before MPI_Finalize */
    comm_t comm(comm_t::world().Dup());

    hipBoneSettings_t settings(argc, argv, comm);
    if (settings.compareSetting("VERBOSE", "TRUE"))
      settings.report();

    if (settings.compareSetting("VERSION", "TRUE") && comm.rank()==0) {
      std::cout << "hipBone " << HIPBONE_VERSION_STR << std::endl;
    }

    //Toggle GPU-aware MPI functionality
    comm.setGpuAware(settings.compareSetting("GPU-AWARE MPI", "TRUE"));
    settings.comm.setGpuAware(comm.gpuAware());

    // set up platform
    platform_t platform(settings);

    // set up mesh
    mesh_t mesh(platform);

    // set up hb solver
    hipBone_t hb(platform, mesh);

    // run
    hb.Run();
  }

  // close down MPI
  comm_t::Finalize();
  return LIBP_SUCCESS;
}
