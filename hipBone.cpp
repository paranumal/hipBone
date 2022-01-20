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

int main(int argc, char **argv){

  // start up MPI
  MPI_Init(&argc, &argv);

  MPI_Comm comm = MPI_COMM_WORLD;

  hipBoneSettings_t settings(argc, argv, comm);
  if (settings.compareSetting("VERBOSE", "TRUE"))
    settings.report();

  // set up platform
  platform_t platform(settings);

  for (int p=1;p<16;++p) {
    platform.settings().changeSetting("POLYNOMIAL DEGREE", std::to_string(p));

    //sweep through lots of tests
    std::vector<int> NN_low {  2,  2,  2,  2,  5,  4,  5,  4,  3,  3,  2,  2,  3,  3,  2};
    std::vector<int> NN_high{122,110, 98, 82, 68, 58, 50, 44, 39, 36, 32, 29, 27, 25, 24};
    std::vector<int> NN_step{ 10,  9,  8,  8,  7,  6,  5,  4,  4,  3,  3,  3,  3,  2,  2};


    // for (int N = NN_low[p-1];N<=NN_high[p-1];N+=NN_step[p-1])
    {
      int N = NN_high[p-1];

      platform.settings().changeSetting("BOX NX", std::to_string(N));
      platform.settings().changeSetting("BOX NY", std::to_string(N));
      platform.settings().changeSetting("BOX NZ", std::to_string(N));

      // set up mesh
      mesh_t mesh(platform);

      // set up hb solver
      hipBone_t hb(platform, mesh);

      // run
      hb.Run();
    }
  }

  // close down MPI
  MPI_Finalize();
  return HIPBONE_SUCCESS;
}
