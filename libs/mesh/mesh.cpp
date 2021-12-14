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

void mesh_t::Setup(platform_t& _platform) {

  platform = _platform;

  comm = platform.comm;
  MPI_Comm_rank(platform.comm, &rank);
  MPI_Comm_size(platform.comm, &size);

  platform.settings().getSetting("POLYNOMIAL DEGREE", N);

  dim = 3;
  Nverts = 8; // number of vertices per element
  Nfaces = 6;
  NfaceVertices = 4;

  // vertices on each face
  int _faceVertices[6][4] =
    {{0,1,2,3},{0,4,5,1},{1,5,6,2},{2,6,7,3},{0,3,7,4},{4,7,6,5}};

  faceVertices.malloc(NfaceVertices*Nfaces);
  faceVertices.copyFrom(_faceVertices[0]);

  // reference element nodes and operators
  ReferenceNodes();

  //build a box mesh
  SetupBox();

  // connect elements using parallel sort
  Connect();

  // set up halo exchange info for MPI (do before connect face nodes)
  HaloSetup();

  // connect face vertices
  ConnectFaceVertices();

  // compute physical (x,y,z) locations of the element nodes
  // PhysicalNodes();

  // connect face nodes (find trace indices)
  ConnectFaceNodes();

  // make a global indexing
  ConnectNodes();

  // make an ogs operator and label local/global gather elements
  GatherScatterSetup();

  // compute geometric factors
  GeometricFactors();

  OccaSetup();
}

} //namespace libp