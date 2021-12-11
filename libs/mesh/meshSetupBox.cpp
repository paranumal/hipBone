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

void mesh_t::SetupBox(){

  //local grid physical sizes
  //Hard code to 2x2x2
  dfloat DIMX=2.0, DIMY=2.0, DIMZ=2.0;

  //number of local elements in each dimension
  dlong nx, ny, nz;
  settings.getSetting("BOX NX", nx);
  settings.getSetting("BOX NY", ny);
  settings.getSetting("BOX NZ", nz);

  // Here size_x, size_y, size_z are the number of ranks in each dimension.
  // If user does not provide them then they will default to 0.
  int size_x, size_y, size_z;
  settings.getSetting("PROCS PX", size_x);
  settings.getSetting("PROCS PY", size_y);
  settings.getSetting("PROCS PZ", size_z);

  // Check if the provided number of processes in each direction multiplies to
  // the correct global number of processes.  If not, abort.
  //
  // If the user doesn't provide this information then these default to 0.
  if ((size_x == 0) && (size_y == 0) && (size_z == 0)) {

    // If the user provided no per-dimension process information, default to
    // the cubic decomposition.

    // size is total number of ranks and populated by the mpi communicator
    size_x = std::cbrt(size); //number of ranks in each dimension
    if (size_x*size_x*size_x != size)
      HIPBONE_ABORT(std::string("3D BOX mesh requires a cubic number of MPI ranks since px,py,pz have not been provided."))

    size_y = size_x;
    size_z = size_x;

  }
  else if (size_x * size_y * size_z != size) {
    // User provided only *some* of px, py, pz, so check if they multiply to
    // the right thing.

    HIPBONE_ABORT(std::string("3D BOX mesh requires the user specifies all of px, py, pz, or none of px, py, pz.  If all are provided, their product must equal the total number of MPI ranks"))
  }

  //local grid physical sizes
  dfloat dimx = DIMX/size_x;
  dfloat dimy = DIMY/size_y;
  dfloat dimz = DIMZ/size_z;

  //rank coordinates
  int rank_z = rank / (size_x*size_y);
  int rank_y = (rank - rank_z*size_x*size_y) / size_x;
  int rank_x = rank % size_x;

  //bottom corner of physical domain
  dfloat X0 = -DIMX/2.0 + rank_x*dimx;
  dfloat Y0 = -DIMY/2.0 + rank_y*dimy;
  dfloat Z0 = -DIMZ/2.0 + rank_z*dimz;

  // damon: the run-time parameters 'weak-scale'
  //global number of elements in each dimension
  hlong NX = size_x*nx;
  hlong NY = size_y*ny;
  hlong NZ = size_z*nz;

  //global number of nodes in each dimension
  hlong NnX = NX+1; //lose a node when periodic (repeated node)
  hlong NnY = NY+1; //lose a node when periodic (repeated node)
  hlong NnZ = NZ+1; //lose a node when periodic (repeated node)

  // build an nx x ny x nz box grid
  Nnodes = NnX*NnY*NnZ; //global node count
  Nelements = nx*ny*nz; //local

  // this stores the element to vertex mapping
  EToV.malloc(Nelements*Nverts);

  // these store the element to vertex mappings in each dimension
  EX.malloc(Nelements*Nverts);
  EY.malloc(Nelements*Nverts);
  EZ.malloc(Nelements*Nverts);

  dlong e = 0;
  dfloat dx = dimx/nx;
  dfloat dy = dimy/ny;
  dfloat dz = dimz/nz;
  for(int k=0;k<nz;++k){
    for(int j=0;j<ny;++j){
      for(int i=0;i<nx;++i){

        // The reason the % is here is for the periodic case.  Literally ignore
        // it
        const hlong i0 = i+rank_x*nx;
        const hlong i1 = (i+1+rank_x*nx)%NnX;
        const hlong j0 = j+rank_y*ny;
        const hlong j1 = (j+1+rank_y*ny)%NnY;
        const hlong k0 = k+rank_z*nz;
        const hlong k1 = (k+1+rank_z*nz)%NnZ;

        EToV[e*Nverts+0] = i0 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+1] = i1 + j0*NnX + k0*NnX*NnY;
        EToV[e*Nverts+2] = i1 + j1*NnX + k0*NnX*NnY;
        EToV[e*Nverts+3] = i0 + j1*NnX + k0*NnX*NnY;

        EToV[e*Nverts+4] = i0 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+5] = i1 + j0*NnX + k1*NnX*NnY;
        EToV[e*Nverts+6] = i1 + j1*NnX + k1*NnX*NnY;
        EToV[e*Nverts+7] = i0 + j1*NnX + k1*NnX*NnY;

        dfloat x0 = X0 + dx*i;
        dfloat y0 = Y0 + dy*j;
        dfloat z0 = Z0 + dz*k;

        dfloat *ex = EX.ptr()+e*Nverts;
        dfloat *ey = EY.ptr()+e*Nverts;
        dfloat *ez = EZ.ptr()+e*Nverts;

        ex[0] = x0;    ey[0] = y0;    ez[0] = z0;
        ex[1] = x0+dx; ey[1] = y0;    ez[1] = z0;
        ex[2] = x0+dx; ey[2] = y0+dy; ez[2] = z0;
        ex[3] = x0;    ey[3] = y0+dy; ez[3] = z0;

        ex[4] = x0;    ey[4] = y0;    ez[4] = z0+dz;
        ex[5] = x0+dx; ey[5] = y0;    ez[5] = z0+dz;
        ex[6] = x0+dx; ey[6] = y0+dy; ez[6] = z0+dz;
        ex[7] = x0;    ey[7] = y0+dy; ez[7] = z0+dz;

        e++;
      }
    }
  }
}

} //namespace libp