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

#ifndef MESH_HPP
#define MESH_HPP 1

#include "core.hpp"
#include "settings.hpp"
#include "ogs.hpp"

namespace libp {

class mesh_t {

public:
  platform_t platform;
  occa::properties props;

  MPI_Comm comm;
  int rank, size;

  int dim;
  int Nverts, Nfaces, NfaceVertices;

  hlong Nnodes; //global number of element vertices
  libp::memory<dfloat> EX; // coordinates of vertices for each element
  libp::memory<dfloat> EY;
  libp::memory<dfloat> EZ;

  dlong Nelements;       //local element count
  hlong NelementsGlobal; //global element count
  libp::memory<hlong> EToV; // element-to-vertex connectivity
  libp::memory<dlong> EToE; // element-to-element connectivity
  libp::memory<int>   EToF; // element-to-(local)face connectivity
  libp::memory<int>   EToP; // element-to-partition/process connectivity

  libp::memory<dlong> VmapM;  // list of vertices on each face
  libp::memory<dlong> VmapP;  // list of vertices that are paired with face vertices

  // MPI halo exchange info
  ogs::halo_t halo;            // halo exchange pointer
  dlong NinternalElements; // number of elements that can update without halo exchange
  dlong NhaloElements;     // number of elements that cannot update without halo exchange
  dlong  totalHaloPairs;   // number of elements to be received in halo exchange
  libp::memory<dlong> internalElementIds;  // list of elements that can update without halo exchange
  libp::memory<dlong> haloElementIds;      // list of elements to be sent in halo exchange
  occa::memory o_internalElementIds;  // list of elements that can update without halo exchange
  occa::memory o_haloElementIds;      // list of elements to be sent in halo exchange

  // CG gather-scatter info
  ogs::ogs_t ogsMasked;
  ogs::halo_t gHalo;
  libp::memory<hlong> globalIds, maskedGlobalIds, maskedGlobalNumbering;
  dlong Nmasked;

  libp::memory<dlong> GlobalToLocal;
  occa::memory o_GlobalToLocal;

  // list of elements that are needed for global gather-scatter
  dlong NglobalGatherElements;
  libp::memory<dlong> globalGatherElementList;
  occa::memory o_globalGatherElementList;

  // list of elements that are not needed for global gather-scatter
  dlong NlocalGatherElements;
  libp::memory<dlong> localGatherElementList;
  occa::memory o_localGatherElementList;


  libp::memory<dfloat> weight, weightG;
  occa::memory o_weight, o_weightG;

  // second order volume geometric factors
  dlong Nggeo;
  libp::memory<dfloat> ggeo;

  // volume node info
  int N, Nq, Np;
  libp::memory<dfloat> r, s, t;    // coordinates of local nodes
  libp::memory<dfloat> x, y, z;    // coordinates of physical nodes

  // indices of vertex nodes
  libp::memory<int> vertexNodes;

  libp::memory<dfloat> D; // 1D differentiation matrix (for tensor-product)
  libp::memory<dfloat> gllz; // 1D GLL quadrature nodes
  libp::memory<dfloat> gllw; // 1D GLL quadrature weights

  // face node info
  int Nfp;        // number of nodes per face
  libp::memory<int> faceNodes; // list of element reference interpolation nodes on element faces
  libp::memory<dlong> vmapM;     // list of volume nodes that are face nodes
  libp::memory<dlong> vmapP;     // list of volume nodes that are paired with face nodes
  libp::memory<int> faceVertices; // list of mesh vertices on each face

  // occa stuff
  occa::stream defaultStream;

  occa::memory o_D; // tensor product differentiation matrix (for Hexes)
  occa::memory o_ggeo; // second order geometric factors

  mesh_t()=default;
  mesh_t(platform_t& _platform) {
    Setup(_platform);
  }
  ~mesh_t()=default;

  void Setup(platform_t& _platform);

  // box mesh
  void SetupBox();

  /* build parallel face connectivity */
  void Connect();

  void ReferenceNodes();

  /* compute x,y,z coordinates of each node */
  void PhysicalNodes();

  // compute geometric factors for local to physical map
  void GeometricFactors();

  // serial face-node to face-node connection
  void ConnectFaceNodes();

  // setup halo region
  void HaloSetup();

  // serial face-vertex to face-vertex connection
  void ConnectFaceVertices();

  /* build global connectivity in parallel */
  void ConnectNodes();

  /* build global gather scatter ops */
  void GatherScatterSetup();

  void OccaSetup();

protected:
  //1D
  void Nodes1D(int N, dfloat r[]);
  void OrthonormalBasis1D(dfloat a, int i, dfloat &P);
  void GradOrthonormalBasis1D(dfloat a, int i, dfloat &Pr);
  void Vandermonde1D(int N, int Npoints, dfloat r[], dfloat V[]);
  void GradVandermonde1D(int N, int Npoints, dfloat r[], dfloat Vr[]);

  void MassMatrix1D(int _Np, dfloat V[], dfloat MM[]);
  void Dmatrix1D(int N, int Npoints, dfloat r[], dfloat Dr[]);

  //Jacobi polynomial evaluation
  dfloat JacobiP(dfloat a, dfloat alpha, dfloat beta, int N);
  dfloat GradJacobiP(dfloat a, dfloat alpha, dfloat beta, int N);

  //Gauss-Legendre-Lobatto quadrature nodes
  void JacobiGLL(int N, dfloat x[], dfloat w[]=nullptr);

  //Nth order Gauss-Jacobi quadrature nodes and weights
  void JacobiGQ(dfloat alpha, dfloat beta, int N, dfloat x[], dfloat w[]);

  //Hexs
  void NodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[]);
  void FaceNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[],  int _faceNodes[]);
  void VertexNodesHex3D(int _N, dfloat _r[], dfloat _s[], dfloat _t[], int _vertexNodes[]);

  /* offsets for second order geometric factors */
  static constexpr int GWJID=0;
  static constexpr int G00ID=1;
  static constexpr int G01ID=2;
  static constexpr int G11ID=3;
  static constexpr int G12ID=4;
  static constexpr int G02ID=5;
  static constexpr int G22ID=6;
};

void meshAddSettings(settings_t& settings);
void meshReportSettings(settings_t& settings);

} //namespace libp

#endif

