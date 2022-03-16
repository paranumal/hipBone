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

#include "mesh.hpp"

namespace libp {

// ------------------------------------------------------------------------
// 1D NODES
// ------------------------------------------------------------------------
void mesh_t::Nodes1D(int _N, dfloat _r[]){
  JacobiGLL(_N, _r); //Gauss-Legendre-Lobatto nodes
}

// ------------------------------------------------------------------------
// ORTHONORMAL BASIS POLYNOMIALS
// ------------------------------------------------------------------------
void mesh_t::OrthonormalBasis1D(dfloat a, int i, dfloat &P){
  P = JacobiP(a,0,0,i); //Legendre Polynomials
}

void mesh_t::GradOrthonormalBasis1D(dfloat a, int i, dfloat &Pr){
  Pr = GradJacobiP(a,0,0,i);
}

// ------------------------------------------------------------------------
// 1D VANDERMONDE MATRICES
// ------------------------------------------------------------------------
void mesh_t::Vandermonde1D(int _N, int Npoints, dfloat _r[], dfloat V[]){

  int _Np = (_N+1);

  for(int n=0; n<Npoints; n++){
    int sk = 0;
    for(int i=0; i<=_N; i++){
      int id = n*_Np+sk;
      OrthonormalBasis1D(_r[n], i, V[id]);
      sk++;
    }
  }
}

void mesh_t::GradVandermonde1D(int _N, int Npoints, dfloat _r[], dfloat Vr[]){

  int _Np = (_N+1);

  for(int n=0; n<Npoints; n++){
    int sk = 0;
    for(int i=0; i<=_N; i++){
      int id = n*_Np+sk;
      GradOrthonormalBasis1D(_r[n], i, Vr[id]);
      sk++;
    }
  }
}

// ------------------------------------------------------------------------
// 1D OPERATOR MATRICES
// ------------------------------------------------------------------------
void mesh_t::MassMatrix1D(int _Np, dfloat V[], dfloat _MM[]){

  // masMatrix = inv(V')*inv(V) = inv(V*V')
  for(int n=0;n<_Np;++n){
    for(int m=0;m<_Np;++m){
      dfloat res = 0;
      for(int i=0;i<_Np;++i){
        res += V[n*_Np+i]*V[m*_Np+i];
      }
      _MM[n*_Np + m] = res;
    }
  }
  linAlg_t::matrixInverse(_Np, _MM);
}

void mesh_t::Dmatrix1D(int _N, int Npoints, dfloat _r[], dfloat _Dr[]){

  int _Np = _N+1;

  memory<dfloat> V(Npoints*_Np);
  memory<dfloat> Vr(Npoints*_Np);

  Vandermonde1D(_N, Npoints, _r, V.ptr());
  GradVandermonde1D(_N, Npoints, _r, Vr.ptr());

  //D = Vr/V
  linAlg_t::matrixRightSolve(_Np, _Np, Vr.ptr(), _Np, _Np, V.ptr(), _Dr);
}

// ------------------------------------------------------------------------
// 1D JACOBI POLYNOMIALS
// ------------------------------------------------------------------------
static dfloat mygamma(dfloat x){
  dfloat lgam = lgamma(x);
  dfloat gam  = signgam*exp(lgam);
  return gam;
}

dfloat mesh_t::JacobiP(dfloat a, dfloat alpha, dfloat beta, int _N){

  dfloat ax = a;

  memory<dfloat> P(_N+1);

  // Zero order
  dfloat gamma0 = pow(2,(alpha+beta+1))/(alpha+beta+1)*mygamma(1+alpha)*mygamma(1+beta)/mygamma(1+alpha+beta);
  dfloat p0     = 1.0/sqrt(gamma0);

  if (_N==0){ return p0;}
  P[0] = p0;

  // first order
  dfloat gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0;
  dfloat p1     = ((alpha+beta+2)*ax/2 + (alpha-beta)/2)/sqrt(gamma1);
  if (_N==1){ return p1;}

  P[1] = p1;

  /// Repeat value in recurrence.
  dfloat aold = 2/(2+alpha+beta)*sqrt((alpha+1.)*(beta+1.)/(alpha+beta+3.));
  /// Forward recurrence using the symmetry of the recurrence.
  for(int i=1;i<=_N-1;++i){
    dfloat h1 = 2.*i+alpha+beta;
    dfloat anew = 2./(h1+2.)*sqrt( (i+1.)*(i+1.+alpha+beta)*(i+1+alpha)*(i+1+beta)/(h1+1)/(h1+3));
    dfloat bnew = -(alpha*alpha-beta*beta)/h1/(h1+2);
    P[i+1] = 1./anew*( -aold*P[i-1] + (ax-bnew)*P[i]);
    aold =anew;
  }

  dfloat pN = P[_N];
  return pN;
}

dfloat mesh_t::GradJacobiP(dfloat a, dfloat alpha, dfloat beta, int _N){

  dfloat PNr = 0;

  if(_N>0)
    PNr = sqrt(_N*(_N+alpha+beta+1.))*JacobiP(a, alpha+1.0, beta+1.0, _N-1);

  return PNr;
}

// ------------------------------------------------------------------------
// 1D GAUSS-LEGENDRE-LOBATTO QUADRATURE
// ------------------------------------------------------------------------
void mesh_t::JacobiGLL(int _N, dfloat _x[], dfloat w[]){

  _x[0] = -1.;
  _x[_N] =  1.;

  if(_N>1){
    memory<dfloat> wtmp(_N-1);
    JacobiGQ(1,1, _N-2, _x+1, wtmp.ptr());
  }

  if (w!=nullptr) {
    int _Np = _N+1;
    memory<dfloat> _MM(_Np*_Np);
    memory<dfloat> V(_Np*_Np);

    Vandermonde1D(_N, _N+1, _x, V.ptr());
    MassMatrix1D(_N+1, V.ptr(), _MM.ptr());

    // use weights from mass lumping
    for(int n=0;n<=_N;++n){
      dfloat res = 0;
      for(int m=0;m<=_N;++m){
        res += _MM[n*(_N+1)+m];
      }
      w[n] = res;
    }
  }
}

// ------------------------------------------------------------------------
// 1D GAUSS QUADRATURE
// ------------------------------------------------------------------------
void mesh_t::JacobiGQ(dfloat alpha, dfloat beta, int _N, dfloat _x[], dfloat w[]){

  // function NGQ = JacobiGQ(alpha,beta,_N, _x, w)
  // Purpose: Compute the _N'th order Gauss quadrature points, _x,
  //          and weights, w, associated with the Jacobi
  //          polynomial, of type (alpha,beta) > -1 ( <> -0.5).
  if (_N==0){
    _x[0] = (alpha-beta)/(alpha+beta+2);
    w[0] = 2;
  }

  // Form symmetric matrix from recurrence.
  memory<dfloat> J((_N+1)*(_N+1), 0);
  memory<dfloat> h1(_N+1);

  for(int n=0;n<=_N;++n){
    h1[n] = 2*n+alpha+beta;
  }

  // J = J + J';
  for(int n=0;n<=_N;++n){
    // J = diag(-1/2*(alpha^2-beta^2)./(h1+2)./h1) + ...
    J[n*(_N+1)+n]+= -0.5*(alpha*alpha-beta*beta)/((h1[n]+2)*h1[n])*2; // *2 for symm

    //    diag(2./(h1(1:_N)+2).*sqrt((1:_N).*((1:_N)+alpha+beta).*((1:_N)+alpha).*((1:_N)+beta)./(h1(1:_N)+1)./(h1(1:_N)+3)),1);
    if(n<_N){
      J[n*(_N+1)+n+1]   += (2./(h1[n]+2.))*sqrt((n+1)*(n+1+alpha+beta)*(n+1+alpha)*(n+1+beta)/((h1[n]+1)*(h1[n]+3)));
      J[(n+1)*(_N+1)+n] += (2./(h1[n]+2.))*sqrt((n+1)*(n+1+alpha+beta)*(n+1+alpha)*(n+1+beta)/((h1[n]+1)*(h1[n]+3)));
    }
  }

  dfloat eps = 1;
  while(1+eps>1){
    eps = eps/2.;
  }
  // printf("MACHINE PRECISION %e\n", eps);

  if (alpha+beta<10*eps) J[0] = 0;

  // Compute quadrature by eigenvalue solve

  //  [V,D] = eig(J);
  memory<dfloat> WR(_N+1);
  memory<dfloat> WI(_N+1);
  memory<dfloat> VR((_N+1)*(_N+1));

  // _x = diag(D);
  linAlg_t::matrixEigenVectors(_N+1, J.ptr(), VR.ptr(), _x, WI.ptr());

  //w = (V(1,:)').^2*2^(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*.gamma(beta+1)/gamma(alpha+beta+1);
  for(int n=0;n<=_N;++n){
    w[n] = pow(VR[0*(_N+1)+n],2)*(pow(2,alpha+beta+1)/(alpha+beta+1))*mygamma(alpha+1)*mygamma(beta+1)/mygamma(alpha+beta+1);
  }

  // sloppy sort
  for(int n=0;n<=_N;++n){
    for(int m=n+1;m<=_N;++m){
      if(_x[n]>_x[m]){
        dfloat tmpx = _x[m];
        dfloat tmpw = w[m];
        _x[m] = _x[n];
        w[m] = w[n];
        _x[n] = tmpx;
        w[n] = tmpw;
      }
    }
  }
}

} //namespace libp
