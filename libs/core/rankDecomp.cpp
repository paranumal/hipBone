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

#include "core.hpp"

namespace libp {

// A function to find largest prime factor
static int maxPrimeFactor(int n) {
  int p = -1;

  // Print the number of 2s that divide n
  while (n % 2 == 0) {
    p = 2;
    n >>= 1; // equivalent to n /= 2
  }
  // n must be odd at this point
  while (n % 3 == 0) {
    p = 3;
    n=n/3;
  }

  // now we have to iterate only for integers
  // who does not have prime factor 2 and 3
  for (int i = 5; i <= sqrt(n); i += 6) {
    while (n % i == 0) {
      p = i;
      n = n / i;
    }
    while (n % (i+2) == 0) {
      p = i+2;
      n = n / (i+2);
    }
  }

  // This condition is to handle the case
  // when n is a prime number greater than 4
  if (n > 4) p = n;

  return p;
}

/*Determine the (x,y,z) coordinates in MPI grid for this process rank*/
void RankDecomp(int  size_x, int  size_y, int  size_z,
                int &rank_x, int &rank_y, int &rank_z,
                const int rank) {

  int size = size_x*size_y*size_z;

  if (size==1) {
    rank_x=0;
    rank_y=0;
    rank_z=0;
    return;
  }

  /*Determine coordinates via recursive factorization*/
  if (size_z>=size_x && size_z>=size_y) { //size_z is largest

    const int p = maxPrimeFactor(size_z);
    const int csize = size/p;
    const int crank = rank%csize;

    /*Recursive call*/
    int crank_z=-1;
    RankDecomp(size_x, size_y, size_z/p,
               rank_x, rank_y, crank_z, crank);
    rank_z = crank_z + (rank/csize)*(size_z/p);

  } else if (size_y>=size_x && size_y>=size_z) { //size_y is largest
    const int p = maxPrimeFactor(size_y);
    const int csize = size/p;
    const int crank = rank%csize;

    /*Recursive call*/
    int crank_y=-1;
    RankDecomp(size_x, size_y/p, size_z,
               rank_x, crank_y, rank_z, crank);
    rank_y = crank_y + (rank/csize)*(size_y/p);
  } else { //size_x is largest
    const int p = maxPrimeFactor(size_x);
    const int csize = size/p;
    const int crank = rank%csize;

    /*Recursive call*/
    int crank_x=-1;
    RankDecomp(size_x/p, size_y, size_z,
               crank_x, rank_y, rank_z, crank);
    rank_x = crank_x + (rank/csize)*(size_x/p);
  }
}

} //namespace libp
