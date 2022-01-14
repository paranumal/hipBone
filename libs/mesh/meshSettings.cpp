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

void meshAddSettings(settings_t& settings) {

  settings.newSetting("-nx", "--dimx",
                      "BOX NX",
                      "10",
                      "Number of elements in X-dimension per rank");
  settings.newSetting("-ny", "--dimy",
                      "BOX NY",
                      "10",
                      "Number of elements in Y-dimension per rank");
  settings.newSetting("-nz", "--dimz",
                      "BOX NZ",
                      "10",
                      "Number of elements in Z-dimension per rank");

  settings.newSetting("-px", "--procx",
                      "PROCS PX",
                      "0",
                      "Number of MPI processes in X-dimension");
  settings.newSetting("-py", "--procy",
                      "PROCS PY",
                      "0",
                      "Number of MPI processes in Y-dimension");
  settings.newSetting("-pz", "--procz",
                      "PROCS PZ",
                      "0",
                      "Number of MPI processes in Z-dimension");

  settings.newSetting("-p", "--degree",
                      "POLYNOMIAL DEGREE",
                      "4",
                      "Degree of polynomial finite element space",
                      {"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15"});
}

void meshReportSettings(settings_t& settings) {

  //report the box settings
  settings.reportSetting("BOX NX");
  settings.reportSetting("BOX NY");
  settings.reportSetting("BOX NZ");

  settings.reportSetting("POLYNOMIAL DEGREE");
}

} //namespace libp