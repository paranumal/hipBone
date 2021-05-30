/*

The MIT License (MIT)

Copyright (c) 2017 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus

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

#ifndef PLATFORM_HPP
#define PLATFORM_HPP

#define HIPBONE_MAJOR_VERSION 1
#define HIPBONE_MINOR_VERSION 0
#define HIPBONE_PATCH_VERSION 0
#define HIPBONE_VERSION       10000
#define HIPBONE_VERSION_STR   "1.0.0"

#include "core.hpp"
#include "settings.hpp"
#include "linAlg.hpp"

void platformAddSettings(settings_t& settings);
void platformReportSettings(settings_t& settings);

class platform_t {
public:
  const MPI_Comm& comm;
  settings_t& settings;
  occa::properties props;

  occa::device device;
  linAlg_t linAlg;

  int rank, size;

  platform_t(settings_t& _settings):
    comm(_settings.comm),
    settings(_settings) {

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DeviceConfig();
    DeviceProperties();

    linAlg.Setup(this);
  }

  ~platform_t(){}

  occa::kernel buildKernel(std::string fileName, std::string kernelName,
                           occa::properties& kernelInfo);

  occa::memory malloc(const size_t bytes,
                      const void *src = NULL,
                      const occa::properties &prop = occa::properties()) {
    return device.malloc(bytes, src, prop);
  }

  occa::memory malloc(const size_t bytes,
                      const occa::memory &src,
                      const occa::properties &prop = occa::properties()) {
    return device.malloc(bytes, src, prop);
  }

  occa::memory malloc(const size_t bytes,
                      const occa::properties &prop) {
    return device.malloc(bytes, prop);
  }

  void *hostMalloc(const size_t bytes,
                   const void *src,
                   occa::memory &h_mem){
    occa::properties hostProp;
    hostProp["host"] = true;
    h_mem = device.malloc(bytes, src, hostProp);
    return h_mem.ptr();
  }

private:
  void DeviceConfig();
  void DeviceProperties();

};

#endif