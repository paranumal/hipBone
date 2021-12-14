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

namespace libp {

void platformAddSettings(settings_t& settings);
void platformReportSettings(settings_t& settings);

namespace internal {

class iplatform_t {
public:
  settings_t settings;
  occa::properties props;

  iplatform_t(settings_t& _settings):
    settings(_settings) {
  }
};

} //namespace internal

class platform_t {
public:
  MPI_Comm comm = MPI_COMM_NULL;
  std::shared_ptr<internal::iplatform_t> iplatform;

  occa::device device;
  std::shared_ptr<linAlg_t> ilinAlg;

  int rank=0, size=0;

  platform_t()=default;

  platform_t(settings_t& settings) {

    iplatform = std::make_shared<internal::iplatform_t>(settings);

    comm = settings.comm;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DeviceConfig();
    DeviceProperties();

    ilinAlg = std::make_shared<linAlg_t>(this);
  }

  platform_t(const platform_t &other)=default;
  platform_t& operator = (const platform_t &other)=default;

  bool isInitialized() {
    return (iplatform!=nullptr);
  }

  void assertInitialized() {
    if(!isInitialized()) {
      HIPBONE_ABORT("Platform not initialized.");
    }
  }

  occa::kernel buildKernel(std::string fileName, std::string kernelName,
                           occa::properties& kernelInfo);

  occa::memory malloc(const size_t bytes,
                      const void *src = NULL,
                      const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return device.malloc(bytes, src, prop);
  }

  occa::memory malloc(const size_t bytes,
                      const occa::memory &src,
                      const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return device.malloc(bytes, src, prop);
  }

  occa::memory malloc(const size_t bytes,
                      const occa::properties &prop) {
    assertInitialized();
    return device.malloc(bytes, prop);
  }

  template <typename T>
  occa::memory malloc(const size_t count,
                      const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return device.malloc(count*sizeof(T), prop);
  }

  template <typename T>
  occa::memory malloc(const size_t count,
                      const libp::memory<T> &src,
                      const occa::properties &prop) {
    assertInitialized();
    return device.malloc(count*sizeof(T), src.ptr(), prop);
  }

  template <typename T>
  occa::memory malloc(const libp::memory<T> &src,
                      const occa::properties &prop = occa::properties()) {
    assertInitialized();
    return device.malloc(src.length()*sizeof(T), src.ptr(), prop);
  }

  void *hostMalloc(const size_t bytes,
                   const void *src,
                   occa::memory &h_mem){
    assertInitialized();
    occa::properties hostProp;
    hostProp["host"] = true;
    h_mem = device.malloc(bytes, src, hostProp);
    return h_mem.ptr();
  }

  linAlg_t& linAlg() {
    assertInitialized();
    return *ilinAlg;
  }

  settings_t& settings() {
    assertInitialized();
    return iplatform->settings;
  }

  occa::properties& props() {
    assertInitialized();
    return iplatform->props;
  }

private:
  void DeviceConfig();
  void DeviceProperties();

};

} //namespace libp

#endif