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

#include "platform.hpp"
#include <hwloc.h>
#include <omp.h>

namespace libp {

// OCCA build stuff
void platform_t::DeviceConfig(){

  //find out how many ranks and devices are on this system
  memory<char> hostnames(size()*MAX_PROCESSOR_NAME);
  memory<char> hostname = hostnames + rank()*MAX_PROCESSOR_NAME;

  int namelen;
  comm_t::GetProcessorName(hostname.ptr(), namelen);
  comm.Allgather(hostnames, MAX_PROCESSOR_NAME);

  int localRank = 0;
  int localSize = 0;
  for (int n=0; n<rank(); n++){
    if (!strcmp(hostname.ptr(), hostnames.ptr()+n*MAX_PROCESSOR_NAME)) localRank++;
  }
  for (int n=0; n<size(); n++){
    if (!strcmp(hostname.ptr(), hostnames.ptr()+n*MAX_PROCESSOR_NAME)) localSize++;
  }

  int plat=0;
  int device_id=0;

  settings_t& Settings = settings();

  if(Settings.compareSetting("THREAD MODEL", "OpenCL"))
    Settings.getSetting("PLATFORM NUMBER", plat);

  // read thread model/device/platform from Settings
  std::string mode;

  if(Settings.compareSetting("THREAD MODEL", "CUDA")){
    mode = "{mode: 'CUDA'}";
  }
  else if(Settings.compareSetting("THREAD MODEL", "HIP")){
    mode = "{mode: 'HIP'}";
  }
  else if(Settings.compareSetting("THREAD MODEL", "OpenCL")){
    mode = "{mode: 'OpenCL', platform_id : " + std::to_string(plat) +"}";
  }
  else if(Settings.compareSetting("THREAD MODEL", "OpenMP")){
    mode = "{mode: 'OpenMP'}";
  }
  else{
    mode = "{mode: 'Serial'}";
  }

  //add a device_id number for some modes
  if (  Settings.compareSetting("THREAD MODEL", "CUDA")
      ||Settings.compareSetting("THREAD MODEL", "HIP")
      ||Settings.compareSetting("THREAD MODEL", "OpenCL")) {
    //for testing a single device, run with 1 rank and specify DEVICE NUMBER
    if (size()==1) {
      Settings.getSetting("DEVICE NUMBER",device_id);
    } else {

      device_id = localRank;

      //check for over-subscribing devices
      int deviceCount = getDeviceCount(mode);
      if (deviceCount>0 && localRank>=deviceCount) {
        LIBP_FORCE_WARNING("Rank " << rank() << " oversubscribing device " << device_id%deviceCount << " on node \"" << hostname.ptr()<< "\"");
        device_id = device_id%deviceCount;
      }
    }

    // add device_id to setup string
    mode.pop_back();
    mode += ", device_id: " + std::to_string(device_id) + "}";
  }

#if !defined(LIBP_DEBUG)
  /*set number of omp threads to use*/
  /*Use hwloc to determine physical core count */
  hwloc_topology_t topology;
  hwloc_topology_init(&topology);
  hwloc_topology_load(topology);

  int depth = hwloc_get_type_depth(topology, HWLOC_OBJ_CORE);
  int NcoresPerNode;
  if (depth == HWLOC_TYPE_DEPTH_UNKNOWN) {
    // Default to 1 if there's a problem.
    NcoresPerNode = 1;
  }
  else {
    NcoresPerNode = hwloc_get_nbobjs_by_depth(topology, depth);
  }

  // Clean up
  hwloc_topology_destroy(topology);

  int Nthreads=0;

  /*Check OMP_NUM_THREADS env variable*/
  std::string ompNumThreads;
  char * ompEnvVar = std::getenv("OMP_NUM_THREADS");
  if (ompEnvVar == nullptr) { // Environment variable is not set
    Nthreads = std::max(NcoresPerNode/localSize, 1); //Evenly divide number of cores

    // If omp max threads is lower than this (due to binding), go with omp
    Nthreads = std::min(Nthreads, omp_get_max_threads());
  } else {
    ompNumThreads = ompEnvVar;
    // Environmet variable is set, but could be empty string
    if (ompNumThreads.size() == 0) {
      // Environment variable is set but equal to empty string
      Nthreads = std::max(NcoresPerNode/localSize, 1); //Evenly divide number of cores;

      // If omp max threads is lower than this (due to binding), go with omp
      Nthreads = std::min(Nthreads, omp_get_max_threads());
    } else {
      Nthreads = std::stoi(ompNumThreads);
    }
  }
  LIBP_WARNING("Rank " << rank() << " oversubscribing CPU on node \"" << hostname.ptr()<< "\"",
                  Nthreads*localSize>NcoresPerNode);
  omp_set_num_threads(Nthreads);
  // omp_set_num_threads(1);

  // if (settings.compareSetting("VERBOSE","TRUE"))
  //   printf("Rank %d: Nsockets = %d, NcoresPerSocket = %d, Nthreads = %d, device_id = %d \n",
  //          rank, Nsockets, Ncores, Nthreads, device_id);
#endif

  device.setup(mode);

  std::string cacheDir;
  char * cacheEnvVar = std::getenv("HIPBONE_CACHE_DIR");
  if (cacheEnvVar == nullptr) {
    // Environment variable is not set
    cacheDir = exePath() + ".occa";
  }
  else {
    // Environmet variable is set, but could be empty string
    cacheDir = cacheEnvVar;

    if (cacheDir.size() == 0) {
      // Environment variable is set but equal to empty string
      cacheDir = exePath() + ".occa";
    }
  }
  setCacheDir(cacheDir);

  comm.Barrier();
}

} //namespace libp
