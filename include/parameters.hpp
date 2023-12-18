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

#ifndef PARAMETERS_HPP
#define PARAMETERS_HPP

#include "core.hpp"
#include "comm.hpp"

namespace libp {

// Class for loading a list of tuning parameters from a .json file and
// finding a best match for a given set of user-provided runtime options
class parameters_t {
 public:
  // Load a list of kernel parameters from a .json file
  void load(std::string filename, comm_t& comm);

  // Find best match for a set of keys in list of loaded parameters
  properties_t findProperties(std::string name, properties_t& keys);

  //Convert a property to a single line string
  std::string toString(properties_t& prop);

 private:
  std::vector<properties_t> dataBase;
};

} //namespace libp

#endif
