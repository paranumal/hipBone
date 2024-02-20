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

#include "parameters.hpp"
#include "memory.hpp"
#include <fstream>

namespace libp {

// Load a list of kernel parameters from a .json file
void parameters_t::load(std::string filename, comm_t& comm) {

  if (comm.rank()==0) {
    std::ifstream file;
    file.open(filename, std::ios::in);

    LIBP_ABORT("Unable to open file: " << filename, !file);

    while (!file.eof()) {
      std::string str;
      std::getline(file, str);

      if (!str.length()) continue;

      properties_t entry;
      entry.load(str);

      if (!entry.isInitialized() || entry.isNull()) continue;

      dataBase.push_back(entry);
    }

    file.close();
  }

  if (comm.size()==1) return;

  int Nentries = (comm.rank()==0) ? dataBase.size() : 0;

  comm.Bcast(Nentries, 0);

  for (int i=0; i<Nentries; ++i) {
    if (comm.rank()==0) {
      std::string str = toString(dataBase[i]);
      int length = str.length()+1;
      memory<char> c_str(length);
      comm.Bcast(length, 0);

      c_str.copyFrom(str.c_str());
      comm.Bcast(c_str, 0);
    } else {
      int length = 0;
      comm.Bcast(length, 0);

      memory<char> c_str(length);
      comm.Bcast(c_str, 0);

      properties_t entry;
      entry.load(c_str.ptr());
      dataBase.push_back(entry);
    }
  }
}

// Find best match for a set of keys in list of loaded parameters
properties_t parameters_t::findProperties(std::string name,
                                          properties_t& keys) {

  properties_t *props = nullptr;

  int bestNmatches = -1;

  for (properties_t& entry : dataBase) {
    //Match name first
    if (entry["Name"].string() != name) continue;

    properties_t& entryKeys = entry["keys"];

    std::vector<std::string> entryKeyNames = entryKeys.keys();

    //Check keys for matches
    int Nmatches = 0;
    for (std::string& key : entryKeyNames) {
      if (keys.has(key)) {
        if (keys[key].isString()) {
          //Match string entries if the dataBase string contains the user's
          // input as a substring
          std::string& entryValue = entryKeys[key].string();
          std::string& keyValue = keys[key].string();
          if (entryValue.find(keyValue) != std::string::npos) {
            ++Nmatches;
          } else {
            Nmatches = -1;
            break; //if no match, this entry is incompatible
          }
        } else {
          if (entryKeys[key] == keys[key]) {
            ++Nmatches;
          } else {
            Nmatches = -1;
            break; //if no match, this entry is incompatible
          }
        }
      }
    }

    if (Nmatches > bestNmatches) {
      bestNmatches = Nmatches;
      props = &entry;
    }
  }

  LIBP_ABORT("Unable to find parameters compatible with Name = " << name
             << ", and keys: " << toString(keys),
             !props);

  return *props;
}

//Convert a property to a single line string
std::string parameters_t::toString(properties_t& prop) {
  std::string s = prop.toString();
  s.erase(std::remove(s.begin(), s.end(), '\n'), s.cend());
  s.erase(std::remove(s.begin(), s.end(), ' '), s.cend());
  return s;
}

} //namespace libp

