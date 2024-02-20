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

#include "settings.hpp"

namespace libp {

using std::vector;
using std::string;

setting_t::setting_t(string shortkey_, string longkey_,
                     string name_, string val_,
                     string description_, vector<string> options_,
                     bool isToggle_)
  : shortkey{shortkey_}, longkey{longkey_},
    name{name_}, val{val_},
    description{description_}, options{options_},
    isToggle{isToggle_}, check{0} {}

void setting_t::updateVal(const string newVal){
  if (!options.size()) {
    val = newVal;
  } else {
    for (size_t i=0;i<options.size();i++) {
      if (newVal==options[i]) {//valid
        val = newVal;
        return;
      }
    }
    stringstream ss;
    ss << "Value: \"" << newVal << "\" "
       << "not valid for setting " << name <<std::endl
       << "Possible values are: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
    LIBP_FORCE_ABORT(ss.str());
  }
}

bool setting_t::compareVal(const string token) const {
  return !(val.find(token) == std::string::npos);
}

string setting_t::toString() const {
  stringstream ss;

  ss << "Name:     [" << name << "]" << std::endl;
  ss << "CL keys:  [" << shortkey << ", " << longkey << "]" << std::endl;
  ss << "Value:    " << val << std::endl;

  if (!description.empty())
    ss << "Description: " << description << std::endl;

  if (options.size()) {
    ss << "Possible values: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
  }

  return ss.str();
}

string setting_t::PrintUsage() const {
  stringstream ss;

  ss << "Name:     [" << name << "]" << std::endl;
  ss << "CL keys:  [" << shortkey << ", " << longkey << "]" << std::endl;

  if (!description.empty())
    ss << "Description: " << description << std::endl;

  if (options.size()) {
    ss << "Possible values: { ";
    for (size_t i=0;i<options.size()-1;i++) ss << options[i] << ", ";
    ss << options[options.size()-1] << " }" << std::endl;
  }

  return ss.str();
}

std::ostream& operator<<(std::ostream& os, const setting_t& setting) {
  os << setting.toString();
  return os;
}

settings_t::settings_t(comm_t _comm):
  comm(_comm) {}

void settings_t::newToggle(const string shortkey, const string longkey,
                           const string name, const string val,
                           const string description) {

  for(auto it = settings.begin(); it != settings.end(); ++it) {
    setting_t &setting = it->second;
    LIBP_ABORT("Setting with key: [" << shortkey << "] already exists.",
                  !setting.shortkey.compare(shortkey));
    LIBP_ABORT("Setting with key: [" << longkey << "] already exists.",
                  !setting.longkey.compare(longkey));
  }

  auto search = settings.find(name);
  if (search == settings.end()) {
    settings[name] = setting_t(shortkey, longkey, name, val, description,
                               {}, true);
    insertOrder.push_back(name);
  } else {
    LIBP_FORCE_ABORT("Setting with name: [" << name << "] already exists.");
  }
}

void settings_t::newSetting(const string shortkey, const string longkey,
                            const string name, const string val,
                            const string description,
                            const vector<string> options) {

  for(auto it = settings.begin(); it != settings.end(); ++it) {
    setting_t &setting = it->second;
    LIBP_ABORT("Setting with key: [" << shortkey << "] already exists.",
                  !setting.shortkey.compare(shortkey));
    LIBP_ABORT("Setting with key: [" << longkey << "] already exists.",
                  !setting.longkey.compare(longkey));
  }

  auto search = settings.find(name);
  if (search == settings.end()) {
    settings[name] = setting_t(shortkey, longkey, name, val, description, options);
    insertOrder.push_back(name);
  } else {
    LIBP_FORCE_ABORT("Setting with name: [" << name << "] already exists.");
  }
}

void settings_t::changeSetting(const string name, const string newVal) {
  auto search = settings.find(name);
  if (search != settings.end()) {
    setting_t& val = search->second;
    val.updateVal(newVal);
  } else {
    LIBP_FORCE_ABORT("Setting with name: [" << name << "] does not exist.");
  }
}

void settings_t::parseSettings(const int argc, char** argv) {

  for (int i = 1; i < argc; ) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      if (comm.rank()==0) PrintUsage();
      comm.Barrier();
      comm_t::Finalize();
      std::exit(LIBP_SUCCESS);
      return;
    }

    auto it = settings.begin();
    for(; it != settings.end(); ++it) {
      setting_t &setting = it->second;
      if (strcmp(argv[i], setting.shortkey.c_str()) == 0 ||
          strcmp(argv[i], setting.longkey.c_str()) == 0) {
        if (setting.check!=0) {
          LIBP_FORCE_ABORT("Cannot set setting [" << setting.name << "] twice in run command.");
        }

        if (setting.isToggle) {
          changeSetting(setting.name, "TRUE");
          i++;
        } else {
          changeSetting(setting.name, string(argv[i+1]));
          i+=2;
        }
        setting.check=1;
        break;
      }
    }

    if (it == settings.end()) {
      LIBP_FORCE_ABORT("Unrecognized setting [" << argv[i] << "]");
    }
  }
}

string settings_t::getSetting(const string name) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    const setting_t& val = search->second;
    return val.getVal<string>();
  } else {
    LIBP_FORCE_ABORT("Unable to find setting: [" << name << "]");
    return string();
  }
}

bool settings_t::compareSetting(const string name, const string token) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    const setting_t& val = search->second;
    return val.compareVal(token);
  } else {
    LIBP_FORCE_ABORT("Unable to find setting: [" << name.c_str() << "]");
    return false;
  }
}

void settings_t::report() {
  std::cout << "Settings:\n\n";
  for (size_t i = 0; i < insertOrder.size(); ++i) {
    const string &s = insertOrder[i];
    const setting_t& val = settings[s];
    std::cout << val << std::endl;
  }
}

void settings_t::reportSetting(const string name) const {
  auto search = settings.find(name);
  if (search != settings.end()) {
    const setting_t& val = search->second;
    std::cout << val << std::endl;
  } else {
    LIBP_FORCE_ABORT("Unable to find setting: [" << name.c_str() << "]");
  }
}

void settings_t::PrintUsage() {
  std::cout << "Usage:\n\n";
  for (size_t i = 0; i < insertOrder.size(); ++i) {
    const string &s = insertOrder[i];
    const setting_t& val = settings[s];
    std::cout << val.PrintUsage() << std::endl;
  }

  std::cout << "Name:     [HELP]" << std::endl;
  std::cout << "CL keys:  [-h, --help]" << std::endl;
  std::cout << "Description: Print this help message" << std::endl;
}


} //namespace libp
