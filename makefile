#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2017-2022 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
#
#####################################################################################

define HIPBONE_HELP_MSG

hipBone makefile targets:

   make hipBone (default)
   make clean
   make clean-libs
   make clean-kernels
   make realclean
   make install
   make info
   make help

Usage:

make hipBone
   Build hipBone executable.
make clean
   Clean the hipBone executable and object files.
make clean-libs
   In addition to "make clean", also clean the mesh, ogs, and core libraries.
make clean-kernels
   In addition to "make clean-libs", also cleans the cached OCCA kernels.
make realclean
   In addition to "make clean-kernels", also clean 3rd party libraries.
make install PREFIX=<path>
   Install hipBone and okl files to PREFIX location (default hipBone/install).
make info
   List directories and compiler flags in use.
make help
   Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter hipBone clean clean-libs clean-kernels \
                realclean install info help,$(MAKECMDGOALS)))
ifneq (,$(MAKECMDGOALS))
$(error ${HIPBONE_HELP_MSG})
endif
endif

ifndef HIPBONE_MAKETOP_LOADED
ifeq (,$(wildcard ./make.top))
$(error cannot locate ${PWD}/make.top)
else
include ./make.top
endif
endif

#includes
INCLUDES=${HIPBONE_INCLUDES} \
		 -I${HIPBONE_LIBS_DIR}/include \
		 -I.


#defines
DEFINES =${HIPBONE_DEFINES}

#.cpp compilation flags
HB_CXXFLAGS=${HIPBONE_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${HIPBONE_LIBS_DIR} -lmesh -logs -lprim -lcore \
     ${HIPBONE_LIBS}

#link flags
LFLAGS=${HB_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(HIPBONE_INCLUDE_DIR)/*.h) \
     $(wildcard $(HIPBONE_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

#install prefix
PREFIX ?= ${HIPBONE_DIR}/install

.PHONY: all clean clean-libs clean-kernels realclean \
	     install help info hipBone

all: hipBone

hipBone:$(OBJS) hipBone.o hipbone_libs
ifneq (,${verbose})
	$(HIPBONE_LD) -o hipBone hipBone.o $(OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_LD) -o hipBone hipBone.o $(OBJS) $(LFLAGS)
endif

hipbone_libs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C ${HIPBONE_LIBS_DIR} mesh ogs core prim verbose=${verbose}
else
	@${MAKE} -C ${HIPBONE_LIBS_DIR} mesh ogs core prim --no-print-directory
endif

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

# rule for .cpp files
%.o: %.cpp $(DEPS) | hipbone_libs
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $*.o -c $*.cpp $(HB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $*.o -c $*.cpp $(HB_CXXFLAGS)
endif

install:
	@${MAKE} -C ${HIPBONE_LIBS_DIR} install PREFIX=${PREFIX} --no-print-directory
	@install -d ${PREFIX}/okl -v
	@install -m 750 okl/*.okl -t ${PREFIX}/okl/ -v
	@install -m 750 hipBone ${PREFIX}/hipBone -v

#cleanup
clean:
	rm -f src/*.o *.o hipBone

clean-libs: clean
	${MAKE} -C ${HIPBONE_LIBS_DIR} clean

clean-kernels: clean-libs
	rm -rf ${HIPBONE_DIR}/.occa/

realclean: clean-kernels
	${MAKE} -C ${OCCA_DIR} clean

help:
	$(info $(value HIPBONE_HELP_MSG))
	@true

info:
	$(info OCCA_DIR  = $(OCCA_DIR))
	$(info HIPBONE_DIR  = $(HIPBONE_DIR))
	$(info HIPBONE_ARCH = $(HIPBONE_ARCH))
	$(info HIPBONE_CXXFLAGS  = $(HB_CXXFLAGS))
	$(info LIBS      = $(LIBS))
	@true
