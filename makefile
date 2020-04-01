#####################################################################################
#
#The MIT License (MIT)
#
#Copyright (c) 2020 Tim Warburton, Noel Chalmers, Jesse Chan, Ali Karakus
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
make info
   List directories and compiler flags in use.
make help
   Display this help message.

Can use "make verbose=true" for verbose output.

endef

ifeq (,$(filter hipBone clean clean-libs clean-kernels \
                realclean info help,$(MAKECMDGOALS)))
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

#libraries
GS_DIR       =${HIPBONE_TPL_DIR}/gslib
CORE_DIR     =${HIPBONE_DIR}/core
OGS_DIR      =${CORE_DIR}/ogs
MESH_DIR     =${CORE_DIR}/mesh

#includes
INCLUDES=${HIPBONE_INCLUDES} \
		 -I${HIPBONE_LIBS_DIR}/include \
		 -I.


#defines
DEFINES =${HIPBONE_DEFINES} \
         -DHIPBONE_DIR='"${HIPBONE_DIR}"'

#.cpp compilation flags
HB_CXXFLAGS=${HIPBONE_CXXFLAGS} ${DEFINES} ${INCLUDES}

#link libraries
LIBS=-L${CORE_DIR} -lmesh -logs -lcore \
     -L$(GS_DIR)/lib -lgs \
     ${HIPBONE_LIBS}

#link flags
LFLAGS=${HB_CXXFLAGS} ${LIBS}

#object dependancies
DEPS=$(wildcard *.hpp) \
     $(wildcard $(HIPBONE_INCLUDE_DIR)/*.h) \
     $(wildcard $(HIPBONE_INCLUDE_DIR)/*.hpp)

SRC =$(wildcard src/*.cpp)

OBJS=$(SRC:.cpp=.o)

.PHONY: all libcore libmesh libogs clean clean-libs \
		clean-kernels realclean help info hipBone

all: hipBone

hipBone:$(OBJS) hipBone.o | libmesh
ifneq (,${verbose})
	$(HIPBONE_LD) -o hipBone hipBone.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
else
	@printf "%b" "$(EXE_COLOR)Linking $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_LD) -o hipBone hipBone.o $(OBJS) $(MESH_OBJS) $(LFLAGS)
endif

${OCCA_DIR}/lib/libocca.so:
	${MAKE} -C ${OCCA_DIR}

libmesh: libogs libgs libcore
ifneq (,${verbose})
	${MAKE} -C ${MESH_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${MESH_DIR} lib --no-print-directory
endif

libogs: libcore
ifneq (,${verbose})
	${MAKE} -C ${OGS_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${OGS_DIR} lib --no-print-directory
endif

libcore: libgs
ifneq (,${verbose})
	${MAKE} -C ${CORE_DIR} lib verbose=${verbose}
else
	@${MAKE} -C ${CORE_DIR} lib --no-print-directory
endif

libgs: ${OCCA_DIR}/lib/libocca.so
ifneq (,${verbose})
	${MAKE} -C $(GS_DIR) install verbose=${verbose}
else
	@${MAKE} -C $(GS_DIR) install --no-print-directory
endif

# rule for .cpp files
%.o: %.cpp $(DEPS) | libmesh
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $*.o -c $*.cpp $(HB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $*.o -c $*.cpp $(HB_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o *.o hipBone

clean-libs: clean
	${MAKE} -C ${OGS_DIR} clean
	${MAKE} -C ${MESH_DIR} clean
	${MAKE} -C ${CORE_DIR} clean

clean-kernels: clean-libs
	rm -rf ${HIPBONE_DIR}/.occa/

realclean: clean-libs
	${MAKE} -C ${GS_DIR} clean
	${MAKE} -C ${OCCA_DIR} clean
	rm -rf ${HIPBONE_DIR}/.occa/

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
