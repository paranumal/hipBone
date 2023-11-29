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

ifndef HIPBONE_MAKETOP_LOADED
ifeq (,$(wildcard ../make.top))
$(error cannot locate ${PWD}/../make.top)
else
include ../make.top
endif
endif

#libraries
LIBCORE_DIR=${HIPBONE_LIBS_DIR}/core
LIBMESH_DIR=${HIPBONE_LIBS_DIR}/mesh
LIBOGS_DIR=${HIPBONE_LIBS_DIR}/ogs
LIBPRIM_DIR=${HIPBONE_LIBS_DIR}/primitives

#includes
INCLUDES=${HIPBONE_INCLUDES}

#defines
LIBCORE_DEFINES=${HIPBONE_DEFINES}
LIBMESH_DEFINES=${HIPBONE_DEFINES}
LIBOGS_DEFINES=${HIPBONE_DEFINES}
LIBPRIM_DEFINES=${HIPBONE_DEFINES}

ifeq (true,${gpu-aware-mpi})
  LIBOGS_DEFINES+= -DGPU_AWARE_MPI
endif

#.cpp compilation flags
LIB_CXXFLAGS=${HIPBONE_CXXFLAGS} ${DEFINES} ${INCLUDES}

#object dependancies
LIB_DEPS=$(wildcard $(HIPBONE_INCLUDE_DIR)/*.h) \
         $(wildcard $(HIPBONE_INCLUDE_DIR)/*.hpp)
LIBOGS_DEPS=${LIB_DEPS} \
            $(wildcard $(HIPBONE_INCLUDE_DIR)/ogs/*.hpp)

LIBCORE_SRC =$(wildcard core/*.cpp)
LIBMESH_SRC =$(wildcard mesh/*.cpp)
LIBOGS_SRC =$(wildcard ogs/*.cpp)
LIBPRIM_SRC =$(wildcard primitives/*.cpp)

LIBCORE_OBJS=$(LIBCORE_SRC:.cpp=.o)
LIBMESH_OBJS=$(LIBMESH_SRC:.cpp=.o)
LIBOGS_OBJS=$(LIBOGS_SRC:.cpp=.o)
LIBPRIM_OBJS=$(LIBPRIM_SRC:.cpp=.o)

#install prefix
PREFIX ?= ${HIPBONE_DIR}/install

.PHONY: all core prim mesh ogs \
        clean realclean silentUpdate-core  \
        silentUpdate-ogs silentUpdate-mesh \
        silentUpdate-prim

all: core prim mesh ogs

core: libcore.a silentUpdate-core
prim: libprim.a silentUpdate-prim
mesh: libmesh.a silentUpdate-mesh
ogs: libogs.a silentUpdate-ogs

libcore.a: $(LIBCORE_OBJS)
ifneq (,${verbose})
	ar -cr libcore.a $(LIBCORE_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libcore.a $(LIBCORE_OBJS)
endif

libprim.a: $(LIBPRIM_OBJS) | libcore.a
ifneq (,${verbose})
	ar -cr libprim.a $(LIBPRIM_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libprim.a $(LIBPRIM_OBJS)
endif

libogs.a: $(LIBOGS_OBJS) | libprim.a
ifneq (,${verbose})
	ar -cr libogs.a $(LIBOGS_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libogs.a $(LIBOGS_OBJS)
endif

libmesh.a: $(LIBMESH_OBJS) | libogs.a
ifneq (,${verbose})
	ar -cr libmesh.a $(LIBMESH_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libmesh.a $(LIBMESH_OBJS)
endif

silentUpdate-core:
	@true

silentUpdate-prim:
	@true

silentUpdate-ogs:
	@true

silentUpdate-mesh:
	@true

# rule for .cpp files
%.o: %.cpp $(DEPS)
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $*.o -c $*.cpp $(CORE_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $*.o -c $*.cpp $(CORE_CXXFLAGS)
endif

core/%.o: core/%.cpp $(LIB_DEPS)
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $@ -c $< ${LIBCORE_DEFINES} $(LIB_CXXFLAGS)
endif

mesh/%.o: mesh/%.cpp $(LIB_DEPS) | libogs.a
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $@ -c $< ${LIBMESH_DEFINES} $(LIB_CXXFLAGS)
endif

ogs/%.o: ogs/%.cpp $(LIBOGS_DEPS) | libprim.a
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS)
endif

primitives/%.o: primitives/%.cpp $(LIBOGS_DEPS) | libcore.a
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $@ -c $< ${LIBPRIM_DEFINES} $(LIB_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $@ -c $< ${LIBPRIM_DEFINES} $(LIB_CXXFLAGS)
endif

install:
	@install -d ${PREFIX}/libs -v
	@install -d ${PREFIX}/libs/core/okl -v
	@install -d ${PREFIX}/libs/ogs/okl -v
	@install -m 750 core/okl/*.okl -t ${PREFIX}/libs/core/okl/ -v
	@install -m 750 ogs/okl/*.okl -t ${PREFIX}/libs/ogs/okl/ -v

#cleanup
clean:
	rm -f core/*.o primitives/*.o mesh/*.o ogs/*.o *.a
