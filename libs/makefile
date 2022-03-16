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

#includes
INCLUDES=${HIPBONE_INCLUDES}
LIBOGS_INCLUDES= -I${HIPBONE_INCLUDES}/ogs

#defines
LIBCORE_DEFINES=${HIPBONE_DEFINES} -DHIPBONE_DIR='"${HIPBONE_DIR}"'
LIBMESH_DEFINES=${HIPBONE_DEFINES} -DMESH_DIR='"${LIBMESH_DIR}"'
LIBOGS_DEFINES=${HIPBONE_DEFINES} -DHIPBONE_DIR='"${HIPBONE_DIR}"'  -DOGS_DIR='"${LIBOGS_DIR}"'

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

LIBCORE_OBJS=$(LIBCORE_SRC:.cpp=.o)
LIBMESH_OBJS=$(LIBMESH_SRC:.cpp=.o)
LIBOGS_OBJS=$(LIBOGS_SRC:.cpp=.o)

.PHONY: all core mesh ogs \
        clean realclean silentUpdate-core  \
        silentUpdate-ogs silentUpdate-mesh

all: core mesh ogs

core: libcore.a silentUpdate-core
mesh: libmesh.a silentUpdate-mesh
ogs: libogs.a silentUpdate-ogs

libcore.a: $(LIBCORE_OBJS)
ifneq (,${verbose})
	ar -cr libcore.a $(LIBCORE_OBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr libcore.a $(LIBCORE_OBJS)
endif

libogs.a: $(LIBOGS_OBJS) | libcore.a
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

ogs/%.o: ogs/%.cpp $(LIBOGS_DEPS) | libcore.a
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS) $(LIBOGS_INCLUDES)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $@ -c $< ${LIBOGS_DEFINES} $(LIB_CXXFLAGS) $(LIBOGS_INCLUDES)
endif

#cleanup
clean:
	rm -f core/*.o mesh/*.o ogs/*.o *.a
