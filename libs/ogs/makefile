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

ifndef HIPBONE_MAKETOP_LOADED
ifeq (,$(wildcard ../../make.top))
  $(error cannot locate ${PWD}/../../make.top)
else
  include ../../make.top
endif
endif

LIBOGS_DIR=${HIPBONE_CORE_DIR}/ogs

#libraries
GS_DIR=${HIPBONE_TPL_DIR}/gslib

#includes
INCLUDES=${HIPBONE_INCLUDES} \
         -I${HIPBONE_CORE_DIR}/include \
         -I${GS_DIR}/src \
         -I./include

#defines
DEFINES=${HIPBONE_DEFINES} \
        -D DOGS='"${CURDIR}"'

#.cpp compilation flags
OGS_CFLAGS=${HIPBONE_CFLAGS} ${DEFINES} ${INCLUDES}
OGS_CXXFLAGS=${HIPBONE_CXXFLAGS} ${DEFINES} ${INCLUDES}

#object dependancies
DEPS:=$(wildcard $(HIPBONE_INCLUDE_DIR)/*.h) \
     $(wildcard ${LIBOGS_DIR}/include/*.h) \
     $(wildcard ${LIBOGS_DIR}/include/*.hpp) \
     $(wildcard ${LIBOGS_DIR}/include/*.tpp) \
     ${HIPBONE_INCLUDE_DIR}/ogs.hpp

SRC=$(wildcard src/*.cpp)
OBJS=$(SRC:.cpp=.o)

CSRC=$(wildcard src/*.c)
COBJS=$(CSRC:.c=.o)

# types of files we are going to construct rules for
.SUFFIXES: .c .cpp

.PHONY: all lib clean realclean silentUpdate

all: lib

lib: ../libogs.a silentUpdate

../libogs.a:$(OBJS) ${COBJS}
ifneq (,${verbose})
	ar -cr ../libogs.a $(OBJS) $(COBJS)
else
	@printf "%b" "$(LIB_COLOR)Building library $(@F)$(NO_COLOR)\n";
	@ar -cr ../libogs.a $(OBJS) $(COBJS)
endif

silentUpdate:
	@true

# rule for .c files
%.o: %.c $(DEPS)
ifneq (,${verbose})
	$(HIPBONE_CC) -o $*.o -c $*.c $(OGS_CFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CC) -o $*.o -c $*.c $(OGS_CFLAGS)
endif

# rule for .cpp files
%.o: %.cpp $(DEPS)
ifneq (,${verbose})
	$(HIPBONE_CXX) -o $*.o -c $*.cpp $(OGS_CXXFLAGS)
else
	@printf "%b" "$(OBJ_COLOR)Compiling $(@F)$(NO_COLOR)\n";
	@$(HIPBONE_CXX) -o $*.o -c $*.cpp $(OGS_CXXFLAGS)
endif

#cleanup
clean:
	rm -f src/*.o ../libogs.a

realclean: clean
	${MAKE} -C $(GS_DIR) clean
