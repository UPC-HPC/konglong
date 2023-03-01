# use DEBUG=1 or DEBUG=0 to enable/disable debug builds, default is DEBUG=1.
SDIR ?= .
ODIR ?= .
SWBUILD ?= $(HOME)/seiswave
#DISTBIN = $(SWBUILD)/bin/$(PROJ)
DISTBIN = $(SWBUILD)/bin
DISTINC = $(SWBUILD)/include/$(PROJ)
DISTLIB = $(SWBUILD)/lib
BINDIR = $(ODIR)/bin
OBJDIR = $(ODIR)/obj
ifneq ($(LIBA),)
    OUTPUT      += $(BINDIR)/$(LIBA)
    DISTOUT     += $(DISTLIB)/$(LIBA)
endif
ifneq ($(LIBSO),)
    OUTPUT      += $(BINDIR)/$(LIBSO)
    DISTOUT     += $(DISTLIB)/$(LIBSO)
endif
ifneq ($(EXE),)
    OUTPUT      += $(BINDIR)/$(EXE)
    DISTOUT     += $(DISTBIN)/$(EXE)
endif

ifneq (,$(wildcard /usr/bin/lsb_release))
  OS := $(shell lsb_release -i | rev | cut -f 1 | rev)
  OS := $(OS)$(shell lsb_release -a | grep -oP '(?<=^Release:\s)\d+')
else
  OS := $(shell grep -oP '(?<=^ID=")[^"]+' /etc/os-release)$(shell grep -oP '(?<=^VERSION_ID=")\d+' /etc/os-release)
endif
$(info ***** OS:          $(OS) )
$(info ***** PROJ:        $(PROJ) )
$(info ***** CXX:         $(CXX) )
$(info ***** DISTINC:     $(DISTINC) )
$(info ***** DISTOUT:     $(DISTOUT) )
$(info ***** OBJDIR:      $(OBJDIR) )
$(info ***** BINDIR:      $(BINDIR) )

DEBUG      ?= 0
debug      ?= $(DEBUG)
#AddressSanitizer
asan       ?= 0
vectorize  ?= 0
avx        ?= 0
sse        ?= 0
native     ?= 1
ifneq ($(avx)$(sse),00)
  native = 0
endif

GCC_VERSION = $(shell gcc -dumpfullversion -dumpversion | sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' -e 's/^[0-9]\{3,4\}$$/&00/')
WITH_CPP14 := $(shell expr $(GCC_VERSION) '>=' 70000)  # 7.0.0
ifeq ($(strip $(WITH_CPP14)),1)
    CXXSTD := -std=c++14
else
    CXXSTD := -std=c++11
endif
#$(info ***** WITH_CPP14:  '$(WITH_CPP14)' )
$(info ***** CXXSTD:      $(CXXSTD) )

ifneq ($(strip $(debug)),0)
    GENFLAGS   += -g -Wall -Wno-unused -O0 -DDEBUG -fno-omit-frame-pointer -fstack-protector-all
else
    GENFLAGS   += -g -Wall -Wno-unused -O3 -ffast-math -DNDEBUG -fstack-protector-all

    ifeq ($(native),1)
      GENFLAGS   += -march=native
    else
        ifeq ($(sse),1)
            GENFLAGS   += -msse4.2
        endif
        ifeq ($(avx),1)
            GENFLAGS   += -mavx
        else ifeq ($(avx),2)
            GENFLAGS   += -mavx2
        endif
    endif
endif

ifeq ($(findstring ic,$(CC)),)  # not icc icx icpc icpx
  GENFLAGS   +=   -mstackrealign
  ifneq ($(strip $(debug)),0)
      GENFLAGS   += -lmcheck
    endif
    ifeq ($(vectorize),0)
        GENFLAGS   += -ftree-vectorize -ftree-vectorizer-verbose=0
    else ifneq ($(vectorize),)
        GENFLAGS   += -ftree-vectorize -fopt-info-vec-missed -ftree-vectorizer-verbose=3
    endif
else
    ifeq ($(vectorize),1) # icc: -qopt-report5
        GENFLAGS   += -qopt-report
    endif
#  GENFLAGS   += -Wno-unused-variable -Wno-unused-private-field -Wno-unused-function -Wno-unused-but-set-variable
endif

ifneq ($(asan),0)
    # export ASAN_OPTIONS=detect_leaks=0  # if you do not want to see mem leak info
    GENFLAGS += -fsanitize=address -fno-omit-frame-pointer -fstack-protector-all -D_GLIBCXX_DEBUG
    LDFLAGS    += -fsanitize=address -fstack-protector-all -D_GLIBCXX_DEBUG
endif

ifeq ($(MKLROOT),)
    GENFLAGS   += -DNO_MKL
endif

all: $(VERSION) $(DEPS_F) $(DEPS_C) $(OUTPUT) install
    @echo OUTPUT = $(OUTPUT)
    @echo INSTALL = $(DISTOUT)

$(VERSION):
    @echo "***** Generating $(SDIR)/$(VERSION) *****"
    @git rev-parse HEAD | awk ' BEGIN {print "#include \"version.h\""} {print "const char * GIT_SHA = \"" $$0"\";"} END {}' > $(SDIR)/$(VERSION)
    @git rev-parse --short HEAD | awk 'BEGIN {} {print "const char * GIT_SHA_SHORT = \""$$0"\";"} END {} ' >> $(SDIR)/$(VERSION)
    @git describe --tags --long --dirty --always | awk 'BEGIN {} {print "const char * GIT_TAG_VERSION = \""$$0"\";"} END {} ' >> $(SDIR)/$(VERSION)
    @git log -1 | awk '/Date/ {print "const char * GIT_DATE = \""$$0"\";"} END {} ' >> $(SDIR)/$(VERSION)
    @date | awk 'BEGIN {} {print "const char * VER_DATE = \""$$0"\";"} END {} ' >> $(SDIR)/$(VERSION)

INC_SRCS    = $(wildcard $(SDIR)/*.h $(SDIR)/*.hpp)
CPP_SRCS    = $(wildcard $(SDIR)/*.cpp)
C_SRCS      = $(wildcard $(SDIR)/*.c)
ifeq ($(VERSION), version.cpp)
    CPP_SRCS     += $(SDIR)/version.cpp
endif
ifeq ($(VERSION), version.c)
    C_SRCS     += $(SDIR)/version.c
endif
F_SRCS      = $(wildcard $(SDIR)/*.f)
OBJ_F       = $(addprefix $(OBJDIR)/,$(notdir $(F_SRCS:.f=.o)))
OBJ_C       = $(addprefix $(OBJDIR)/,$(notdir $(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o)))
OBJECTS     = $(addprefix $(OBJDIR)/,$(notdir $(CPP_SRCS:.cpp=.o) $(C_SRCS:.c=.o) $(F_SRCS:.f=.o)))
GENFLAGS   += $(patsubst %,-I%,$(INCLUDEDIRS)) -I$(FFTW_INC) -I$(OPENBLAS_INC) -I$(SWBUILD)/include
CFLAGS     += $(GENFLAGS)
CXXFLAGS   += $(GENFLAGS) -Wno-reorder
FFLAGS     += $(GENFLAGS) -Wno-tabs -Wno-unused-dummy-argument
#LDFLAGS    += -L/usr/local/lib -L$(DISTLIB) -Wl,--warn-unresolved-symbols
LDFLAGS    += -Wl,-rpath=$(SWBUILD)/lib  -L$(SWBUILD)/lib
ifneq ($(FFTW_LIB),)
  LDFLAGS    += -Wl,-rpath=$(FFTW_LIB) -L$(FFTW_LIB) -Wl,-rpath=$(OPENBLAS_LIB) -L$(OPENBLAS_LIB) -Wl,-rpath=$(SCALAPACK_LIB) -L$(SCALAPACK_LIB)
endif
ifeq ($(OS),RedHatEnterpriseServer7)
  LDFLAGS    += -Wl,-rpath=/s0/GEOEAST/ieco1.0/support/mvapich2.tcp/lib -L/s0/GEOEAST/ieco1.0/support/mvapich2.tcp/lib
endif

ifeq ($(findstring mpi,$(CC)),mpi)
  (echo "Set CC to mpicc or mpic++ after loading make.mk!"; exit 1)
endif

ifneq ($(findstring if,$(FC)),) #FC is explictly set to ifort or ifx
  LIBFORTRAN = -lifcoremt -limf
#  ifeq ($(findstring x,$(CC)),) # not icx icpx
#    FC       = ifort
#  else
#    FC       = ifx
#  endif
else
  ifeq ($(DEVTOOLSET),)
    LIBFORTRAN = -lgfortran
  else
    LIBFORTRAN = -Wl,-rpath=$(DEVTOOLSET)/libportable -L$(DEVTOOLSET)/libportable -lgfortran
  endif
  FC         = gfortran
endif

ifeq ($(BOOST_INC),)
  GENFLAGS   += -I/usr/include/boost169 -I/usr/include/lapacke -I/usr/include/openblas
else
  GENFLAGS   += -I$(BOOST_INC)
endif

MODULE_AVAIL := $(command -v module)
ifneq ($(MODULE_AVAIL),)
  OPENBLAS_LOADED := $(shell module avail openblas | grep '(L)')
endif
#$(info ***** OPENBLAS_LOADED:      '$(OPENBLAS_LOADED)' )
ifneq ($(MKLROOT),)
    GENFLAGS   += -m64 -I$(MKLROOT)/include -I$(MKLROOT)/include/fftw
    LDFLAGS    += -Wl,-rpath=$(MKLROOT)/lib/intel64,-rpath=$(MKLROOT)/../compiler/lib/intel64,-rpath=$(MKLROOT)/../../compiler/latest/linux/compiler/lib/intel64 \
              -L$(MKLROOT)/lib/intel64 -L$(MKLROOT)/../compiler/lib/intel64 -L$(MKLROOT)/../../compiler/latest/linux/compiler/lib/intel64\
                      -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lirc -lsvml -liomp5 -lm -ldl
else
  ifneq ($(OPENBLAS_LOADED),)
    LAPACK     = -lopenblas -lscalapack
  else
    LAPACK     = -lopenblas -llapack -llapacke
  endif
endif


ifneq ($(CWPROOT),)
# do a symbolic link to point $(CWPROOT)/include to "su/" directory under some CPATH dir, e.g., /usr/local/include/
# this way source code use #include "su/header.h" for clarity, however cwp.h need "Complex.h" below #-I$(CWPROOT)/src/Complex/include
    GENFLAGS += -DUSE_CWP -I$(CWPROOT)/include
    LIBCWP         = -Wl,-rpath=$(CWPROOT)/lib -L$(CWPROOT)/lib  -Wl,--no-as-needed -lsu -lpar -lcwp
endif

WITH_OIIO := $(shell test -d /usr/include/OpenImageIO && echo 1 || echo 0)
ifeq ($(WITH_OIIO),1)
    LDFLAGS    += -L/usr/lib/x86_64-linux-gnu -lOpenImageIO
else
    GENFLAGS   += -DNO_OIIO=1
endif

LOCAL_LIBDIR = ${HOME}/local/lib.$(OS)
levmar:
    if [ ! -f $(LOCAL_LIBDIR)/liblevmar.a ]; then \
        wget -c http://users.ics.forth.gr/~lourakis/levmar/levmar-2.6.tgz -O - | tar xz && \
        cd levmar-2.6 && perl -p -i -e 's|^#define LINSOLVERS_RETAIN_MEMORY|//#define LINSOLVERS_RETAIN_MEMORY|g' levmar.h && \
        make liblevmar.a && mkdir -p $(LOCAL_LIBDIR) && mv liblevmar.a $(LOCAL_LIBDIR) && \
        cd .. && rm -rf levmar-2.6; \
    fi ;
LIBLEVMAR = $(LOCAL_LIBDIR)/liblevmar.a $(LAPACK)

#DEPS        = $(patsubst %,%,$(OBJECTS:.o=.d))
DEPS_F        = $(patsubst %,%,$(OBJ_F:.o=.d))
DEPS_C        = $(patsubst %,%,$(OBJ_C:.o=.d))
MAKEDEPEND_F  = $(SILENT)$(CC) -MF"$(@:.o=.d)" -MG -MM -MP -MT"$(@:.d=.o)" "$<" -cpp $(CXXFLAGS) -Wno-tabs
MAKEDEPEND_C  = $(SILENT)$(CC) -MF"$(@:.o=.d)" -MG -MM -MP -MT"$(@:.d=.o)" "$<" -cpp $(CXXFLAGS)
MKODIR      = -@mkdir -p $(OBJDIR)
MKBDIR      = -@mkdir -p $(BINDIR)
MKDISTOUT   = -@mkdir -p $(dir $(DISTOUT))
MKDISTINC   = -@mkdir -p $(DISTINC)
CPDISTOUT   = $(subst ^, ,$(join $(addprefix cp^,$(OUTPUT)),$(patsubst %,^%;,$(DISTOUT))))

#SILENT = @

.NOTPARALLEL:  install
.PHONY: clean $(VERSION)

$(OBJDIR)/%.d: $(SDIR)/%.f
    $(MKODIR)
    $(MAKEDEPEND_F)

$(OBJDIR)/%.d: $(SDIR)/%.c*
    $(MKODIR)
    $(MAKEDEPEND_C)

$(OBJDIR)/%.o: $(SDIR)/%.c
    $(MKODIR)
    $(SILENT)$(CC) -c -o $@ $< $(CFLAGS)

$(OBJDIR)/%.o: $(SDIR)/%.cpp
    $(MKODIR)
    $(SILENT)$(CXX) -c -o $@ $< $(CXXFLAGS)

$(OBJDIR)/%.o: $(SDIR)/%.cu
    $(MKODIR)
    $(SILENT)$(NVCC) -c -o $@ $< $(NVCC_FLAGS)

$(OBJDIR)/%.o: $(SDIR)/%.f
    $(MKODIR)
    $(SILENT)${FC} -c -o $@ $< $(FFLAGS)

$(BINDIR)/%.a: $(filter-out $(OBJDIR)/main.o,$(OBJECTS)) $(ARCHIVES)
    $(MKBDIR)
    $(SILENT)$(AR) rcs $@ $^

$(BINDIR)/%.so: $(filter-out $(OBJDIR)/main.o,$(OBJECTS))
    $(MKBDIR)
    $(SILENT)$(CXX) -shared -fPIC -o $@ $^ $(LIBS) $(ARCHIVES)

$(BINDIR)/$(EXE): $(OBJECTS) $(ARCHIVES)
    $(MKBDIR)
    $(SILENT)$(CXX) -o $@ $^ -no-pie $(LDFLAGS) $(LIBS) $(LIBSEXE) $(ARCHIVES)

install:
    $(MKDISTOUT)
    $(MKDISTINC)
    $(CPDISTOUT)
ifneq ($(strip $(INC_SRCS)),)
    $(SILENT) \cp -u $(INC_SRCS) $(DISTINC)
endif

clean:
    $(SILENT)$(RM) $(OUTPUT)
    $(SILENT)$(RM) $(OBJDIR)/*.[od]

realclean:
    $(SILENT)$(RM) $(OUTPUT)
    $(SILENT)$(RM) -rf $(OBJDIR)
    $(SILENT)$(RM) $(DISTOUT)
    $(SILENT)$(RM) -rf  $(DISTINC)

#include $(DEPS)

