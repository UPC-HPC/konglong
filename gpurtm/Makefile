###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS= -I./include
CC_LIBS=

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=$(CUDA_ROOT_DIR)/bin/nvcc
NVCC_FLAGS= -O3  --use_fast_math  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35
NVCCLINKS= -O3  --use_fast_math  -gencode arch=compute_35,code=sm_35 -gencode arch=compute_35,code=compute_35
NVCC_LIBS= -L$(CUDA_ROOT_DIR)/lib64 -lcudart -lcufft

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -lcufft

##########################################################
##########################################################

## Make variables ##

# Target executable name:
EXE = run_test

# Object files:
CUOBJS = cuda_kernel.o
CPPOBJS = main.o

##########################################################

## Compile ##

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(CPPOBJS) $(CUOBJS)
	$(CC) $(CC_FLAGS) $(CPPOBJS) $(CUOBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
# Compile main .cpp file to object files:
%.o : %.cpp
	$(CC) $(CC_FLAGS) -c $<  


# Compile CUDA source files to object files:
%.o : %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)


# Clean objects in object directory.
clean:
	$(RM) *.o $(EXE)

