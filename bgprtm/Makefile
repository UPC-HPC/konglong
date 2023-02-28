PROJ = swpro
LIBA = libWaveProp.a
#LIBSO = libXXX.so
#EXE = XXX
VERSION = version.cpp
#SDIR = .
#ODIR = 
#DISTBIN = 
#DISTLIB = 

GENFLAGS = -fopenmp -fPIC
CXXFLAGS = $(CXXSTD)
INCLUDEDIRS = ../libfftavx ../.. ../model_builder ../libgpurtm
LDFLAGS = -Wl,-no-undefined -fopenmp
#INCLUDEDIRS = ../libFFTV ../io ../libfftavx ../libDeghost 
#LDFLAGS = -Wl,-no-undefined -fopenmp -lCommon
LIBS = -lswio -l:libjseisIO.a -l:libyaml-cpp.a  -lm -lCommon


include ../make.mk
CC  = mpic++
CXX = mpic++


