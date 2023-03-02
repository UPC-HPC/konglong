#include "Wavefield.h"
#include "string.h"
#include "stdlib.h"


Wavefield::Wavefield(int nx, int ny, int nz) : nx_(nx) , ny_(ny),nz_(nz)
{

  w0 = NULL;
  w1 = NULL;
  wb = NULL;
  wx = NULL;
  wy = NULL;
  wz = NULL;
  wr = NULL;
  isInitialized = false;

}

Wavefield::~Wavefield() {
  if(isInitialized) deallocateMemory();
}

/*
 * Allocate memory for TTI propagation (multithreaded)
 */
void Wavefield::allocateMemory() {


  if(!isInitialized) {
    size_t nxyz = (size_t)nx_ * ny_ * nz_;
    size_t nSize = nxyz*sizeof(float);

    w0 = (float *)malloc(nSize);
    w1 = (float *)malloc(nSize);
    wb = (float *)malloc(nSize);
    wx = (float *)malloc(nSize);
    wy = (float *)malloc(nSize);
    wz = (float *)malloc(nSize);
    wr = (float *)malloc(nSize);
  }

  isInitialized = true;
}

/*
 * Free all memory allocated by Wavefield instance
 */
void Wavefield::deallocateMemory() {

  if(isInitialized) {
    free(w0);
    free(w1);
    free(wb);
    free(wx);
    free(wy);
    free(wz);
    free(wr);
    }

  isInitialized = false;
}

/*
 * Zero all memory allocated by Wavefield instance (multithreaded)
 */
void Wavefield::cleanMemory() {
    if(!isInitialized) 
    {
        size_t nxyz = (size_t)nx_ * ny_ * nz_; 
        size_t nSize = nxyz*sizeof(float);
        memset(w0, 0, nSize);
        memset(w1, 0, nSize);
        memset(wb, 0, nSize);
        memset(wx, 0, nSize);
        memset(wy, 0, nSize);
        memset(wz, 0, nSize);
        memset(wr, 0, nSize);
    }
}

