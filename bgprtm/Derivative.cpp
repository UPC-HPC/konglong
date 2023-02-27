#include "Derivative.h"
#include "Model.h"

Derivative::Derivative(Grid *grid, Boundary *inBnd, int modelType, int inNThreads) {
  bnd = inBnd;
  nThreads = inNThreads;
  iswitch = 2;
  if(modelType == TTI) iswitch = 1;

  gridType = grid->mytype;

  nx = grid->nx;
  ny = grid->ny;
  nz = grid->nz;
  dx = grid->dx;
  dy = grid->dy;
  dz = grid->dz;

  nxz = (size_t)nx * (size_t)nz;
  nyz = (size_t)ny * (size_t)nz;

  slopex = grid->slopex;
  slopey = grid->slopey;
  jacobz = &(grid->jacobz)[0];
  jacobx = &(grid->jacobx)[0];
  jacoby = &(grid->jacoby)[0];
}

oid Derivative::setupGrid(int gridType, float slopex, float slopey, float *jacobx, float *jacoby, float *jacobz) {
  this->gridType = gridType;
  this->slopex = slopex;
  this->slopey = slopey;
  this->jacobx = jacobx;
  this->jacoby = jacoby;
  this->jacobz = jacobz;
}

void Derivative::getDiverge(Wavefield *myWavefield) {
  if((gridType == IRREGULAR) || (gridType == RECTANGLE)) getDiverge0(myWavefield);
  if((gridType == XYPYRAMID) || (gridType == XPYRAMID) || (gridType == YPYRAMID)) getDivergePXY(myWavefield);
}

void Derivative::getGradient(Wavefield *myWavefield) {
  if(nx > 1) {
    this->dx1(myWavefield->w1, myWavefield->wx, 1);
    if(bnd) bnd->applyX(myWavefield->wx, 1);
  }

  if(ny > 1) {
    this->dy1(myWavefield->w1, myWavefield->wy, 1);
    if(bnd) bnd->applyY(myWavefield->wy, 1);
  }

  this->dz1(myWavefield->w1, myWavefield->wz, 1);
  if(bnd) bnd->applyZ(myWavefield->wz, 1);

  if(gridType != RECTANGLE) {
    destretchz(myWavefield->wz);

    if((gridType == YPYRAMID) || (gridType == XYPYRAMID)) {
      dePyramidy(myWavefield->wy, myWavefield->wz);
    }

    if((gridType == XPYRAMID) || (gridType == XYPYRAMID)) {
      dePyramidx(myWavefield->wx, myWavefield->wz);
    }
  }
}
void Derivative::getLaplacian(Wavefield *myWavefield) {
  float *work = myWavefield->w1;
  float *out = myWavefield->wb;
  float *buff = (nx == 1 && ny == 1) ? out : myWavefield->wr; // 1D no need addition

  if(nx > 1) {
    this->dx2(work, out);
    if(bnd) bnd->applyX(out, 1);
  }

  if(ny > 1) {
    this->dy2(work, buff);
    if(bnd) bnd->applyY(buff, 1);
    addVolume(out, buff);
  }

  this->dz2(work, buff);
  if(bnd) bnd->applyZ(buff, 1);

  if(gridType != RECTANGLE) destretchz2(buff);

  if(nx > 1 || ny > 1) addVolume(out, buff);
}
oid Derivative::getDiverge0(Wavefield *myWavefield) { // work for regular grid or IRREGULAR grid
  float *buff = myWavefield->wr;
  float *out = myWavefield->wb;

  this->dz1(myWavefield->wz, out, iswitch);
  if(bnd) bnd->applyZ(out, 2);

  if(gridType == IRREGULAR) destretchz(out);

  if(nx > 1) {
    this->dx1(myWavefield->wx, buff, iswitch);
    if(bnd) bnd->applyX(buff, 2);
    addVolume(out, buff);
  }

  if(ny > 1) {
    this->dy1(myWavefield->wy, buff, iswitch);
    if(bnd) bnd->applyY(buff, 2);
    addVolume(out, buff);
  }
}
void Derivative::getDivergePXY(Wavefield *myWavefield) { // work for regular grid or IRREGULAR grid
  if((gridType != XYPYRAMID) && (gridType != YPYRAMID)) {
    printf("ERROR: calling error in getDiverge Pyramid\n");
    exit(-1);
  }

  float *buff = myWavefield->wr;
  float *out = myWavefield->wb;

  this->dz1(myWavefield->wz, out, iswitch);

  destretchz(out);

  if((gridType == XYPYRAMID) || (gridType == YPYRAMID)) {
    this->dz1(myWavefield->wz, buff, 2);
    dePyramidy(buff, out);
  }

  if((gridType == XYPYRAMID) || (gridType == XPYRAMID)) {
    this->dz1(myWavefield->wz, buff, iswitch);
    dePyramidx(buff, out);
  }

  if(nx > 1) {
    this->dx1(myWavefield->wx, buff, iswitch);
    if((gridType == XYPYRAMID) || (gridType == XPYRAMID)) rescalex(buff);
    addVolume(out, buff);
  }

  if(ny > 1) {
    this->dy1(myWavefield->wy, buff, iswitch);
    if((gridType == XYPYRAMID) || (gridType == YPYRAMID)) rescaley(buff);
    addVolume(out, buff);
  }}

void Derivative::addVolume(float *a, float *b) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    float *__restrict tmpbufa = a + iy * nxz;
    float *__restrict tmpbufb = b + iy * nxz;
    for(size_t ixz = 0; ixz < nxz; ixz++)
      tmpbufa[ixz] += tmpbufb[ixz];
  }
}

void Derivative::dePyramidy(float *outy, float *outz) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    __m128 scaly = _mm_mul_ps(_mm_set1_ps(-slopey), _mm_sub_ps(_mm_set1_ps(float(iy)), _mm_set1_ps(float(ny * 0.5f))));
    for(int ix = 0; ix < nx; ix++) {
      float *voli = outy + (iy * nx + ix) * nz;
      float *vola = outz + (iy * nx + ix) * nz;
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddi = _mm_load_ps(voli + iz);
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacoby + iz);
        ddi = _mm_mul_ps(jji, ddi);
        _mm_store_ps(vola + iz, _mm_add_ps(ddo, _mm_mul_ps(jji, _mm_mul_ps(scaly, ddi))));
        _mm_store_ps(voli + iz, ddi);  // change input doe rescalex
      }
    }
  }
}
void Derivative::dePyramidx(float *outx, float *outz) {
#pragma omp parallel for num_threads(nThreads) schedule(static) collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      __m128 scalx = _mm_mul_ps(_mm_set1_ps(-slopex), _mm_sub_ps(_mm_set1_ps(float(ix)), _mm_set1_ps(float(nx * 0.5f))));
      //float scalx = -slopex * (ix - nx*0.5);
      float *voli = outx + (iy * nx + ix) * nz;
      float *vola = outz + (iy * nx + ix) * nz;
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddi = _mm_load_ps(voli + iz);
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacobx + iz);
        ddi = _mm_mul_ps(jji, ddi);
        _mm_store_ps(vola + iz, _mm_add_ps(ddo, _mm_mul_ps(jji, _mm_mul_ps(scalx, ddi))));
        _mm_store_ps(voli + iz, ddi);  // change input doe rescalex
      }
    }
  }
}
void Derivative::rescaley(float *volapply) {
#pragma omp parallel for num_threads(nThreads) schedule(static) collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *__restrict vola = volapply + (iy * nx + ix) * nz;
#ifdef FORCE_SSE
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacoby + iz);
        _mm_store_ps(vola + iz, _mm_mul_ps(ddo, jji));
      }
#else
      for(int iz = 0; iz < nz; iz++)
        vola[iz] *= jacoby[iz];
#endif

    }
  }
}
void Derivative::rescalex(float *volapply) {
#pragma omp parallel for num_threads(nThreads) schedule(static) collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *__restrict vola = volapply + (iy * nx + ix) * nz;
#ifdef FORCE_SSE
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacobx + iz);
        _mm_store_ps(vola + iz, _mm_mul_ps(ddo, jji));
      }
#else
      for(int iz = 0; iz < nz; iz++)
        vola[iz] *= jacobx[iz];
#endif
    }
  }
}
void Derivative::destretchz2(float *volapply) {
#pragma omp parallel for num_threads(nThreads) schedule(static) collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *__restrict vola = volapply + (iy * nx + ix) * nz;
#ifdef FORCE_SSE
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacobz + iz);
        _mm_store_ps(vola + iz, _mm_mul_ps(ddo, _mm_mul_ps(jji, jji)));
      }
#else
      for(int iz = 0; iz < nz; iz++)
        vola[iz] *= jacobz[iz] * jacobz[iz];
#endif
    }
  }
}
void Derivative::destretchz(float *volapply) {
#pragma omp parallel for num_threads(nThreads) schedule(static) collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *__restrict vola = volapply + (iy * nx + ix) * nz;
#ifdef FORCE_SSE
      for(int iz = 0; iz < nz; iz += SSEsize) {
        __m128 ddo = _mm_load_ps(vola + iz);
        __m128 jji = _mm_load_ps(jacobz + iz);
        _mm_store_ps(vola + iz, _mm_mul_ps(ddo, jji));
      }
#else
      for(int iz = 0; iz < nz; iz++)
        vola[iz] *= jacobz[iz];
#endif
    }
  }
}

             
