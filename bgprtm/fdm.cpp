#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <math.h>
//#include <iostream>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include "GetPar.h"
#include "libCommon/io_util.h"
using libCommon::read_all;
using libCommon::write_all;
#include "fdm.hpp"
#include "volumefilter.h"

using namespace std;
Fdm::Fdm() {
  fd = NULL;
  data = NULL;
  swap = false;
  local = true;
  memset(&fdmh, 0, sizeof(FdmHeader));
  sethead(0.0f, 0.0f, 0.0f, 1, 1, 1, 1.0, 1.0, 1.0, 1, 1, 1, 1, 1, 1);
  gethostname(hostname, 255);
}

Fdm::Fdm(int nx0, int ny0, int nz0) {
  fd = NULL;
  data = NULL;
  swap = false;
  local = true;
  memset(&fdmh, 0, sizeof(FdmHeader));
  sethead(0.0f, 0.0f, 0.0f, nx0, ny0, nz0, 1.0, 1.0, 1.0, 1, 1, 1, 1, 1, 1);
  gethostname(hostname, 255);
}

Fdm::Fdm(int nx0, int ny0, int nz0, float dx0, float dy0, float dz0) {
  fd = NULL;
  data = NULL;
  swap = false;
  local = true;
  memset(&fdmh, 0, sizeof(FdmHeader));
  sethead(0.0f, 0.0f, 0.0f, nx0, ny0, nz0, dx0, dy0, dz0, 1, 1, 1, 1, 1, 1);
  gethostname(hostname, 255);
}

Fdm::~Fdm() {
  FileClose();
// data is a pointer which pass in from out side, sometime we have double free, we do not know
  if((data != NULL) && local) {
    free(data);
  };
  data = NULL;
}

void Fdm::reset() {
  if((data != NULL) && local) free(data);
  data = NULL;
  swap = false;
  local = true;
  memset(&fdmh, 0, sizeof(FdmHeader));
  filename.clear();
}

void Fdm::setFileName(const char *fname) {
  filename = fname;
}

int Fdm::readheader() {
  int head_size;
  float buf[128];

  head_size = 128;

  size_t myread = fread(buf, sizeof(float), head_size, fd);
  if(myread != size_t(head_size)) {
    printf("The head_size is %ld\n", myread);
    printf("ERROR (fdm_read on %s): Can't read the fdm file: %s\n", hostname, filename.c_str());
    fflush(stdout);
    return 1;
  }

 if(buf[2] < 1 || buf[2] > 100000) {
    swap = 1;
    swapEndian32(buf, 128);
  } else {
    swap = 0;
  }

  fdmh.x0 = buf[0];
  fdmh.y0 = buf[3];
  fdmh.z0 = buf[6];
  fdmh.nx = (int) buf[1];
  fdmh.ny = (int) buf[4];
  fdmh.nz = (int) buf[7];
  fdmh.dx = buf[2];
  fdmh.dy = buf[5];
  fdmh.dz = buf[8];

  fdmh.fx = (int) buf[14];
  fdmh.fy = (int) buf[15];
  fdmh.fz = 1;
  fdmh.xi = (int) buf[16];
  fdmh.yi = (int) buf[17];
  fdmh.zi = 1;

  fdmh.lx = fdmh.fx + (fdmh.nx - 1) * fdmh.xi;
  fdmh.ly = fdmh.fy + (fdmh.ny - 1) * fdmh.yi;
  fdmh.lz = fdmh.fz + (fdmh.nz - 1) * fdmh.zi;

  fdmh.vmin = buf[12];
  fdmh.vmax = buf[13];
  return 0;
}
int Fdm::do_mkdir(const char *path, mode_t mode) {
  struct stat st;
  int status = 0;
  if(stat(path, &st) != 0) {
    printf("Making dir %s ...\n", path);
    if(mkdir(path, mode) != 0) status = -1;
  } else if(!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    status = -1;
  }
  return (status);
}

int Fdm::mkpath(const char *path, mode_t mode) {
  char *pp;
  char *sp;
  int status;
  char *copypath = strdup(path);
  status = 0;
  pp = copypath;
  while(status == 0 && (sp = strchr(pp, '/')) != 0) {
    if(sp != pp) {
      *sp = '\0';
      status = do_mkdir(copypath, mode);
      *sp = '/';
    }
    pp = sp + 1;
  }
  if(status == 0) status = do_mkdir(path, mode);
  free(copypath);
  return (status);
}
int Fdm::mkpath4file(const char *file, mode_t mode) {
  char *pp;
  char *sp;
  int status;
  char *copypath = strdup(file);
  status = 0;
  pp = copypath;
  while(status == 0 && (sp = strchr(pp, '/')) != 0) {
    if(sp != pp) {
      *sp = '\0';
      status = do_mkdir(copypath, mode);
      *sp = '/';
    }
    pp = sp + 1;
  }
  free(copypath);
  return (status);
}

int Fdm::FileOpen() {
  FileClose();
  mkpath4file(filename.c_str(), 0777);
  fd = fopen(filename.c_str(), "wb");
  assertion(fd != nullptr, "ERROR (fdm_savecube on %s): can't open the file: %s!", hostname, filename.c_str());
  return 0;
}

int Fdm::FileOpenRead() {
  FileClose();
  if(!(fd = fopen(filename.c_str(), "rb"))) {
    printf("ERROR (%s) : can't open the file: %s!\n", hostname, filename.c_str());
    fflush(stdout);
    return 1;
  }
//printf("The file is opened %d\n", fd);
  return 0;
}

int Fdm::read(const char *path, float *vol) {
  int i;

  setFileName(path);
  FileOpenRead();
  memset(&fdmh, 0, sizeof(FdmHeader));
  readheader();
  fseek(fd, 512, SEEK_SET);
  size_t myseek = ftell(fd);
  if(myseek != 512) {
    printf("ERROR (Fdm::read on %s): Can't seek the file  %ld.\n", hostname, myseek);
    fflush(stdout);
    return 1;
  }

  size_t n = ((size_t) fdmh.nx) * fdmh.ny * (size_t) fdmh.nz;
  if(!vol) {
    data = (float *) calloc(n, sizeof(float));
    local = true;
  } else {
    local = false;
  }

  int chunk_size = 10 * 1024 * 1024; // 4*10MB
  size_t nread = 0;

  float *pd = vol ? vol : data;

  float vmin = 1.e20;
  float vmax = -1.e20;
  while(nread < n) {
    if(nread + chunk_size >= n) {
      chunk_size = n - nread;
    }
    if(fread(pd, sizeof(float), chunk_size, fd) != size_t(chunk_size)) {
      return 1;
    }
    if(swap) swapEndian32((float *) pd, chunk_size);

    for(i = 0; i < chunk_size; i++) {
      vmin = min(vmin, pd[i]);
      vmax = max(vmax, pd[i]);
    }

    nread += chunk_size;
    pd += chunk_size;
  }

  fdmh.vmin = vmin;
  fdmh.vmax = vmax;
  FileClose();
  return 0;
}
void Fdm::fillHeader(FdmHeader *header, float *buf) {
  header->x0 = buf[0];
  header->y0 = buf[3];
  header->z0 = buf[6];
  header->nx = (int) buf[1];
  header->ny = (int) buf[4];
  header->nz = (int) buf[7];
  header->dx = buf[2];
  header->dy = buf[5];
  header->dz = buf[8];

  header->fx = (int) buf[14];
  header->fy = (int) buf[15];
  header->fz = 1;
  header->xi = (int) buf[16];
  header->yi = (int) buf[17];
  header->zi = 1;

  header->lx = header->fx + (header->nx - 1) * header->xi;
  header->ly = header->fy + (header->ny - 1) * header->yi;
  header->lz = header->fz + (header->nz - 1) * header->zi;

  header->vmin = buf[12];
  header->vmax = buf[13];
}

void Fdm::fillHeader(FdmHeader &hdr) {
  memcpy(&fdmh, &hdr, sizeof(FdmHeader));
}
int Fdm::read(const char *fileName, float *data, float x0, float y0, int nx, int ny) {
  this->data = data;
  local = false;

  int myfd = open64(fileName, O_RDONLY);
  if(myfd < 0) {
    fprintf(stderr, "\n**** Unable to open file (%s) '%s' ! Cannot continue.\n", hostname, fileName);
    perror("Error");
    exit(-1);
  }

  memset(&fdmh, 0, sizeof(FdmHeader));

  int head_size = 128;
  float buf[head_size];

  ssize_t nbytes = read_all(myfd, buf, sizeof(float) * head_size);
  if(nbytes != static_cast<ssize_t>(sizeof(float) * head_size)) {
    printf("The head_size is %ld\n", nbytes);
    printf("ERROR (fdm_read on %s): Can't read the fdm file: %s\n", hostname, filename.c_str());
    fflush(stdout);
    return 1;
  }

  if(buf[2] < 1 || buf[2] > 100000) {
    swap = 1;
    swapEndian32(buf, 128);
  } else {
    swap = 0;
  }

  FdmHeader ghead;
  fillHeader(&ghead, buf);
    int ifx = roundf((x0 - ghead.x0) / ghead.dx) + 1;
  int ify = roundf((y0 - ghead.y0) / ghead.dy) + 1;

  // printf("ifx,ify,nx,ny: %d %d %d %d \n", ifx, ify, nx, ny);
  // printf("x0,y0,gx0,gy0: %f %f %f %f \n", x0, y0, ghead.x0, ghead.y0);
  //info();

  float *pd = data;

  float vmin = 1.e20;
  float vmax = -1.e20;
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {

      int jx = MAX(1, MIN(ghead.nx, ix + ifx));
      int jy = MAX(1, MIN(ghead.ny, iy + ify));
      size_t offset = ((size_t)(jy - 1) * (size_t) ghead.nx + (jx - 1)) * (size_t) ghead.nz + head_size;
      ssize_t nbytes = ::pread(myfd, pd, sizeof(float) * ghead.nz, sizeof(float) * offset);
      if(nbytes != static_cast<ssize_t>(sizeof(float) * ghead.nz)) printf("ERROR, cannot read model!\n");
      for(int iz = 0; iz < ghead.nz; iz++) {
        vmin = min(vmin, pd[iz]);
        vmax = max(vmax, pd[iz]);
      }
      pd += ghead.nz;
    }
  }

  fillHeader(&fdmh, buf);
  fdmh.x0 = x0;
  fdmh.y0 = y0;

  fdmh.nx = nx;
  fdmh.ny = ny;

  fdmh.fx = ifx;
  fdmh.fy = ify;

  fdmh.lx = fdmh.fx + (nx - 1) * fdmh.xi;
  fdmh.ly = fdmh.fy + (ny - 1) * fdmh.yi;

  fdmh.vmin = vmin;
  fdmh.vmax = vmax;

  close(myfd);
  return 0;
}

void Fdm::isResample(const char *fileName, int maxFreq, int &xratio, int &yratio, int &zratio, float vmin0) {

  int fd;

  fd = open64(fileName, O_RDONLY);
  if(fd < 0) {
    fprintf(stderr, "\n**** Unable to open file (%s) '%s' ! Cannot continue.\n", hostname, fileName);
    perror("Error");
    exit(-1);
  }

  int nx1 = fdmh.nx;
  int ny1 = fdmh.ny;
  size_t nxy1 = (size_t) nx1 * (size_t) ny1;
  float *work1 = new float[nxy1];
  read_all(fd, work1, sizeof(float) * nxy1);
  float vmin = 1.e20;
  for(int iy = 0; iy < ny1; iy++) {
    for(int ix = 0; ix < nx1; ix++) {
      size_t id1 = (size_t) iy * (size_t) nx1 + ix;
      vmin = MIN(vmin, work1[id1]);
    }
  }

  if(vmin < vmin0) vmin = vmin0;
  float dmin = vmin / (2.0f * maxFreq);
  printf("vmin=%f, dmin=%f \n", vmin, dmin);

  xratio = int(dmin / fdmh.dx + 0.5f);
  yratio = int(dmin / fdmh.dy + 0.5f);
  zratio = int(dmin / fdmh.dz + 0.5f);

  xratio = MAX(1, xratio);
  yratio = MAX(1, yratio);
  zratio = MAX(1, zratio);

  delete[] work1;
  close(fd);
}

void Fdm::readBinary(const char *fileName, int nx1, int ny1, int nz1, int ixOffset, int iyOffset, int xratio,
                     int yratio, int zratio) {
  int fd;

  fd = open64(fileName, O_RDONLY);
  if(fd < 0) {
    fprintf(stderr, "\n**** Unable to open file '%s' ! Cannot continue.\n", fileName);
    perror("Error");
    exit(-1);
  }

  int nx2 = fdmh.nx;
  int ny2 = fdmh.ny;
  int nz2 = fdmh.nz;

  size_t size2 = (size_t) nz2 * nx2 * ny2;

  ///printf("MEM: extra grid size on disk: [%d %d %d]. Total size: %ld bytes.\n", nx1, ny1, nz1,
  //    static_cast<size_t>(nx1) * ny1 * nz1 * sizeof(float));
  //printf("MEM: reading: [%d %d %d]. Total size %ld bytes.\n", nx2, ny2, nz2,
  //    static_cast<size_t>(nx2) * ny2 * nz2 * sizeof(float));

  data = (float *) malloc(sizeof(float) * size2);
  local = true;

  float *work1 = new float[nz1];
  for(int iy = 0; iy < ny2; iy++) {
    for(int ix = 0; ix < nx2; ix++) {
      size_t offset1 = (size_t)((iy * yratio + iyOffset) * nx1 * nz1) + (size_t)(ix * xratio + ixOffset) * nz1;
      size_t offset2 = (size_t)(iy * nx2 * nz2) + (size_t) ix * nz2;
      ssize_t nbytes = ::pread(fd, work1, sizeof(float) * nz1, sizeof(float) * offset1);
      if(nbytes != static_cast<ssize_t>(sizeof(float) * nz1)) {
        printf("ERROR, cannot read model!\n");
        abort();
      }
      if(zratio != 1) {
        for(int iz = 0; iz < nz2; iz++)
          data[offset2 + iz] = work1[iz * zratio];
      } else {
        memcpy(data + offset2, work1, nz2 * sizeof(float));
      }
    }
  }
  delete[] work1;
  close(fd);
}

void Fdm::readBinary(const char *fileName) {
  int fd;

  fd = open64(fileName, O_RDONLY);
  if(fd < 0) {
    fprintf(stderr, "\n**** Unable to open file '%s' ! Cannot continue.\n", fileName);
    perror("Error");
    exit(-1);
  }

  int nx2 = fdmh.nx;
  int ny2 = fdmh.ny;
  int nz2 = fdmh.nz;

  size_t nxy2 = (size_t) nx2 * (size_t) ny2;
  size_t size2 = (size_t) nz2 * nxy2;

  //printf("MEM: reading: [%d %d %d]. Total size %ld bytes.\n", nx2, ny2, nz2,
  //    static_cast<size_t>(nx2) * ny2 * nz2 * sizeof(float));

  data = (float *) malloc(sizeof(float) * size2);
  local = true;

  for(int iz = 0; iz < nz2; iz++) {
    read_all(fd, data + nxy2 * iz, sizeof(float) * nxy2);
  }

  close(fd);
}

void Fdm::readVelBin(const char *fileName, int nx1, int ny1, int ixOffset, int iyOffset) {
  int fd;

  fd = open64(fileName, O_RDONLY);
  if(fd < 0) {
    fprintf(stderr, "\n**** Unable to open file '%s' ! Cannot continue.\n", fileName);
    perror("Error");
    exit(-1);
  }

  int nx2 = fdmh.nx;
  int ny2 = fdmh.ny;
  int nz = fdmh.nz;

  size_t nxz1 = (size_t) nx1 * (size_t) nz;
  size_t nxz2 = (size_t) nx2 * (size_t) nz;
  size_t size2 = (size_t) ny2 * nxz2;

  data = (float *) malloc(sizeof(float) * size2);
  local = true;

  //printf("MEM: velocity grid size on disk: [%d %d %d]. Total size: %ld bytes.\n", nx1, ny1, nz,
  //    static_cast<size_t>(nx1) * ny1 * nz * sizeof(float));
  //printf("MEM: reading: [%d %d %d]. Total size %ld bytes.\n", nx2, ny2, nz,
  //    static_cast<size_t>(nx2) * ny2 * nz * sizeof(float));

  float *work1 = new float[nz];
  for(int iy = 0; iy < ny2; iy++) {
    for(int ix = 0; ix < nx2; ix++) {
      size_t offset1 = (size_t)(iy + iyOffset) * nxz1 + (size_t)((ix + ixOffset) * nz);
      size_t offset2 = (size_t)(iy * nxz2) + (size_t) ix * nz;
      ssize_t nbytes = ::pread(fd, work1, sizeof(float) * nz, sizeof(float) * offset1);
      if(nbytes != static_cast<ssize_t>(sizeof(float) * nz)) {
        printf("ERROR, cannot read model!\n");
        abort();
      }
      memcpy(data + offset2, work1, sizeof(float) * nz);
    }
  }
  delete[] work1;

  close(fd);
}

void Fdm::decimateRatio(float maxFreq, int &xratio, int &yratio, int &zratio, int nThreads) {

  int nx1 = fdmh.nx;
  int ny1 = fdmh.ny;
  int nz1 = fdmh.nz;

  float vmin = 1E15;
  //#pragma omp parallel for reduction(min:vmin) num_threads(nThreads) schedule(static) //needs omp 3.1
  for(int iy = 0; iy < ny1; iy++) {
    for(int ix = 0; ix < nx1; ix++) {
      for(int iz = 0; iz < nz1; iz++) {
        size_t id1 = ((size_t) iy * (size_t) nx1 + ix) * (size_t) nz1 + iz;
        vmin = MIN(vmin, data[id1]);
      }
    }
  }

  if(vmin > 1000000.0f || vmin < 100.0f) {
    printf("Error: something wrong with the velocity model, minimum velocity=%f \n", vmin);
    exit(-1);
  }

  if(vmin > 5000.0) {
    printf("warning: velocity is in feet/s , minimum velocity=%f \n", vmin);
    vmin /= 3.28084;
    printf("warning: converted to meter/s , minimum velocity=%f \n", vmin);
  }

  float dmin = vmin / (2.0f * maxFreq);
  printf("vmin=%f, dmin=%f \n", vmin, dmin);

  xratio = int(dmin / fdmh.dx);
  yratio = int(dmin / fdmh.dy);
  zratio = int(dmin / fdmh.dz);

  xratio = MAX(1, xratio);
  yratio = MAX(1, yratio);
  zratio = MAX(1, zratio);

  if(xratio > 1 || yratio > 1 || zratio > 1) {
    int nx2 = (nx1 - 1) / xratio + 1;
    int ny2 = (ny1 - 1) / yratio + 1;
    int nz2 = (nz1 - 1) / zratio + 1;

    //update the header
    sethead(fdmh.x0, fdmh.y0, fdmh.z0, nx2, ny2, nz2, fdmh.dx * xratio, fdmh.dy * yratio, fdmh.dz * zratio, fdmh.fx,
            fdmh.fy, fdmh.fz, fdmh.xi, fdmh.yi, fdmh.zi);

    size_t size2 = (size_t) nx2 * (size_t) ny2 * (size_t) nz2;
    float *data2 = (float *) malloc(size2 * sizeof(float));

    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny1; iy++) {
      for(int ix = 0; ix < nx1; ix++) {
        for(int iz = 0; iz < nz1; iz++) {
          size_t id1 = ((size_t) iy * (size_t) nx1 + ix) * (size_t) nz1 + iz;
          data[id1] = 1.0f / data[id1];
        }
      }
    }

    int sizex = MAX(1, int(0.5 * xratio));
    int sizey = MAX(1, int(0.5 * yratio));
    int sizez = MAX(1, int(0.5 * zratio));

    avgVolume3D(data, nx1, ny1, nz1, sizex, sizey, sizez);

    #pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny2; iy++) {
      for(int ix = 0; ix < nx2; ix++) {
        for(int iz = 0; iz < nz2; iz++) {
          size_t id1 = ((size_t)(iy * yratio) * (size_t) nx1 + (ix * xratio)) * (size_t) nz1 + iz * zratio;
          size_t id2 = ((size_t) iy * (size_t) nx2 + ix) * (size_t) nz2 + iz;
          data2[id2] = 1.0f / data[id1];
        }
      }
    }

    if(local) free(data);
    data = data2;
    local = true;


    //printf("MEM: shrink: [%d %d %d]. Total size %ld bytes.\n", nx2, ny2, nz2,
    //    static_cast<size_t>(nx2) * ny2 * nz2 * sizeof(float));
  }
}

void Fdm::info() {
  printf("FDM info:\n");
  printf("          orig: (x0,y0,z0) = %f,%f,%f\n", fdmh.x0, fdmh.y0, fdmh.z0);
  printf("          step: (dx,dy,dz) = %f,%f,%f\n", fdmh.dx, fdmh.dy, fdmh.dz);
  printf("          size: (nx,ny,nz) = %d,%d,%d\n", fdmh.nx, fdmh.ny, fdmh.nz);
  printf("      line inc: (xi,yi,zi) = %d,%d,%d\n", fdmh.xi, fdmh.yi, fdmh.zi);
  printf("    first line:              %d,%d,%d\n", fdmh.fx, fdmh.fy, fdmh.fz);
  printf("     last line:              %d,%d,%d\n", fdmh.lx, fdmh.ly, fdmh.lz);
  printf("     v min/max: %f / %f\n", fdmh.vmin, fdmh.vmax);
  fflush(stdout);
}

void Fdm::setheader(float *header) {
  int *iheader;
  char *magic;

  memset(header, 0, 512);

  iheader = (int *) header;

  header[0] = fdmh.x0;
  header[1] = (float) fdmh.nx;
  header[2] = fdmh.dx;

  header[3] = fdmh.y0;
  header[4] = (float) fdmh.ny;
  header[5] = fdmh.dy;

  header[6] = fdmh.z0;
  header[7] = (float) fdmh.nz;
  header[8] = fdmh.dz;

  // format unused
  header[9] = 0.0;
  header[10] = 0.0;
  header[11] = 0.0;

  header[12] = fdmh.vmin;
  header[13] = fdmh.vmax;

  header[14] = (float) fdmh.fx;
  header[15] = (float) fdmh.fy;

  header[16] = (float) fdmh.xi;
  header[17] = (float) fdmh.yi;

  magic = (char *)(header + 18);

  magic[0] = 'S';
  magic[1] = 'F';
  magic[2] = 'D';
  magic[3] = 'M';

  iheader[19] = 2; // version

  iheader[20] = 1; // distance unit

  // angle unit fixed to radians
  iheader[21] = 2; // angle unit

  // north angle
  header[22] = 0;

  // rotation angle
  header[23] = 0;
}

int Fdm::WriteHeader() {
  char *littleEndian;
  short siValue = 1;
  float *header = new float[128];
  memset(header, 0, 128 * sizeof(float));
  setheader(header);
  littleEndian = (char *) &siValue;
  //if (*littleEndian)  swapEndian32 (header, 128);
  int wsize = fwrite(header, sizeof(float), 128, fd);
  if(wsize < 128) {
    printf("ERROR (fdm_savecube): can't write header to the fdm file: %s!\n", filename.c_str());
    fflush(stdout);
    return 1;
  }

  delete[] header;
  return 0;
}

int Fdm::AppendData(float *myData, size_t thisSize) {
  size_t chunk_size;
  size_t length;
  char *littleEndian;
  short siValue = 1;
  littleEndian = (char *) &siValue;
  //if (*littleEndian) swapEndian32 (myData, thisSize);

  chunk_size = 8 * 1024 * 1024;
  float *ptr = myData;
  length = 0;

  while(length < thisSize) {
    if(length + chunk_size >= thisSize) {
      chunk_size = thisSize - length;
    }
    if(fwrite(ptr, sizeof(float), chunk_size, fd) < chunk_size) {
      printf("ERROR (fdm_savecube): failed to append the data %ld    %s\n", chunk_size, filename.c_str());
      fflush(stdout);
      return 1;
    }
    length += chunk_size;
    ptr += chunk_size;
  }

  //if (*littleEndian) swapEndian32 (myData, thisSize);
  return 0;
}

int Fdm::savecube() {
  size_t mysize = fdmh.nx * fdmh.ny;
  mysize *= fdmh.nz;
  FileOpen();
  //for(size_t i=0; i<mysize; i++)  if(ABS(data[i])>1.0e-12) printf("The FDM data Z %ld, %e\n", i, data[i]);
  if(WriteHeader()) return 1;
  if(AppendData(data, mysize)) return 1;
  FileClose();
  return 0;
}

void Fdm::unload() {
  if(!this->onDisk()) {
    this->savecube();
    if(local) free(data);
    data = NULL;
    local = true;
  }
}

void Fdm::load() {
  if(this->onDisk()) {
    if(local && data) free(data);
    data = NULL;
    local = true;
    string inFileName = this->filename;
    this->read(inFileName);
  }
}
void Fdm::sethead(float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz, int fx, int fy,
                  int fz, int xi, int yi, int zi) {
  fdmh.x0 = x0, fdmh.y0 = y0, fdmh.z0 = z0;
  fdmh.nx = nx, fdmh.ny = ny, fdmh.nz = nz;
  fdmh.dx = dx, fdmh.dy = dy, fdmh.dz = dz;
  fdmh.fx = fx, fdmh.fy = fy, fdmh.fz = fz;
  fdmh.xi = xi, fdmh.yi = yi, fdmh.zi = zi;
  fdmh.lx = fdmh.fx + (fdmh.nx - 1) * fdmh.xi;
  fdmh.ly = fdmh.fy + (fdmh.ny - 1) * fdmh.yi;
  fdmh.lz = fdmh.fz + (fdmh.nz - 1) * fdmh.zi;

}

void Fdm::setdata(float *_data) {
  data = _data;
  local = false;
}

// regrid fdm file from fdm to ofdm. recip = 1 is for interlation in slowness.
// input  fdm: fdm, ofdm. ofdm's header should be set already for regriding.
// output fdm: ofdm, ofdm->data has been regrid.
bool Fdm::regrid(Fdm *ofdm, int recip) {
  size_t ns;
  float as, bs, cs, ds;
  float xrat, yrat, x, y, fracx, fracy;
  float xm, ym, zm, oxm, oym, ozm;

  int iz, ix, iy;
  int indz1, indz2, iyrat, ixrat;
  int indy1, indy2, indx1, indx2;

  float z1, zrat1, zfrac;
  float *va, *vb, *vc, *vd, *pd;
  float *ss;

  //-- interpolating the ads velicty cube into v(nz,nx,ny)
  //
  //           |                     |
  //   indy2 --D---------------------C----
  //           |                     |
  //         __|..............X      |
  //           |              .      |
  //           |              .      |
  //      fracy|              .      |
  //           |              .      |
  //           |              .      |
  //   indy1 --A---------------------B----
  //           |<    fracx   >|      |
  //         indx1                 indx2

  FdmHeader &ofdmh = ofdm->getHeader();

  xm = fdmh.x0 + (fdmh.nx - 1) * fdmh.dx;
  ym = fdmh.y0 + (fdmh.ny - 1) * fdmh.dy;
  zm = fdmh.z0 + (fdmh.nz - 1) * fdmh.dz;
  oxm = ofdmh.x0 + (ofdmh.nx - 1) * ofdmh.dx;
  oym = ofdmh.y0 + (ofdmh.ny - 1) * ofdmh.dy;
  ozm = ofdmh.z0 + (ofdmh.nz - 1) * ofdmh.dz;

  if(oxm < fdmh.x0 || oym < fdmh.y0 || ozm < fdmh.z0) {
    printf("ERROR (fdm_regrid): There is no overlapping between input and output range.\n");
    printf("      input:  (x0,y0,z0) = (%f,%f,%f).\n", fdmh.x0, fdmh.y0, fdmh.z0);
    printf("      output: (xm,ym,zm) = (%f,%f,%f).\n", oxm, oym, ozm);
    return 0;
  }
  if(xm < ofdmh.x0 || ym < ofdmh.y0 || zm < ofdmh.z0) {
    printf("ERROR (fdm_regrid): There is no overlapping between input and output range.\n");
    printf("      input:  (xm,ym,zm) = (%f,%f,%f).\n", xm, ym, zm);
    printf("      output: (x0,y0,z0) = (%f,%f,%f).\n", ofdmh.x0, ofdmh.y0, ofdmh.z0);
    return 0;
  }

  ns = ((size_t) ofdmh.nx) * ofdmh.ny * ofdmh.nz;
  float *odata = (float *) calloc(ns, sizeof(float));
  if(!odata) {
    printf("ERROR (fdm_regrid): Can't alloc memory for output fdm data (%.2f(MB)).\n", ((ns / 1024.0) / 1024.0));
    return 0;
  }

  ss = (float *) malloc(sizeof(float) * fdmh.nz);

  for(iy = 0; iy < ofdmh.ny; iy++) {
    y = ofdmh.y0 + iy * ofdmh.dy;
    y = MAX(y, fdmh.y0);
    y = MIN(y, ym);

    yrat = (y - fdmh.y0) / fdmh.dy;
    iyrat = (int) yrat;
    fracy = yrat - iyrat;
    indy1 = MIN(iyrat, fdmh.ny - 1);
    indy2 = MIN(indy1 + 1, fdmh.ny - 1);

    for(ix = 0; ix < ofdmh.nx; ix++) {
      x = ofdmh.x0 + ix * ofdmh.dx;
      x = max(x, fdmh.x0);
      x = min(x, xm);

      xrat = (x - fdmh.x0) / fdmh.dx;
      ixrat = (int) xrat;
      fracx = xrat - ixrat;
      indx1 = min(ixrat, fdmh.nx - 1);
      indx2 = min(indx1 + 1, fdmh.nx - 1);

      as = (1. - fracx) * (1. - fracy);
      bs = fracx * (1. - fracy);
      cs = fracx * fracy;
      ds = (1. - fracx) * fracy;

      va = data + (indy1 * fdmh.nx + indx1) * fdmh.nz;
      vb = data + (indy1 * fdmh.nx + indx2) * fdmh.nz;
      vc = data + (indy2 * fdmh.nx + indx2) * fdmh.nz;
      vd = data + (indy2 * fdmh.nx + indx1) * fdmh.nz;

      if(recip) {
        for(iz = 0; iz < fdmh.nz; iz++)
          ss[iz] = as * 1.f / va[iz] + bs * 1.f / vb[iz] + cs * 1.f / vc[iz] + ds * 1.f / vd[iz];

        // for now the interval velocity for migration is defined as the velocity below the depth
        // simple interpolation
        pd = odata + (iy * ofdmh.nx + ix) * ofdmh.nz;
        for(iz = 0; iz < ofdmh.nz; iz++) {
          z1 = ofdmh.z0 + iz * ofdmh.dz;

          if(z1 <= fdmh.z0) {
            pd[iz] = 1.f / ss[0];
          } else if(z1 >= zm) {
            pd[iz] = 1.f / ss[fdmh.nz - 1];
          } else {
            zrat1 = (z1 - fdmh.z0) / fdmh.dz;
            indz1 = (int) zrat1;
            zfrac = zrat1 - indz1;
            indz2 = MIN(indz1 + 1, fdmh.nz - 1);
            pd[iz] = 1.f / ((1 - zfrac) * ss[indz1] + zfrac * ss[indz2]);
          }
        }
      } else {
        for(iz = 0; iz < fdmh.nz; iz++)
          ss[iz] = as * va[iz] + bs * vb[iz] + cs * vc[iz] + ds * vd[iz];

        // for now the interval velocity for migration is defined as the velocity below the depth
        // simple interpolation
        pd = odata + (iy * ofdmh.nx + ix) * ofdmh.nz;
        for(iz = 0; iz < ofdmh.nz; iz++) {
          z1 = ofdmh.z0 + iz * ofdmh.dz;

          if(z1 <= fdmh.z0) {
            pd[iz] = ss[0];
          } else if(z1 >= zm) {
            pd[iz] = ss[fdmh.nz - 1];
          } else {
            zrat1 = (z1 - fdmh.z0) / fdmh.dz;
            indz1 = (int) zrat1;
            zfrac = zrat1 - indz1;
            indz2 = min(indz1 + 1, fdmh.nz - 1);
            pd[iz] = (1 - zfrac) * ss[indz1] + zfrac * ss[indz2];
          }
        }
      }
    }
  }

  free(ss);

  return 1;
}

float Fdm::getvalue(vector3 xloc) const {
  return getvalue(xloc.x, xloc.y, xloc.z);
}

float Fdm::getvalue(float x, float y, float z) const {
  float ax = (x - fdmh.x0) / fdmh.dx;
  float ay = (y - fdmh.y0) / fdmh.dy;
  float az = (z - fdmh.z0) / fdmh.dz;
  int ix = (int) floorf(ax);
  int iy = (int) floorf(ay);
  int iz = (int) floorf(az);
  float bx = ax - ix;
  float by = ay - iy;
  float bz = az - iz;
  ix = MAX(0, MIN(fdmh.nx - 2, ix));
  iy = MAX(0, MIN(fdmh.ny - 2, iy));
  iz = MAX(0, MIN(fdmh.nz - 2, iz));
  float weight[8];
  size_t xID[8];
  weight[0] = (1 - bx) * (1 - by) * (1 - bz);
  weight[1] = bx * (1 - by) * (1 - bz);
  weight[2] = (1 - bx) * by * (1 - bz);
  weight[3] = bx * by * (1 - bz);
  weight[4] = (1 - bx) * (1 - by) * bz;
  weight[5] = bx * (1 - by) * bz;
  weight[6] = (1 - bx) * by * bz;
  weight[7] = bx * by * bz;
  int ixp1 = min(fdmh.nx - 1, ix + 1); // avoid outbound on nx=1
  int iyp1 = min(fdmh.ny - 1, iy + 1); // avoid outbound on ny=1
  int izp1 = min(fdmh.nz - 1, iz + 1); // avoid outbound on nz=1
  //  printf("The get Value %d  %d  %d, +1: %d %d %d. FDM: %d %d %d \n", ix, iy, iz, ixp1, iyp1, izp1, fdmh.nx, fdmh.ny,
  //      fdmh.nz);
  xID[0] = iz + fdmh.nz * (ix + fdmh.nx * iy);
  xID[1] = iz + fdmh.nz * (ixp1 + fdmh.nx * iy);
  xID[2] = iz + fdmh.nz * (ix + fdmh.nx * iyp1);
  xID[3] = iz + fdmh.nz * (ixp1 + fdmh.nx * iyp1);
  xID[4] = izp1 + fdmh.nz * (ix + fdmh.nx * iy);
  xID[5] = izp1 + fdmh.nz * (ixp1 + fdmh.nx * iy);
  xID[6] = izp1 + fdmh.nz * (ix + fdmh.nx * iyp1);
  xID[7] = izp1 + fdmh.nz * (ixp1 + fdmh.nx * iyp1);
  float rev = 0;
  for(int i = 0; i < 8; i++) {
    // printf("xID[%d]=%ld\n", i, xID[i]);
    rev += weight[i] * data[xID[i]];
  }
  return rev;
}

float Fdm::getvalue(vector2 xloc) const {
  float ax = (xloc.x - fdmh.x0) / fdmh.dx;
  float az = (xloc.z - fdmh.z0) / fdmh.dz;
  float bx = (ax + 32768) - int(ax + 32768);
  float bz = (az + 32768) - int(az + 32768);
  int ix = int(ax);
  int iz = int(az);
  ix = MAX(0, MIN(fdmh.nx - 2, ix));
  iz = MAX(0, MIN(fdmh.nz - 2, iz));
  float weight[4];
  size_t xID[4];
  weight[0] = (1 - bx) * (1 - bz);
  weight[1] = bx * (1 - bz);
  weight[2] = (1 - bx) * bz;
  weight[3] = bx * bz;
  xID[0] = iz + fdmh.nz * ix;
  xID[1] = iz + fdmh.nz * (ix + 1);
  xID[2] = iz + 1 + fdmh.nz * ix;
  xID[3] = iz + 1 + fdmh.nz * (ix + 1);

  float rev = 0;
  for(int i = 0; i < 4; i++)
    rev += weight[i] * data[xID[i]];
  return rev;
}

void Fdm::swapEndian32(void *buf, size_t len) {
  size_t i;
  char c0, c1, *cp;
  float *fbuf = (float *) buf;

  for(i = 0; i < len; i++, fbuf++) {
    cp = (char *) fbuf;
    c0 = cp[0];
    c1 = cp[1];
    cp[0] = cp[3];
    cp[1] = cp[2];
    cp[2] = c1;
    cp[3] = c0;
  }
}

void saveFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, const char *path) {
  Fdm fdm;
  if(nx == 1 && ny > 1) fdm.sethead(0, 0, 0, ny, nx, nz, dy, dx, dz, 1, 1, 0, 1, 1, 1);
  else fdm.sethead(0, 0, 0, nx, ny, nz, dx, dy, dz, 1, 1, 0, 1, 1, 1);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 const char *path) {
  Fdm fdm;
  if(dx == 0) dx = 10.0f;
  if(dy == 0) dy = 10.0f;
  int fx = roundf(x0 / dx) + 1;
  int fy = roundf(y0 / dy) + 1;
  int fz = 0;
  if(nx == 1 && ny > 1) fdm.sethead(y0, x0, z0, ny, nx, nz, dy, dx, dz, fy, fx, fz, 1, 1, 1);
  else fdm.sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, fx, fy, fz, 1, 1, 1);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 int ifx, int ify, int ifz, int incx, int incy, int incz, const char *path) {
  Fdm fdm;
  if(nx == 1 && ny > 1) fdm.sethead(y0, x0, z0, ny, nx, nz, dy, dx, dz, ify, ifx, ifz, incy, incx, incz);
  else fdm.sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, ifx, ify, ifz, incx, incy, incz);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 float cdpdx, float cdpdy, const char *path) {
  Fdm fdm;
  if(dx == 0) dx = 10.0f;
  if(dy == 0) dy = 10.0f;
  int xi = dx / cdpdx;
  int yi = dy / cdpdy;
  int zi = 1;
  int fx = roundf(x0 / cdpdx) + 1;
  int fy = roundf(y0 / cdpdy) + 1;
  int fz = 0;
  if(nx == 1 && ny > 1) fdm.sethead(y0, x0, z0, ny, nx, nz, dy, dx, dz, fy, fx, fz, xi, yi, zi);
  else fdm.sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, fx, fy, fz, xi, yi, zi);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

void saveFdmCube(float *data, const char *path, float x0, float y0, float z0, int nx, int ny, int nz, float dx,
                 float dy, float dz, float gx0, float gy0, float gz0) {
  Fdm fdm;
  if(dx == 0) dx = 10.0f;
  if(dy == 0) dy = 10.0f;
  if(dz == 0) dz = 10.0f;
  int fx = roundf((x0 - gx0) / dx) + 1;
  int fy = roundf((y0 - gy0) / dy) + 1;
  int fz = roundf((z0 - gz0) / dz);
  if(nx == 1 && ny > 1) fdm.sethead(y0, x0, z0, ny, nx, nz, dy, dx, dz, fy, fx, fz, 1, 1, 1);
  else fdm.sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, fx, fy, fz, 1, 1, 1);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

void saveGlobalModel(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                     const char *path) {
  Fdm fdm;
  int fx = 1;
  int fy = 1;
  int fz = 0;
  if(nx == 1 && ny > 1) fdm.sethead(y0, x0, z0, ny, nx, nz, dy, dx, dz, fy, fx, fz, 1, 1, 1);
  else fdm.sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, fx, fy, fz, 1, 1, 1);
  fdm.setdata(data);
  fdm.savecube(path);
  fdm.setdata(0);
}

// cut traces
void cutFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, int n0, int newnz,
                const char *path) {
  int i;
  float *tmp, *pt, *pd;

  if(n0 + newnz >= nz) {
    printf("ERROR: out of z-range: n0 (%d) + newnz (%d) > nz (%d).\n", n0, newnz, nz);
    return;
  }

  tmp = (float *) calloc(nx * ny * newnz, sizeof(float));
  pt = tmp;
  pd = data + n0;
  for(i = 0; i < nx * ny; i++) {
    memcpy(pt, pd, sizeof(float) * newnz);
    pt += newnz;
    pd += nz;
  }

  saveFdmCube(tmp, nx, ny, newnz, dx, dy, dz, path);

  free(tmp);
}

// expand one trace to severals
void expandFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, const char *path) {
  int i;
  float *tmp = (float *) calloc(nx * ny * nz, sizeof(float));
  float *pt = tmp;
  for(i = 0; i < nx * ny; i++) {
    memcpy(pt, data, sizeof(float) * nz);
    pt += nz;
  }

  saveFdmCube(tmp, nx, ny, nz, dx, dy, dz, path);

  free(tmp);
}

void Fdm::readBinary(string &fileName, int nx1, int ny1, int nz1, int ixOffset, int iyOffset, int xratio, int yratio,
                     int zratio) {
  readBinary(fileName.c_str(), nx1, ny1, nz1, ixOffset, iyOffset, xratio, yratio, zratio);
}

void Fdm::readBinary(string &fileName) {
  readBinary(fileName.c_str());
}

void Fdm::readVelBin(string &fileName, int nx1, int ny1, int ixOffset, int iyOffset) {
  readVelBin(fileName.c_str(), nx1, ny1, ixOffset, iyOffset);
}

int Fdm::read(string &path, float *data, float x0, float y0, int nx, int ny) {
  return read(path.c_str(), data, x0, y0, nx, ny);
}

int Fdm::read(string &path, float *vol) {
  return read(path.c_str(), vol);
}

void Fdm::FileClose() {
  if(fd) {
    if(fflush(fd) == EOF) {
      // fprintf(stderr, "%s: %s\n", hostname, explain_fflush(fd));
      fprintf(stderr, "%s: fflush error\n", hostname);
      exit(EXIT_FAILURE);
    }
    if(fclose(fd) == EOF) {
      // fprintf(stderr, "%s: %s\n", hostname, explain_fclose(fd));
      fprintf(stderr, "%s: fclose error\n", hostname);
      exit(EXIT_FAILURE);
    }
    fd = NULL;
  }
}

bool Fdm::haveSmallValues(float threshold) {
  if(data == NULL)
    return false;
  size_t n = ((size_t) fdmh.nx) * fdmh.ny * (size_t) fdmh.nz;
  for(size_t i = 0; i < n; i++)
    if(fabs(data[i]) < threshold)
      return true;
  return false;
}

bool Fdm::haveNanInf() {
  if(data == NULL)
    return false;
  size_t n = ((size_t) fdmh.nx) * fdmh.ny * (size_t) fdmh.nz;
  for(size_t i = 0; i < n; i++)
    if(isnanf(data[i]) || isinff(data[i]))
      return true;
  return false;
}

