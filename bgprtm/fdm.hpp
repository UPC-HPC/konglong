#ifndef FDM_H
#define FDM_H

#ifndef MIN
#define MIN(x,y)((x) < (y) ? (x) : (y))
#define MAX(x,y)((x) > (y) ? (x) : (y))
#endif

#include <string.h>
#include <string>
#include "stdio.h"
#include "Vector3.h"
#include <iostream>

using std::string;

struct FdmHeader {
  float x0, y0, z0;          // origin
  int nx, ny, nz;          // size
  float dx, dy, dz;          // step
  int fx, fy, fz;          // minimun of crossline, inline and z-line
  int lx, ly, lz;          // maximun of crossline, inline and z-line
  int xi, yi, zi;          // crossline, inline and z-line increments

  float vmin, vmax;           // minimun and maximum value of the cube
    void print(string info = ""){
        std::cout<<info<<" x0 "<<x0<<" y0 "<<y0<<" z0 "<<z0<<std::endl;
        std::cout<<info<<" nx "<<nx<<" ny "<<ny<<" nz "<<nz<<std::endl;
        std::cout<<info<<" dx "<<dx<<" dy "<<dy<<" dz "<<dz<<std::endl;
        std::cout<<info<<" fx "<<dx<<" fy "<<dy<<" fz "<<dz<<std::endl;
        std::cout<<info<<" lx "<<dx<<" ly "<<dy<<" lz "<<dz<<std::endl;
        std::cout<<info<<" xi "<<dx<<" yi "<<dy<<" zi "<<dz<<std::endl;
    }
};
class Fdm {
public:
  Fdm();
  ~Fdm();

  void reset();
  Fdm(int nx0, int ny0, int nz0);
  Fdm(int nx0, int ny0, int nz0, float dx0, float dy0, float dz0);
  int readheader(const char *path) {
    setFileName(path);
    FileOpenRead();
    if(readheader()) return 1;
    FileClose();
    return 0;
  }
  int readheader();
  int read(const char *path, float *vol = NULL);
  int read(string &path, float *vol = NULL);
  int read(const char *path, float *data, float x0, float y0, int nx, int ny);
  int read(string &path, float *data, float x0, float y0, int nx, int ny);
  void info();
  int savecube(const char *path) {
    setFileName(path);
    return savecube();
  }

  int savecube();
  void sethead(float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz, int fx, int fy,
               int fz, int xi, int yi, int zi);
  void setdata(float *data);
  float getvalue(float x, float y, float z) const;
  float getvalue(vector3 xloc) const;
  float getvalue(vector2 xloc) const;
  float *getdata() {
    if(!onDisk())load();
    return data;
  }
  bool regrid(Fdm *ofdm, int recip);
  void setFileName(const char *fname);
  int FileOpen();
  int FileOpenRead();
  void FileClose();
  int WriteHeader();
  int AppendData(float *myData, size_t thisSize);
  void CleanData(float a = 0) {
    for(size_t i = 0; i < fdmh.nx * fdmh.ny * size_t(fdmh.nz); i++)
      data[i] = a;
  }
  FdmHeader &getHeader() {
    return fdmh;
  }
  void isResample(const char *fileName, int maxFreq, int &xratio, int &yratio, int &zratio, float vMin);
  void readBinary(const char *fileName, int nx1, int ny1, int nz1, int ixOffset, int iyOffset, int xratio, int yratio,
                  int zratio);
  void readBinary(string &fileName, int nx1, int ny1, int nz1, int ixOffset, int iyOffset, int xratio, int yratio,
                  int zratio);
  void readVelBin(const char *fileName, int nx1, int ny1, int ixOffset, int iyOffset);
  void readVelBin(string &fileName, int nx1, int ny1, int ixOffset, int iyOffset);
  void decimateRatio(float maxFreq, int &xratio, int &yratio, int &zratio, int nThreads);

  void readBinary(const char *fileName);
  void readBinary(string &fileName);

  static int do_mkdir(const char *path, mode_t mode);
  static int mkpath(const char *path, mode_t mode);
  static int mkpath4file(const char *file, mode_t mode);

  void fillHeader(FdmHeader &header);

  bool onDisk() {return data == NULL; }
  void load(); // read the data from disk, and not touch the header
  void unload(); // put the data into disk, and free the memory, not touch the header

  bool haveSmallValues(float threshold);
  bool haveNanInf();

public:
  FdmHeader fdmh;
  float *data;
  bool swap;
  bool local;
  string filename;
  FILE *fd;
  char hostname[256];
  void setheader(float *header);
  void swapEndian32(void *buf, size_t len);
  void fillHeader(FdmHeader *header, float *buf);
};

void saveFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, const char *path);
void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 const char *path);
void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 float cdpdx, float cdpdy, const char *path);
void saveFdmCube(float *data, const char *path, float x0, float y0, float z0, int nx, int ny, int nz, float dx,
                 float dy, float dz, float gx0 = 0.0f, float gy0 = 0.0f, float gz0 = 0.0f);
void saveFdmCube(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                 int ifx, int ify, int ifz, int incx, int incy, int incz, const char *path);
void expandFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, const char *path);
void cutFdmCube(float *data, int nx, int ny, int nz, float dx, float dy, float dz, int n0, int newnz,
                const char *path);
void saveGlobalModel(float *data, float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz,
                     const char *path);

#endif

