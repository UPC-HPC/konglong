#ifndef SOURCE_H
#define SOURCE_H

#include "Grid.h"
#include "CacheFile.h"
#include "Vector3.h"
#include "Util.h"
#include "Wavelet.h"

class Wavefield;

enum SourceType {
  POINT = 1, SURFACE = 2, BODY = 3
};
// 3 types of sources

/*
 #define ABSORB 0x00
 #define SOURCE_GHOST 0x01
 #define RECEIVER_GHOST 0x02
 #define GHOST 0x03 // SOURCE_GHOST & SOURCE_GHOST
 #define ANTISYM_WF 0x10
 #define SYMMETRIC_WF 0x20 // place holder, probably useless anyway
 #define FREESURFACE 0x11 // ANTISYM_WF & SOURCE_GHOST
 */

#define ABSORB 0
#define SOURCE_GHOST 1
#define RECEIVER_GHOST 2
#define GHOST 3 // SOURCE_GHOST & SOURCE_GHOST
#define ANTISYM_WF 4
#define SYMMETRIC_WF 5 // place holder, probably useless anyway
#define FREESURFACE 6 // ANTISYM_WF & SOURCE_GHOST

namespace PROP {
enum Direction { // Occasionally this is used interchangeable with Receiver::SPREADID  SOURCE = 0, RECEIVER = 1, make sure they are same int value
  NA = -1, FORWARD = 0, BACKWARD = 1
};
enum Operation {
  MOD = 0, RTM = 1, DEMIG = 2, RTMM = 3
};
}

class Source {
public:
  SourceType sourceType;   // type of the source, point, surface, body    1, 2, 3
  int sfBoundType;
  Grid *grid;   // SOurce was defined on an area of a grid
  int nsx;   // size of the source Note it may be >1 even for point source (The spread)
  int nsy;
  int nsz;
  int ix0;   // position at grid
  int iy0;
  int iz0;
  int nt;
  int it0;   // the position of t==0
  float t0;
  bool userWavelet;   // the user wavelet flag
  float dt, dt_dxyz;
  float slow;
  float zsurf = 0;
  size_t mysize;   // == nsx*nsy*nsz*nt
  float *vol4d;   // a 4D volume for source wave field
  float *vol4d90;   // after 90 degree phase shift
  float *point;
  CacheFile *vfile;
  int nThreads;
  Source(Grid *grid0, int nthread = 1);
//  Source(SourceType type0, Grid *grid0, int nth);by wolf
  ~Source();
  Wavelet* setupWavelet(int wtype, int phaseType, float slow0, int nt0, float dt0, float maxfreq, PROP::Operation oper, int dim,
      float fhigh = 0.0f, float flow = 0.0f, float min_t_delay = 0);
//  void SetSourcePoint(vector3 x, int spread, int wtype, int phaseType, float slow0, int nt0, float dt0, float maxfreq, PROP::Operation oper,
//      int dim, float fhigh = 0.0f, float flow = 0.0f);//by wolf

//  void SetSourceBody(CacheFile *myfile);
//  void SetSourceBody(); // by wolf
  void create();
//  void SourceSpread(float *mylet, vector3 x, int spread);// by wolf
  //    void SourceSpread_backup(float* mylet, vector3 x, int spread);
  //    void SourceSpreadXY_backup(float* mylet, vector3 x, int spread);
  //    void SourceSpreadXY(float* mylet, vector3 x, int spread);
//  void PutTrace(int ix, int iy, int iz, float *buffs);// by wolf
//  void apply(Wavefield *myWavefield, int it);// by wolf
//  void purerotate(float *wavelet, float mydegree);// by wolf
//  void rotate(float *wavelet, float mydegree);// by wolf

  static int getWaveletType();
  static int getWaveletPhaseType();
  static int getSurfaceType(PROP::Direction id);
  static vector<float> getSpectrumPowPhase(PROP::Direction direction, PROP::Operation oper, int dim_bits, int verbose = 0);
};

#endif

