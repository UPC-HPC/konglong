#ifndef PROPAGATOR_H
#define PROPAGATOR_H

#include <time.h>
#include <memory>
#include <unistd.h>

#include <jseisIO/jseisUtil.h>
using jsIO::oJseis3D;
using jsIO::oJseisND;
using jsIO::jseisUtil;

#include "boundarytaper.h"
#include "Grid.h"
#include "OutputGrid.h"
#include "Model.h"
#include "volumefilter.h"
#include "ModelVolumeID.h"
#include "ModelPrepare.h"
#include "Source.h"
#include "Receiver.h"
#include "ExtractReceiverData.h"
using std::shared_ptr;
using std::make_shared;

//forward declarations
class WaveFieldCompress;
class Wavefield;
class ModelLoader;
class Derivative;
class KernelCPU;
class ExtractReceiverData;
class Boundary;
class Propagator {
public:
  shared_ptr<Grid> myGrid { };
  Model *myModel = 0;
  Source *myshot = 0;
  shared_ptr<Receiver> receivers[2] { };
  BndTaper *myBndTaper = 0;
  float apertx = 0;
  float aperty = 0;
  float zmin;
  float zmax;
  float zsurf = 0; // z for water surface, currently defined as -"zMin" (zmin is always 0 as a hack)
  float maxFreq;
  float minFreq;
  int dim;     // bit combination of enum Dimension
  int nxbnd;     // thickness of boundary in X
  int nybnd;     // thickness of boundary in Y
  int nzbnd;     // thickness of boundary in Z
  float dxmax;     // maximum X increasement  do not go crazy, regular grid dx on propagation
  float dymax;
  float dzmin;
  float dxreg = 0;     // final output grid on X  use for snap the midx midy to the grid for regular grid
  float dyreg = 0;     // final output grid on Y
   float x0reg = 0;
  float y0reg = 0;
  float velmin = 0;     // Minimum velocity
  float tmax;
  float dtpro = 0;
  float dtsav = 0;
  int gridType = 0;
  int engineType;
  size_t gridsize = 0;
  int nvmodel = 0;
  int count_mem3d = 0;
  float **volModel;
  Wavefield *myWavefield = 0;
  Wavefield *myDemigWavefield = 0;
  int nThreads;
  int nt = 0, it0 = 0;
  PROP::Direction prop_dir = PROP::FORWARD;
  vector<int> sfBoundType { 0, 0 };
  size_t nxz = 0;
  vector3 recvMin;
  vector3 recvMax;
  float drangex;
  float drangey;
  WaveFieldCompress *wfComp;
  ModelLoader *modelLoader = 0;
  Boundary *bnd;
  Boundary *bndDemig;
  Derivative *derive;
  Derivative *deriveDemig;
  KernelCPU *kernelCPU = 0;
  KernelCPU *kernelDemig = 0;
  float vmax = 0;
  int nzuppad = 0;
  bool allocWx;
time_t time_start = { };
  char hostname[_POSIX_HOST_NAME_MAX];
  vector<int> posLogical;

public:
  Propagator(float zmin, float zmax, float maxFreq, float tmax);
  ~Propagator();
  static vector<float> sourceDelayUpdateLoc(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID,
      const Traces *traces = nullptr);
  static unique_ptr<ExtractReceiverData> mod(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID,
      vector3 &recvMin, vector3 &recvMax, Traces *traces = nullptr);
  static shared_ptr<Grid> rtm(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID, vector3 &recvMin,
      vector3 &recvMax, vector<Traces*> vec_traces = vector<Traces*> { nullptr }, vector<float*> destImages = vector<float*> { },
      OutputGrid *destGrid = nullptr);

  void prepare(float xMinValid, float xMaxValid, float yMinValid, float yMaxValid, vector3 recvMin, vector3 recvMax);
  void reprepare();
  void ic_prepare();
  void allocvolume();
  void populateModelDARemval();
  void demigration(unique_ptr<ExtractReceiverData> &erd, PROP::Direction direct);   // save all the wave fields
  void modelling(unique_ptr<ExtractReceiverData> &erd, PROP::Direction direct);
  void migration(CacheFile *outvol, PROP::Direction direct);
  unique_ptr<Traces> setupReceiver(int spreadSize = 0, int i_srcrcv = 1, unique_ptr<Traces> traces = { }, bool keepTraces = false);
  int setupReceiver(PROP::Operation oper, int spreadSize, int i_srcrcv, vector<float> &x, vector<float> &y, vector<float> &z,
      vector<float> &delay);
  bool determine_ghost(PROP::Direction direct);
  shared_ptr<Receiver> setupReceiverForModeling(int spreadSize, unique_ptr<Traces> traces = { });
  void copySnapshotGrid(float *w0, float *ws, int nElements) const;
  void cleanVolumes();
  void ModelBoundaryTaper(int thickness = 8);
  void TaperVolume(float *vol, int thickness);
  void fillVolume(float *vol, float v);
  void copyValue(float *vol, int iz0);
 void Vel2(float dt);
  void WaveFieldTaper();
  float calcDeltaTime();
  float getDtSave(float recordingFreq);
  static float getDtSave(); // _dt_pro need to be known
  void freeVolume(int flag);
  void printTime(int it, PROP::Direction direct, float *w);
  void kernel();
  void kernel(Wavefield *myWavefield, Derivative *deriveDemig, KernelCPU *myKernel);

  float getDiskSize(int ntsave);
  vector<float> getIcLiveFull(float tmax);

private:

  void CalVTIscaler(Wavefield *myWavefield);
  void CalTTIscaler(Wavefield *myWavefield);
  void divideRho(Wavefield *wf);
  void multiplyRho(Wavefield *wf);
//  void multiplycnnRho(Wavefield *wf);
  void CalVTIscaler2D(Wavefield *myWavefield);
  void CalTTIscaler2D(Wavefield *myWavefield);
  void update2nd(Wavefield *myWavefield);
  void update2nd(float *w0, float *w1, float *wb);
  void ApplyScaler(float *wb, float *wr);
  void tapersource(float *ws, int nx, int ny, int nz, int ix0, int iy0, int iz0);
  void applyVel(Wavefield *myWavefield);
  void applyReflectivity(Wavefield *mySrcWavefield, Wavefield *myRecWavefield);
  void apply_symmetry(float *wf, int sym);
//  void apply_symmetry_backup(float *wf);
//  void apply_symmetry_backup1(float *wf, int sym);
  void boundary_attenuation(float *wf, float *wf0, float dt);
 float* allocMem3d(int nx, int ny, int nz, int nThreads);
  void allocWavefields();
  void allocModels();
  void getGridType();
  void addVelSurface();
  void model_bnd_smooth(int iModelVolID, int nxbnd, int nybnd, int nzbnd);
  void rhoCanonical();
  float* SqrtVol(float *volin);
  float* SmoothingVol(float *volin);
  float* CanonicalVol(float *sqrtRho, float *Rho);
  void qcVel(string ext);
  bool prepSnapJS(bool &snap3D, unique_ptr<oJseis3D> &js_snap, unique_ptr<oJseis3D> &js_snap_y, unique_ptr<oJseisND> &js_snap3D,
      vector<float> &buf_xslice, int nsnap, int isav, PROP::Direction direct = PROP::NA);
  void saveSnap(int i_snap, bool snap3D, unique_ptr<oJseis3D> &js_snap, unique_ptr<oJseis3D> &js_snap_y, unique_ptr<oJseisND> &js_snap3D,
      vector<float> &buf_xslice);

  void count_prop_memomry();

};

#endif

