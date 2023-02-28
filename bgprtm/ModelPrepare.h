#ifndef MODELPREPARE_H_
#define MODELPREPARE_H_


#include "ModelVolumeID.h"
#include <string>
using std::string;

class Grid;
class Fdm;
class Model;
class ModelLoader;

enum Dimension { // bits
  OneD = 1, TwoD = 2, ThreeD = 4, Sim3D = 8
};

class ModelPrepare {
public:
  /*
   *
   */
  ModelPrepare(Grid *myGrid, Model *myModel, ModelLoader *modelLoader, int nThreads);

  /*
   *
   */
  virtual ~ModelPrepare();

  static int getDimension();

  void velPrepare(const char *velFile);
  void velPrepare(string velFile);
  void velPrepare_backup(const char *velFile);
  void velPrepare_backup(string velFile);

  void vel2Prepare(const char *vel2File);
  void vel2Prepare(string vel2File);
  void vel2Prepare_backup(const char *vel2File);
  void vel2Prepare_backup(string vel2File);

  void rhoPrepare(string rhoFile);
  void rhoPrepare_backup(string rhoFile);

  void reflectivityPrepare(string reflectivityFile);

  void qPrepare(string qFile);

  void vtiPrepare(const char *epsFile, const char *delFile);
  void vtiPrepare_backup(const char *epsFile, const char *delFile);
  void vtiPrepare(string epsFile, string delFile);
  void vtiPrepare_backup(string epsFile, string delFile);


  void ttiPrepare(const char *pjxFile, const char *pjyFile);
  void ttiPrepare_backup(const char *pjxFile, const char *pjyFile);
  void ttiPrepare(string pjxFile, string pjyFile);
  void ttiPrepare_backup(string pjxFile, string pjyFile);
  float *CanonicalVol(float *sqrtRho, float *Rho);
  float *SqrtVol(float *volin);
  float calcDeltaTime();

private:
  float *allocMem3d(int nx, int ny, int nz, int nThreads) const;


  void Prepare4ModelRegrid(Fdm *myfdm);

  float *convertToComputeVol(const Model &model,
                             const Grid &grid,
                             ModelVolID id,
                             int nThreads) const;

  float *AniRegrid(Fdm *myfdm, int isgn);
  float *SmoothingVol(float *volin);

  void freeMemory();

private:
  Grid               *myGrid;
  Model             *myModel;
  ModelLoader *myModelLoader;
  int               nThreads;
  float              *volVel;
  float              *volVel2; //for dual flood;
  float              *volEps;
  float              *volDel;
  float              *volPjx;
  float              *volPjy;
  float              *volRho;
  float              *volReflectivity;
  float              *volQ = 0;
  float              invQmax = 0;
};



#endif /* MODELPREPARE_H_ */

