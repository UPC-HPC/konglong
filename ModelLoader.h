#ifndef MODEL_LOADER_H
#define MODEL_LOADER_H

#include "ModelVolumeID.h"
#include "Model.h"
#include "model_builder.h"

#define GLOBAL "global"
#define LOCAL "local"
#define VEL_LOCAL_DEFAULT "vel_local.fdm"
#define VEL2_LOCAL_DEFAULT "vel_dual_local.fdm"
#define RHO_LOCAL_DEFAULT "rho_local.fdm"
#define REFLECTIVITY_LOCAL_DEFAULT "reflectivity_local.fdm"
#define Q_LOCAL_DEFAULT "Q_local.fdm"
#define EPS_LOCAL_DEFAULT "eps_local.fdm"
#define DEL_LOCAL_DEFAULT "del_local.fdm"
#define PJX_LOCAL_DEFAULT "pjx_local.fdm"
#define PJY_LOCAL_DEFAULT "pjy_local.fdm"
#define DIPX_LOCAL_DEFAULT "dipx_local.fdm"
#define DIPY_LOCAL_DEFAULT "dipy_local.fdm"
#define DIP_LOCAL_DEFAULT "dip_local.fdm"
#define AZIMUTH_LOCAL_DEFAULT "azimuth_local.fdm"
#define GEOM_LOCAL_DEFAULT "geometry_local.bin"

class ModelLoader {
public:
  /// Constructor.
  ///
  /// @param model      The model to load data into.
  /// @param nThreads   Number of threads.
  ModelLoader(Model &model, const char *key, int nThreads);

  virtual ~ModelLoader();

  Node read_grid(Node trunk, Node branch, int &nz, int &nx, int &ny, float &dz, float &dx, float &dy, float &z0,
                 float &x0, float &y0);

  // load the prepared local models, FDMs only
  void loadLocalModels(float **volModel, float x0, float y0, int nx, int ny, float &vmax);
  void reloadLocalModels(float **volModel, float x0, float y0, int nx, int ny, float &vmax, ModelVolID id);
  ///
  /// Loads a given volume from file into the model.
  /// @note: Special case for PJX: will load PJY as well.
  ///
  /// @param id   The volume to load. The fdm put into
  ///             model.fdms[id].
  void loadModelVolume(ModelVolID id);

  virtual void saveModels();

  virtual void loadFile(string &file, float *vol);
  Fdm *loadFile(string modKey, ModelVolID id);
  void loadTTIFile();

  float getXMinValid() {
    return xMinValid;
  }

  float getXMaxValid() {
    return xMaxValid;
  }

  float getYMinValid() {
    return yMinValid;
  }

  float getYMaxValid() {
    return yMaxValid;
  }

  void loadFdmFile(string &fileName, float *vol);
  void loadBinFile(string &fileName, float *vol);
  void loadRecordFile(string &fileName, float *vol);

protected:
  Model &model;
  string key;
  int nThreads;
  float xMinValid, xMaxValid, yMinValid, yMaxValid;

  int nx, ny, nz;
  float dx, dy, dz;
  float x0, y0, z0;
  int xratio, yratio, zratio;
  float elevation_shift;
};

#endif

