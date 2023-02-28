/*
 * ImaingCondition.h
 *
 */

#ifndef IMAINGCONDITION_H_
#define IMAINGCONDITION_H_

#include "GatherInfo.h"
#include "OutputGrid.h"
#include "Lagrange.h"
#include <unistd.h>
#include <memory>
#include "ImagingCorrelator.h"
#include "WaveFieldHolder.hpp"
#include "ImageHolder.hpp"

#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;
using jsIO::oJseis3D;

using std::string;
using std::shared_ptr;

// forward declarations
class CacheFile;
class Grid;
class WaveFieldCompress;
class vector3;
class Interpolant;

enum ImageType {
  STACK = 0, OFCIG = 1, DTCIG = 2, ADCIG = 3, DECON = 4
};
enum Compensation {
  ILLUM_NONE = 0, ILLUM_SHOT = 1, ILLUM_GLOBAL = 2
};

/** RTM Imaging Conditions
 *
 */
class ImagingCondition {
public:
  /** ctor
   *
   */
  ImagingCondition(CacheFile *srcWaveField, CacheFile *recWaveField, ImageType imgType = STACK);

  /** dtor
   *
   */
  virtual ~ImagingCondition();

  shared_ptr<Grid> getPartialImageGrid(float xMinValid, float xMaxValid, float yMinValid, float yMaxValid, float zmax, OutputGrid *ogrid);
  shared_ptr<Grid> genImage(Grid *grid, WaveFieldCompress *wfComp, float xMinValid, float xMaxValid, float yMinValid, float yMaxValid,
      float dtsav, vector<float*> destImages = vector<float*> { }, OutputGrid *destGrid = nullptr);

  void getVel(float *vel, float dt);

  static void saveImage(const char *rawImageFile, float *regimg, float fz, float dz, int nz, float fx, float dx, int nx, float fy, float dy,
      int ny);

  static void saveImage(string rawImageFile, float *regimg, float fz, float dz, int nz, float fx, float dx, int nx, float fy, float dy,
      int ny); // string: do not use reference so Node.as<string>() can be directly passed

  static void saveImage(const char *imageFile, float *image, float fz, float dz, int nz, float fx, float dx, int nx, float fy, float dy,
      int ny, float cdpdx, float cdpdy);

  static void saveImage(string rawImageFile, float *regimg, float fz, float dz, int nz, float fx, float dx, int nx, float fy, float dy,
      int ny, float cdpdx, float cdpdy, float *dest = nullptr, OutputGrid *destGrid = nullptr, int nThreads = 1);

  static void saveImage(string file, float *data, Grid *grid, float *dest = nullptr, OutputGrid *destGrid = nullptr, int nThreads = 1);

  static void whitening(float *illum, int nx, int ny, int nz, float eps, int nThreads);

  static void illum_compenstate(float *img, float *illum, int nx, int ny, int nz, float eps, int nThreads);

protected:

  //
    void stackImageInterp( WaveFieldHolder& srcWFHolder, WaveFieldHolder& recWFHolder, ImageHolder& imageHolder, int isrc, int irec,
        Grid *grid, libfftv::FFTVFilter& fftv, vector<float>& wtaperT, vector<float>& wtaperZ, ImagingCorrelator& correlator,
            vector<bool>& srcTSProcessed, vector<bool>& recTSProcessed);
    void calIllum(float** srcWF, float* amp);
    int  getImagingOption();
    bool doLaplacian();
    void applyFFTV(libfftv::FFTVFilter& fftv, float** wfBuf);
    void calTapes(Grid *grid, vector<float>& wtaperT, vector<float>& wtaperZ);
    void applyTape(float** wfBuf, int it, vector<float>& wtaperT, vector<float>& wtaperZ);


  static void saveSuImage(float fx, int nx, float dx, float fy, int ny, float dy, float fz, int nz, float dz, float *image,
      const char *fileName);

  static void saveDatFrame(const char *filename, float *image, int nz, int nxy, float dz, float z0 = 0);

  void applyDamping(float *illum, float *img, int nx, int ny, int nz);

  void computeDamping(float *illum, int nx, int ny, int nz);

  void interpIllum(float *in, float *out, int nx, int ny, int nz);

    void imageConeMute(Grid *grid, vector<float*> images, int nx, int ny, int nz, int nThreads);
    static float radius_mute(float z, bool isAngleMute, float tangent, float offset_min, vector<vector<float>> &map_mute);

  void imageConeMute(Grid *grid, float *image, int nx, int ny, int nz, int nThreads);

  void applytTaper(float *regimg, int nx, int ny, int nz, int ntaper);
  void resampleSaveImage(string file, float *data, Grid *grid, shared_ptr<Grid> grid2, shared_ptr<Grid> imgGrid, vector<float> &pimg,
      shared_ptr<Lagrange> lagInterp, float *dest, OutputGrid *ogrid, int ntaper, float scaler, bool qcResample = false);

  bool doOutput(string type);
  void initOutputs();

  string getWorkDir() const;
  string getLocalFileName(int idx) const;
    void vectorInfo(float* vec, size_t n, string info);
    void interp(libfftv::FFTVFilter& fftv, float **waveBuf);

    // ***************** for QC  ***********************
    void outputSrcQC(float* srcBuf, int it);
    void outputRecQC(float* recBuf, int it);
    void prepareQCImgs(ImageHolder& imageHolder);
    void outputImgQC(int it);

protected:

  vector<string> outputs;
  vector<float> image, imgFWI, raw, amp;
  vector<float> weight;
  CacheFile *srcWaveField;
  CacheFile *recWaveField;
  size_t compSize; // compressed size
  vector<ushort> vSrcBuf, vRecBuf;
  ushort *compSrcBuf, *compRecBuf;
  ImageType imgType;
  float *vel;
  bool initialDamp;
  int illumComp; // enum Compensation
  int nDblx = 1, nDbly = 1, nDblz = 1;

  int nThreads;
  char hostname[_POSIX_HOST_NAME_MAX];
  int rank = 0;

    int count_tmp = 0;


    // for qc
    bool doQC = false;
    vector<float*> qcImgs;
    vector<string> qcImgNames;
    vector<shared_ptr<oJseis3D> > js_wfs, js_wfs_y;
    vector<shared_ptr<oJseis3D> > js_imgs, js_imgs_y;
    vector<float> buf_xslice;
};

#endif /* IMAINGCONDITION_H_ */

