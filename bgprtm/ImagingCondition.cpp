/*
 * ImagingCondition.cpp
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <limits>
#include <iostream>
#include "libCommon/io_util.h"
#include "ImagingCondition.h"
#include "CacheFile.h"
#include "fdm.hpp"
#include "Grid.h"
#include "GetPar.h"
#include "AsyncIO.h"
#include "WaveFieldCompress.h"
#include "libCommon/Timer.hpp"
using libCommon::time_now;
#include "Laplacian3D.h"
#include "Vector3.h"
#include "Interpolant.h"
#include "volumefilter.h"
#include "libCommon/Utl.hpp"
#include "FdEngine.h"
#include "MpiPrint.h"

using MpiPrint::print1m;
using MpiPrint::printm;
using namespace std;
ImagingCondition::ImagingCondition(CacheFile *srcWaveField, CacheFile *recWaveField, ImageType imgType) : srcWaveField(srcWaveField), recWaveField(
    recWaveField), imgType(imgType) {
  //check source and receiver file
  assert(srcWaveField->seq == SEQ_ZXYT);
  assert(recWaveField->seq == SEQ_ZXYT);
  assert(srcWaveField->nx == recWaveField->nx);
  assert(srcWaveField->ny == recWaveField->ny);
  assert(srcWaveField->nz == recWaveField->nz);
  assert(srcWaveField->nt == recWaveField->nt);

  print1m("ImagingCondition: nx=%d; ny=%d; nz=%d; nt=%d \n", srcWaveField->nx, srcWaveField->ny, srcWaveField->nz, srcWaveField->nt);

  vel = NULL;

  initialDamp = true;
  illumComp = global_pars["ShotIllumComp"].as<int>(ILLUM_NONE);

  nThreads = init_num_threads();

  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  int mpi_is_init;
  MPI_Initialized(&mpi_is_init);
  if(mpi_is_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  compSize = WaveFieldCompress::nshort_volume(srcWaveField->nz, srcWaveField->nx, srcWaveField->ny);
  vSrcBuf.resize(compSize), vRecBuf.resize(compSize);
  compSrcBuf = &vSrcBuf[0], compRecBuf = &vRecBuf[0];

  int default_nDbl = FdEngine::getFdDispersion() > 0.6f ? 2 : 1;
  nDblz = global_pars["ic_double_z"].as<int>(default_nDbl);
  nDblx = srcWaveField->nx > 1 ? global_pars["ic_double_x"].as<int>(default_nDbl) : 1;
  nDbly = srcWaveField->ny > 1 ? global_pars["ic_double_y"].as<int>(default_nDbl) : 1;
  assertion(nDblz == 1 || nDblz == 2, "ic_double_z can only be 1 or 2!");
  assertion(nDblx == 1 || nDblx == 2, "ic_double_x can only be 1 or 2!");
  assertion(nDbly == 1 || nDbly == 2, "ic_double_y can only be 1 or 2!");

  initOutputs();
}

ImagingCondition::~ImagingCondition() {
}

bool ImagingCondition::doOutput(string type) {
  return std::find(outputs.begin(), outputs.end(), type) != outputs.end();
}

void ImagingCondition::initOutputs() {
  if(global_pars["outputs"] && global_pars["outputs"].IsSequence()) outputs = global_pars["outputs"].as<vector<string> >();
  if(!doOutput("AMP") && (global_pars["AMPoutputFile"] || illumComp == ILLUM_GLOBAL)) outputs.push_back("AMP");
  if(!doOutput("RTM") && global_pars["RTMoutputFile"]) outputs.push_back("RTM");
  if(!doOutput("FWI") && global_pars["FWIoutputFile"]) outputs.push_back("FWI");
  if(!doOutput("RAW") && global_pars["RAWoutputFile"]) outputs.push_back("RAW");
}

void ImagingCondition::getVel(float *vel, float dt) {
  this->vel = vel;
//  int nx = srcWaveField->nx;
//  int ny = srcWaveField->ny;
//  int nz = srcWaveField->nz;
//
//  int nxy = nx * ny;
//#pragma omp parallel for schedule(static) num_threads(nThreads)
//  for(int ixy = 0; ixy < nxy; ixy++) {
//    int ix = ixy % nx, iy = ixy / nx;
//    for(int iz = 0; iz < nz; iz++) {
//      size_t id = size_t(iy * nx + ix) * nz + iz;
//      vel[id] = sqrtf(vel[id]) / dt;
//    }
//  }
}
// tag 1
void ImagingCondition::stackImageInterp(WaveFieldHolder &srcWFHolder, WaveFieldHolder &recWFHolder, ImageHolder &imageHolder, int isrc,
    int irec, Grid *grid, libfftv::FFTVFilter &fftv, vector<float> &wtaperT, vector<float> &wtaperZ, ImagingCorrelator &correlator,
    vector<bool> &srcTSProcessed, vector<bool> &recTSProcessed) {

  size_t cubeSize = srcWFHolder.getCubeSize();
  int dbl_size = srcWFHolder.get_dbl_size();

  float **srcWF = srcWFHolder.getWaveField(isrc);
  float **recWF = recWFHolder.getWaveField(irec);

  int ng = imageHolder.getGatherN();
  int ig = isrc - irec + ng / 2;
  float *img = imageHolder.getImgCube(ig);
  float *fwi = imageHolder.getFWICube(ig);
  float *raw = imageHolder.getRawCube(ig);
  float *amp = imageHolder.getAMPCube(0);
  int imageOpt = getImagingOption();

  if(!srcTSProcessed[isrc]) {
    if(doQC) outputSrcQC(srcWF[0], isrc);
    srcTSProcessed[isrc] = true;
    applyTape(srcWF, isrc, wtaperT, wtaperZ);
    applyFFTV(fftv, srcWF);
    if(imageOpt == IMAGE1) interp(fftv, srcWF);
  }

  if(!recTSProcessed[irec]) {
    if(doQC) outputRecQC(recWF[0], irec);
    recTSProcessed[irec] = true;
    applyFFTV(fftv, recWF);
    if(imageOpt == IMAGE1) interp(fftv, recWF);
  }

  if(isrc == irec && amp != nullptr) calIllum(srcWF, amp);

  correlator.run(&fftv, img, fwi, raw, srcWF, recWF, getImagingOption());
  if(doQC && isrc == irec) outputImgQC(isrc);
}

void ImagingCondition::calIllum(float **srcWF, float *amp) {
  size_t nxyz = (size_t)srcWaveField->nx * (size_t)srcWaveField->ny * (size_t)srcWaveField->nz;
  float *srcBuf = srcWF[0];
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(size_t i = 0; i < nxyz; i++)
    amp[i] += srcBuf[i] * srcBuf[i];
}

int ImagingCondition::getImagingOption() {
  string LowcutFilter = expEnvVars(global_pars["LowcutFilter"].as<string>("Inversion"));
  std::transform(LowcutFilter.begin(), LowcutFilter.end(), LowcutFilter.begin(), ::tolower);
  if(LowcutFilter == "laplacian") return IMAGE1;
  else if(LowcutFilter == "inversion") return INVERSION;
  else if(LowcutFilter == "inversion2") return INVERSION2;
  else if(LowcutFilter == "none") return IMAGE1;
  if(global_pars["RAWoutputFile"]) return ALLIN3;
  return IMAGE1;
                                                                              }

bool ImagingCondition::doLaplacian() {
  string LowcutFilter = expEnvVars(global_pars["LowcutFilter"].as<string>("Inversion"));
  std::transform(LowcutFilter.begin(), LowcutFilter.end(), LowcutFilter.begin(), ::tolower);
  if(LowcutFilter == "laplacian") return true;
  else return false;
}

void ImagingCondition::applyFFTV(libfftv::FFTVFilter &fftv, float **wfBuf) {
  float highTapeZ = global_pars["highTapeZ"].as<float>(0.9);
  float highTapeXY = global_pars["highTapeXY"].as<float>(0.7);
  fftv.SetFilterType(libfftv::HIGHTAPE);
  fftv.sethigh(highTapeZ);
  if(nDblz > 1) fftv.run(wfBuf[0], wfBuf[0], NULL, 1); //z
  fftv.sethigh(highTapeXY);
  if(nDblx > 1) fftv.run(wfBuf[0], wfBuf[0], NULL, 2);          //x
  if(nDbly > 1) fftv.run(wfBuf[0], wfBuf[0], NULL, 3);          //y
}

void ImagingCondition::calTapes(Grid *grid, vector<float> &wtaperT, vector<float> &wtaperZ) {
  wtaperT.clear();
  wtaperZ.clear();

  // moved all mute/taper related parameters to Propagator::getIcLiveFull(), except ntaperZ
  int ntaperT = nearbyintf((srcWaveField->t1 - srcWaveField->t0) / srcWaveField->dt);
  if(ntaperT > 0) wtaperT.resize(ntaperT);
  for(int i = 0; i < ntaperT; i++) {
    float temp = (float) M_PI * float(i) / float(ntaperT);
    wtaperT[i] = 0.5f * (1.0f - cosf(temp));
  }
  float waterDepth = global_pars["waterDepth"].as<float>(0.0);
  int ntaperZ = max(0, grid->getIDz(waterDepth));
  if(ntaperZ > 0) wtaperZ.resize(ntaperZ);
  for(int i = 0; i < ntaperZ; i++) {
    float temp = (float) M_PI * float(i) / float(ntaperZ);
    wtaperZ[i] = 0.5f * (1.0f - cosf(temp));
  }
}

void ImagingCondition::applyTape(float **wfBuf, int it, vector<float> &wtaperT, vector<float> &wtaperZ) {
  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  int ntaperT = wtaperT.size();
  int ntaperZ = wtaperZ.size();
  int nxy = nx * ny;
  if(it < ntaperT) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      int ix = ixy % nx, iy = ixy / nx;
      for(int iz = 0; iz < nz; iz++) {
        size_t idd = ((size_t)iy * nx + ix) * nz + iz;
        wfBuf[0][idd] *= wtaperT[it];
      }
    }
  } else {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      int ix = ixy % nx, iy = ixy / nx;
      for(int iz = 0; iz < ntaperZ; iz++) {
        size_t idd = ((size_t)iy * nx + ix) * nz + iz;
        wfBuf[0][idd] *= wtaperZ[iz];
      }
    }
  }
}

shared_ptr<Grid> ImagingCondition::getPartialImageGrid(float xMinValid, float xMaxValid, float yMinValid, float yMaxValid, float zmax,
    OutputGrid *ogrid) {
  if(!getBool(global_pars["ignoreAbsXY"], true)) {
    if(global_pars["absMinX"] || global_pars["absMaxX"] || global_pars["absMinY"] || global_pars["absMaxY"]) {
      if(global_pars["absMinX"] && xMinValid < global_pars["absMinX"].as<float>()) xMinValid = global_pars["absMinX"].as<float>(xMinValid);
      if(global_pars["absMaxX"] && xMaxValid > global_pars["absMaxX"].as<float>()) xMaxValid = global_pars["absMaxX"].as<float>(xMaxValid);
      if(global_pars["absMinY"] && yMinValid < global_pars["absMinY"].as<float>()) yMinValid = global_pars["absMinY"].as<float>(yMinValid);
      if(global_pars["absMaxY"] && yMaxValid > global_pars["absMaxY"].as<float>()) yMaxValid = global_pars["absMaxY"].as<float>(yMaxValid);
      //
      printf("Xmin=%f, Xmax=%f, Ymin=%f, Ymax=%f \n", xMinValid, xMaxValid, yMinValid, yMaxValid);
    }
  }

  int incx = global_pars["incXline"].as<int>(ogrid ? ogrid->incx : 1);
  int incy = global_pars["incInline"].as<int>(ogrid ? ogrid->incy : 1);

  double cdpdx = global_pars["CDPdx"].as<double>();
  double cdpdy = global_pars["CDPdy"].as<double>();

  // read in the RTM output parameters
  double dx = cdpdx * incx;
  double dy = cdpdy * incy;
  double dz = global_pars["RTMdz"].as<double>();

  // global origin
  double outputImageGridXAnchor = global_pars["outputImageGridXAnchor"].as<double>(0.0);
  double outputImageGridYAnchor = global_pars["outputImageGridYAnchor"].as<double>(0.0);
  double outputImageGridZAnchor = global_pars["outputImageGridZAnchor"].as<double>(0.0);

  //
  outputImageGridXAnchor = global_pars["absMinX"].as<double>(outputImageGridXAnchor);
  outputImageGridYAnchor = global_pars["absMinY"].as<double>(outputImageGridYAnchor);
  outputImageGridZAnchor = global_pars["absMinZ"].as<double>(outputImageGridZAnchor);

  int ifx = floor((xMinValid - outputImageGridXAnchor) / dx + 0.8);
  int ify = floor((yMinValid - outputImageGridYAnchor) / dy + 0.8);
  float z0 = 0;
  if(global_pars["RTMz0"]) z0 = global_pars["RTMz0"].as<float>() - global_pars["zMin"].as<float>(0.0f);

  // local origin
  float fx = ifx * dx + outputImageGridXAnchor;
  float fy = ify * dy + outputImageGridYAnchor;
  float fz = z0 + outputImageGridZAnchor;

  int nx = (xMaxValid - fx) / dx + 1.2;
  int ny = (yMaxValid - fy) / dy + 1.2;
  int nz = (zmax - fz) / dz + 1.2;
  nx = max(1, nx);
  ny = getDimension() < 3 ? 1 : max(1, ny);

  if(!(incy == 1 && incx == 1)) { // shift fx/fy to snap with ogrid
    assertion(ogrid, "incInline=%d, incXline=%d --> for non-API RTM, increment of IL/XL>1 might be buggy!", incy, incx);
    int ix0s = (int)nearbyintf(fx / cdpdx) + 1;
    int iy0s = (int)nearbyintf(fy / cdpdy) + 1;
    int ix_shift = ((ogrid->ix0 - ix0s) % incx + incx) % incx; // make sure it's positive
    if(ix_shift) fx += ix_shift * cdpdx;
    int iy_shift = ((ogrid->iy0 - iy0s) % incy + incy) % incy; // make sure it's positive
    if(iy_shift) fy += iy_shift * cdpdy;
  }

  float zmin = fz;
  shared_ptr<Grid> imgGrid(new Grid(RECTANGLE, nx, ny, nz, dx, dy, dz, zmin, zmax, nThreads));
  print1m("imgGrid size required: %d*%d*%d=%ld, imgGrid.zmin=%f\n", nz, nx, ny, imgGrid->mysize, imgGrid->zmin);

  imgGrid->setupRectangle();
  imgGrid->setOrigin(fx, fy);
  imgGrid->setIncXY(incx, incy);
  //imgGrid->print();
  return imgGrid;
}

void ImagingCondition::resampleSaveImage(string file, float *dat0, Grid *grid, shared_ptr<Grid> grid2, shared_ptr<Grid> imgGrid,
    vector<float> &pimg, shared_ptr<Lagrange> lagInterp, float *dest, OutputGrid *ogrid, int ntaper, float scaler, bool qcResample) {
  float *dat = dat0;
  shared_ptr<Grid> gridSave = grid2; // grid2 output, TODO: SeisAPI version

  if(imgGrid) {  // partial image output
    if(qcResample)
      jseisUtil::save_zxy(getJsFilename("qcImgInterp", "0").c_str(), dat0, grid2->nz, grid2->nx, grid2->ny, grid2->dz, grid2->dx, grid2->dy,
                          -grid2->getIDz(0.0), grid2->x0, grid2->y0);
    if(lagInterp) lagInterp->apply(dat0, &pimg[0]);
    else imgGrid->FillVolume(grid2.get(), dat0, &pimg[0]);
    dat = &pimg[0], gridSave = imgGrid;
    if(qcResample)
      jseisUtil::save_zxy(getJsFilename("qcImgInterp", "1").c_str(), &pimg[0], imgGrid->nz, imgGrid->nx, imgGrid->ny, imgGrid->dz,
                          imgGrid->dx, imgGrid->dy, -imgGrid->getIDz(0.0), imgGrid->x0, imgGrid->y0);
  }

  applytTaper(dat, gridSave->nx, gridSave->ny, gridSave->nz, ntaper);
  if(scaler != 1.0f) for(size_t isamp = 0; isamp < gridSave->mysize; isamp++)
    dat[isamp] *= scaler;
  saveImage(file, dat, gridSave.get(), dest, ogrid, nThreads);
}

shared_ptr<Grid> ImagingCondition::genImage(Grid *grid, WaveFieldCompress *wfComp, float xMinValid, float xMaxValid, float yMinValid,
    float yMaxValid, float dtsav, vector<float*> destImages, OutputGrid *ogrid) {

  int resample_b4_stack = global_pars["resample_b4_stack"].as<int>(0); // MPI version default, 1 or 2 may have two different flavors later
  shared_ptr<Grid> grid2(new Grid(grid, nDblz, nDblx, nDbly));

  shared_ptr<Grid> imgGrid; // partial image grid, not ogrid
  shared_ptr<Lagrange> lagInterp;
  vector<float> pimg;

  int ntaper = resample_b4_stack ? 20 : 14; // hmm
  if(resample_b4_stack) {
    imgGrid = getPartialImageGrid(xMinValid, xMaxValid, yMinValid, yMaxValid, grid->zmax, ogrid);
    pimg.resize(imgGrid->mysize);
    string img_interp_option = expEnvVars(global_pars["image_interp"].as<string>("lagrange"));
    if(img_interp_option == "lagrange") {
      int nPts = global_pars["npoints_lagrange_interp"].as<int>(32);
      lagInterp = make_shared<Lagrange>(grid2.get(), imgGrid.get(), nPts, nThreads);
    }

    print1m("ImageI: grid2 nx=%d,  ny=%d,  nz=%d, x0=%f, y0=%f\n", grid2->nx, grid2->ny, grid2->nz, grid2->x0, grid2->y0);
    print1m("ImageO: nx=%d, ny=%d, nz=%d, x0=%f, y0=%f\n", imgGrid->nx, imgGrid->ny, imgGrid->nz, imgGrid->x0, imgGrid->y0);
  }

  int nDestimg = destImages.size();

  // first record gather information
  float maxTimeLag = global_pars["MaxTimeLag"].as<float>(0.0);
  float dt = srcWaveField->dt; // should be same value as Propgator::getDtSave()
  int ng = 2 * (int)(maxTimeLag / dt) + 1;
  float t0 = -ng / 2 * dt;

  // output gather informaiton if necessary
  string gatherFile = getWorkDir() + "/delayTimeInfo.txt";
  if(!libCommon::Utl::haveFile(gatherFile)) {
    OutputGrid g;
    GatherInfo gInfo(g.nz, ng, g.nx, g.ny, g.z0, t0, g.x0, g.y0, g.dz, dt, g.dx, g.dy);
    gInfo.save(gatherFile);
  }

  int nAmp = doOutput("AMP") ? 1 : 0;
  int nImg = doOutput("RTM") ? ng : 0;
  int nFWI = doOutput("FWI") ? ng : 0;
  int nRaw = doOutput("RAW") ? ng : 0;
  int nTotal = (nAmp > 0) + (nImg > 0) + (nFWI > 0) + (nRaw > 0);
  assertion(!nDestimg || nDestimg == nTotal, "nDestimg=%d does not match with nTotalImage=%d (%d>0,%d>0,%d>0,%d>0)", nDestimg, nTotal, nAmp,
            nImg, nFWI, nRaw);

  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  int nt = srcWaveField->nt;
  int dbl_size = nDblz * nDblx * nDbly;
  size_t cubeSize = (size_t)nz * (size_t)nx * (size_t)ny;

  int nx2 = nx * nDblx;
  int nz2 = nz * nDblz;
  int ny2 = ny * nDbly;

  // set fftv
  libfftv::FFTVFilter fftv(nz, nx, ny, grid->dz, grid->dx, grid->dy, nThreads, 1);

  // setup taps
  vector<float> wtaperT, wtaperZ;
  calTapes(grid, wtaperT, wtaperZ);

  ImageHolder imageHolder(cubeSize * dbl_size, nAmp, nImg, nFWI, nRaw);
  WaveFieldHolder srcWFHolder(srcWaveField, wfComp, ng, dbl_size);
  WaveFieldHolder recWFHolder(recWaveField, wfComp, ng, dbl_size);

  // create correlator
  float minFreq = global_pars["minFreq"].as<float>(1.0);
  float kmin = minFreq / global_pars["vMin"].as<float>(1492);
  printf("ImagingCondition: kmin=%f \n", kmin);
  ImagingCorrelator correlator(nx, ny, nz, srcWaveField->dx, srcWaveField->dy, srcWaveField->dz, nx2, ny2, nz2, kmin, nThreads);

  vector<bool> srcWFProcessed(nt, false);
  vector<bool> recTSProcessed(nt, false);

  prepareQCImgs(imageHolder);

  int ng_2 = ng / 2;
  float pt10 = nt / 10.0, pt = 0.0;
  int it_shift = global_pars["it_shift_ic"].as<int>(0);
  for(int isrc = 0; isrc < nt; isrc++) {
    for(int irec = max(0, isrc - ng_2 + it_shift); irec <= min(nt - 1, isrc + ng_2 + it_shift); irec++) {
      stackImageInterp(srcWFHolder, recWFHolder, imageHolder, isrc, irec, grid, fftv, wtaperT, wtaperZ, correlator, srcWFProcessed,
                       recTSProcessed);
    }
    if(isrc / pt10 >= pt)
      printf("[%s] %s > Imaging:  Time step:  %4d %3.1d, max amplitude %e, MemFree: %s\n", hostname, time_now().c_str(), isrc,
             int(10.0 * pt++), imageHolder.getMaxAmp(), libCommon::Utl::free_memory().c_str());
  }
  printf("[%s] %s > Imaging:  100%% done! MemFree: %s\n", hostname, time_now().c_str(), libCommon::Utl::free_memory().c_str());

  for(int i = 0; i < ng; i++) {
    float *image = imageHolder.getImgCube(i);
    float *imgFWI = imageHolder.getFWICube(i);
    float *raw = imageHolder.getRawCube(i);
    float **srcBuf = srcWFHolder.getWaveField(nt - 1);
    float **recBuf = recWFHolder.getWaveField(nt - 1);

    if(this->getImagingOption() == INVERSION2) {
      correlator.costerm(image);
      correlator.sinterm(imgFWI, image);
    }
    correlator.interleave(image, srcBuf, recBuf);
    correlator.interleave(imgFWI, srcBuf, recBuf);
    correlator.interleave(raw, srcBuf, recBuf);
  }

  // release memeory
  srcWFHolder.clear();
  recWFHolder.clear();

  Grid *grid2_tmp = new Grid(grid, nDblz, nDblx, nDbly);
  imageConeMute(grid2_tmp, vector<float*> { &image[0], &imgFWI[0], &raw[0] }, grid2_tmp->nx, grid2_tmp->ny, grid2_tmp->nz, nThreads);
  delete grid2_tmp;

  //TODO: 2. apply damping
  float *amp = imageHolder.getAMPCube(0);
  if(doOutput("AMP") || illumComp) {
    vector<float> illum(cubeSize);
    memcpy((char*)&illum[0], (char*)amp, cubeSize * sizeof(float));
    interpIllum(&illum[0], amp, nx, ny, nz);
  }
  if(illumComp == ILLUM_SHOT) {
    if(initialDamp) {
      computeDamping(amp, nx2, ny2, nz2);
      initialDamp = false;
    }
    for(int i = 0; i < ng; i++) {
      applyDamping(amp, imageHolder.getImgCube(i), nx2, ny2, nz2);
      applyDamping(amp, imageHolder.getFWICube(i), nx2, ny2, nz2);
    }
  }

  for(int i = 0; i < ng; i++) {
    float *image = imageHolder.getImgCube(i);

    //TODO: 3. zscale
    int nxy2 = nx2 * ny2;
    float zscale = global_pars["zscale"].as<float>(0.0f);
    if(zscale > 0.0) {
      printf("%s > applying Z scale %g \n", time_now().c_str(), zscale);
#pragma omp parallel for schedule(static)
      for(int ixy = 0; ixy < nxy2; ixy++) {
        int ix = ixy % nx2, iy = ixy / nx2;
        for(int iz = 0; iz < nz2; iz++) {
          size_t id = ((size_t)iy * nx2 + ix) * nz2 + iz;
          float zl = iz * grid2->dz * 0.001;
          image[id] *= powf(zl, zscale);
        }
      }
    }

    //TODO: 4. low-cut filter
    bool laplacian = this->doLaplacian();
    if(laplacian) laplacian3D(image, grid2.get());

    //TODO: 5 apply velocity
    float vpower = global_pars["vPower"].as<float>(laplacian ? 2.0f : 0.0f);
    if(vpower > 0.1 && laplacian) {
      printf("%s > applying velocity  \n", time_now().c_str());
#pragma omp parallel for num_threads(nThreads) schedule(static)
      for(int ixy = 0; ixy < nxy2; ixy++) {
        int ix = ixy % nx2, iy = ixy / nx2;
        int iy1 = iy / 2, ix1 = ix / 2;
        for(int iz = 0; iz < nz2; iz++) {
          int iz1 = iz / 2;
          size_t id1 = size_t(iy1 * nx + ix1) * nz + iz1;
          size_t id2 = size_t(iy * nx2 + ix) * nz2 + iz;
          image[id2] *= powf(vel[id1], vpower);
        }
      }
    }
  }

  int iDest = 0;
  if(nAmp)
    resampleSaveImage(getLocalFileName(0), imageHolder.getAMPCube(0), grid, grid2, imgGrid, pimg, lagInterp,
                      nDestimg ? destImages[iDest++] : nullptr, ogrid, 0, 1.0f);

  for(int i = 0; i < ng; i++) {
    size_t offset_og = (ogrid == nullptr ? 0 : (size_t)(i + ng / 2) * ogrid->nx * ogrid->ny * ogrid->nz);
    if(nImg)
      resampleSaveImage(getLocalFileName(i + nAmp), imageHolder.getImgCube(i), grid, grid2, imgGrid, pimg, lagInterp,
                        nDestimg ? destImages[iDest++] + offset_og : nullptr, ogrid, ntaper, dtsav, bool(global_pars["qcImgInterp"]));

    if(nFWI)
      resampleSaveImage(getLocalFileName(i + ng / 2 + nAmp + nImg), imageHolder.getFWICube(i), grid, grid2, imgGrid, pimg, lagInterp,
                        nDestimg ? destImages[iDest++] + offset_og : nullptr, ogrid, ntaper, dtsav);
    if(nRaw)
      resampleSaveImage(getLocalFileName(i + ng / 2 + nAmp + nImg + nFWI), imageHolder.getRawCube(i), grid, grid2, imgGrid, pimg, lagInterp,
                        nDestimg ? destImages[iDest++] + offset_og : nullptr, ogrid, ntaper, dtsav);
  }

  return imgGrid ? imgGrid : grid2;
}

void ImagingCondition::applytTaper(float *regimg, int nx, int ny, int nz, int ntaper) {
  if(ntaper <= 0) return;

  vector<float> wtaper(ntaper);
  for(int i = 0; i < ntaper; i++) {
    // for i=0 and i=ntaper-1, the values are not 0 or 1, this avoids losing edge traces on tapering
    float x = float(i + 1) / (ntaper + 1);
    float xx = x * x;
    float xxx = x * xx;
    wtaper[i] = 10.0f * xxx + xx * (-15.0f * xx + 6.0f * xxx);
  }

  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t id1 = size_t(iy * nx + ix) * size_t(nz) + iz;

      if(ix < ntaper) regimg[id1] *= wtaper[ix];
      if(ix > nx - ntaper - 1) regimg[id1] *= wtaper[nx - ix - 1];
      if(ny > 1 && iy < ntaper) regimg[id1] *= wtaper[iy];
      if(ny > 1 && iy > ny - ntaper - 1) regimg[id1] *= wtaper[ny - iy - 1];

    }
  }
}

void ImagingCondition::whitening(float *illum, int nx, int ny, int nz, float eps, int nThreads) {
  if(!illum) return;

  float ampmax = 0.0f;
  int nxy = nx * ny;
#pragma omp parallel for reduction(max : ampmax) num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      ampmax = max(ampmax, fabsf(illum[idd]));
    }
  }

  float amp_eps = ampmax * eps;
#pragma omp parallel for num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      illum[idd] = max(illum[idd], amp_eps);
    }
  }
}

void ImagingCondition::illum_compenstate(float *img, float *illum, int nx, int ny, int nz, float eps, int nThreads) {
  if(!illum) return;

  float ampmax = 0.0f;
  int nxy = nx * ny;
#pragma omp parallel for reduction(max : ampmax) num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      ampmax = max(ampmax, fabsf(illum[idd]));
    }
  }

  float amp_eps = ampmax * eps;
#pragma omp parallel for num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      img[idd] /= max(illum[idd], amp_eps);
    }
  }
}

void ImagingCondition::computeDamping(float *illum, int nx, int ny, int nz) {

  //smooth illum map
  if(getBool(global_pars["smoothSrcAmp"], true)) {
    int smoothSrcAmpXY = global_pars["smoothSrcAmpXY"].as<int>(10);
    int smoothSrcAmpZ = global_pars["smoothSrcAmpZ"].as<int>(10);
    int swx = smoothSrcAmpXY, swy = smoothSrcAmpXY, swz = smoothSrcAmpZ;
    print1m("noise reduction on illumination with %d, %d\n", smoothSrcAmpXY, smoothSrcAmpZ);

    if(ny > 1) avgVolume3D(illum, nx, ny, nz, swx, swy, swz);
    else avgVolume2D(illum, nx, nz, swx, swz);
  }

  //
  double ampSumTotal = 0.0;
  int nxy = nx * ny;
#pragma omp parallel for reduction(+ : ampSumTotal) num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      ampSumTotal += illum[idd];
    }
  }

  //
  unsigned long count = (unsigned long)nx * ny * nz;
  double average = ampSumTotal / count;

  float ampDamping = global_pars["ampDamping"].as<float>(1.0);
  print1m("ampDamping=%g\n", ampDamping);
  average = pow(average, ampDamping);

  // apply damping now
  double oAverage = 1.0 / average;
  print1m("sum=%g average=%g 1/average=%g\n", ampSumTotal, average, oAverage);

  // This dampWeight of 1.0 is fairly aggressive that will favor resolution of subtle features at a cost of additional noise in the section
  //. The average damping of sourceAmpDamp{}  is about 1/10th of the source amplitude sourceAmpSum[] in a constant velocity

  float dampWeight = global_pars["dampWeight"].as<float>(0.07f);
  print1m("dampWeight=%g\n", dampWeight);

  print1m("Finalizing correlation sums using source amplitude.  weight=%g\n", dampWeight);

  //taper
  int ntaper = 10;
  float *wtaper = new float[ntaper];
  for(int i = 0; i < ntaper; i++) {
    float temp = M_PI * float(i) / float(ntaper);
    wtaper[i] = 0.5f * (1.0f - cosf(temp));
  }

  // loop over indexes
#pragma omp parallel for num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;

      float aa = illum[idd] * oAverage;
      float bb = expf(-aa);
      illum[idd] = 1.0 / (average * (aa + dampWeight * bb));

      if(iy < ntaper && ny > 1) illum[idd] *= wtaper[iy];
      if(iy > ny - ntaper - 1 && ny > 1) illum[idd] *= wtaper[ny - iy - 1];
      if(ix < ntaper) illum[idd] *= wtaper[ix];
      if(ix > nx - ntaper - 1) illum[idd] *= wtaper[nx - ix - 1];
    }
  }


  delete[] wtaper;
}

void ImagingCondition::applyDamping(float *illum, float *img, int nx, int ny, int nz) {
  if(!img) return;

  int nxy = nx * ny;
#pragma omp parallel for num_threads(nThreads)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    for(int iz = 0; iz < nz; iz++) {
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      img[idd] *= illum[idd];
    }
  }

}

void ImagingCondition::imageConeMute(Grid *grid, vector<float*> images, int nx, int ny, int nz, int nThreads) {
  for(auto image : images)
    imageConeMute(grid, image, nx, ny, nz, nThreads);
}

float ImagingCondition::radius_mute(float z, bool isAngleMute, float tangent, float offset_min, vector<vector<float> > &map_mute) {
  if(isAngleMute) return z * tangent + offset_min;
  int n = map_mute.size();
  assertion(n > 1 && map_mute[0].size() == 2L,
            "'ic_mute_depth_offset_map' must be vector-of-vector (size>1), with the inner vector being [depth,offset] pair!");
  int i = 0;
  if(z > map_mute[0][0]) for(i = 0; i < n - 1; i++)
    if(map_mute[i + 1][0] >= z) break;
  if(i > n - 2) i = n - 2;
  float z0 = map_mute[i][0], z1 = map_mute[i + 1][0];
  float r = ((z - z0) * map_mute[i + 1][1] + (z1 - z) * map_mute[i][1]) / (z1 - z0);
  // printf("z=%f, r=%f, z0=%f, z1=%f\n", z, r, z0, z1);
  return r;
}

void ImagingCondition::imageConeMute(Grid *grid, float *image, int nx, int ny, int nz, int nThreads) {
  if(!image) return;
  bool is_angle_mute = getBool("ic_conemute_angle", false);
  bool is_map_mute = bool(global_pars["ic_mute_depth_offset_map"]);
  printf("Entering imageConeMute ...\n");
  if(!is_angle_mute && !is_map_mute) return;
  printf("Doing imageConeMute ...\n");

  assertion(bool(global_pars["_sourceX"]), "'sourceX' was not set!");
  float sourceX = global_pars["_sourceX"].as<float>();
  float sourceY = global_pars["_sourceY"].as<float>();
  float sourceZ = global_pars["_sourceZ"].as<float>(0.0f);
  bool qc_weight = bool(global_pars["angleMuteQc"]);

  float tangent = tanf(global_pars["ic_conemute_angle"].as<float>(30.0f) * M_PI / 180.0f);
  float offset_min = global_pars["ic_conemute_offset_min"].as<float>(0.0f);
  vector<vector<float>> mute_map = global_pars["ic_mute_depth_offset_map"].as<vector<vector<float>>>(vector<vector<float>> { });
  size_t cubeSize = (size_t)nx * (size_t)ny * (size_t)nz;
  if(qc_weight) weight.resize(cubeSize);
  float dz = grid->dz;
  int iz0 = grid->getIDz(sourceZ);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iz = 0; iz < nz; iz++) {
    float mz = grid->getmyz(iz) - sourceZ;
    float r1 = radius_mute(mz, is_angle_mute, tangent, offset_min, mute_map);
    float r2 = 2.0f * fabsf(r1);
    for(int iy = 0; iy < ny; iy++) {
      float my = grid->getmyy(iy, iz) - sourceY;
      float yy = my * my;
      for(int ix = 0; ix < nx; ix++) {
        float mx = grid->getmyx(ix, iz) - sourceX;
        float xx = mx * mx;
        float rr = sqrtf(xx + yy);
        float w = 1.0f;
        if(rr > r1) {
          float d1 = rr - r1;
          float d2 = r2 - r1;
          if(r1 <= 0 || d1 > d2) w = 0.0f;
          else w = 0.5f * (1.0f + cosf(M_PI * d1 / d2));
        }

        size_t idd = ((size_t)iy * nx + ix) * nz + iz;
        image[idd] *= w;
        if(qc_weight) weight[idd] = w;
      }
    }
  }
}

void ImagingCondition::interpIllum(float *in, float *out, int nx, int ny, int nz) {
  int nxy = nx * ny;
  if(nDblx * nDbly * nDblz == 1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ixy = 0; ixy < nxy; ixy++) {
      size_t off = (size_t)ixy * nz;
      memcpy(out + off, in + off, sizeof(float) * nz);
    }
    return;
  }

  size_t nx2 = nx * nDblx;
  size_t nz2 = nz * nDblz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int ixy = 0; ixy < nxy; ixy++) {
    int ix = ixy % nx, iy = ixy / nx;
    int iy1 = iy * nDbly;
    int iy2 = iy * nDbly + (nDbly == 1 ? 0 : 1); // this make it work for 2D, though calcs are redundant
    int ix1 = ix * nDblx;
    int ix2 = ix * nDblx + (nDblx == 1 ? 0 : 1); // this make it work for 1D, though calcs are redundant
    for(int iz = 0; iz < nz; iz++) {
      size_t iz1 = iz * nDblz;
      size_t iz2 = iz * nDblz + (nDblz == 1 ? 0 : 1);
      size_t idd = ((size_t)iy * nx + ix) * nz + iz;
      size_t id0 = (iy1 * nx2 + ix1) * ((size_t)nz2) + iz1; //y1x1z1
      size_t id1 = (iy2 * nx2 + ix1) * ((size_t)nz2) + iz1; //y2x1z1
      size_t id2 = (iy1 * nx2 + ix2) * ((size_t)nz2) + iz1; //y1x2z1
      size_t id3 = (iy1 * nx2 + ix1) * ((size_t)nz2) + iz2; //y1x1z2
      size_t id4 = (iy2 * nx2 + ix1) * ((size_t)nz2) + iz2; //y2x1z2
      size_t id5 = (iy1 * nx2 + ix2) * ((size_t)nz2) + iz2; //y1x2z2
      size_t id6 = (iy2 * nx2 + ix2) * ((size_t)nz2) + iz1; //y2x2z1
      size_t id7 = (iy2 * nx2 + ix2) * ((size_t)nz2) + iz2; //y2x2z2
      int iyp1 = MIN(iy + 1, ny - 1);
      int ixp1 = MIN(ix + 1, nx - 1);
      int izp1 = MIN(iz + 1, nz - 1);
      out[id0] = in[idd];
      out[id1] = 0.5f * (in[((size_t)iy * nx + ix) * nz + iz] + in[(iyp1 * nx + ix) * ((size_t)nz) + iz]);
      out[id2] = 0.5f * (in[((size_t)iy * nx + ix) * nz + iz] + in[(iy * nx + ixp1) * ((size_t)nz) + iz]);
      out[id3] = 0.5f * (in[((size_t)iy * nx + ix) * nz + iz] + in[((size_t)iy * nx + ix) * nz + izp1]);
      out[id4] = 0.25f
          * (in[((size_t)iy * nx + ix) * nz + iz] + in[((size_t)iy * nx + ix) * nz + izp1] + in[(iyp1 * nx + ix) * ((size_t)nz) + iz]
              + in[(iyp1 * nx + ix) * ((size_t)nz) + izp1]);
      out[id5] = 0.25f
          * (in[((size_t)iy * nx + ix) * nz + iz] + in[((size_t)iy * nx + ix) * nz + izp1] + in[(iy * nx + ixp1) * ((size_t)nz) + iz]
              + in[(iy * nx + ixp1) * ((size_t)nz) + izp1]);
      out[id6] = 0.25f
          * (in[((size_t)iy * nx + ix) * nz + iz] + in[(iyp1 * nx + ix) * ((size_t)nz) + iz] + in[(iy * nx + ixp1) * ((size_t)nz) + iz]
              + in[(iyp1 * nx + ixp1) * ((size_t)nz) + iz]);
      out[id7] = 0.125f
          * (in[((size_t)iy * nx + ix) * nz + iz] + in[(iyp1 * nx + ix) * ((size_t)nz) + iz] + in[(iy * nx + ixp1) * ((size_t)nz) + iz]
              + in[((size_t)iy * nx + ix) * nz + izp1] + in[(iyp1 * nx + ix) * ((size_t)nz) + izp1]
              + in[(iyp1 * nx + ixp1) * ((size_t)nz) + iz] + in[(iy * nx + ixp1) * ((size_t)nz) + izp1]
              + in[(iyp1 * nx + ixp1) * ((size_t)nz) + izp1]);
    }
  }
}

void ImagingCondition::saveSuImage(float fx, int nx, float dx, float fy, int ny, float dy, float fz, int nz, float dz, float *image,
    const char *fileName) {
  assertion(false, "saveSuImage not implement yet");
}

void ImagingCondition::saveImage(const char *imageFile, float *image, float fz, float dz, int nz, float fx, float dx, int nx, float fy,
    float dy, int ny, float cdpdx, float cdpdy) {

  if(strstr(imageFile, ".fdm") || strstr(imageFile, ".FDM")) {
    int incx = nearbyintf(dx / cdpdx);
    int incy = nearbyintf(dx / cdpdx);
    int ix0 = roundf(fx / cdpdx) + 1;
    int iy0 = roundf(fy / cdpdy) + 1;

    saveFdmCube(image, fx, fy, fz, nx, ny, nz, dx, dy, dz, cdpdx, cdpdy, imageFile);
    //saveFdmCube(image, fx, fy, fz, nx, ny, nz, cdpdx, cdpdy, dz, ix0, iy0, 0, incx, incy, 1, imageFile);
  } else if(strstr(imageFile, ".js") || strstr(imageFile, ".JS")) {
    if(dx == 0) dx = 10.0f;
    if(dy == 0) dy = 10.0f;
    int incx = nearbyintf(dx / cdpdx);
    int incy = nearbyintf(dy / cdpdy);
    int ix0 = roundf(fx / dx) + 1;
    int iy0 = roundf(fy / dy) + 1;
    if(nx == 1 && ny > 1) jseisUtil::save_zxy(imageFile, image, nz, ny, nx, dz, dy, dx, 0, fy, fx, iy0, incy, ix0, incx);
    else jseisUtil::save_zxy(imageFile, image, nz, nx, ny, dz, dx, dy, 0, fx, fy, ix0, incx, iy0, incy);
  } else if(strstr(imageFile, ".dat") || strstr(imageFile, ".DAT")) {
    saveDatFrame(imageFile, image, nz, ny * nx, dz, fz);
  } else {
    saveSuImage(fx, nx, dx, fy, ny, dy, fz, nz, dz, image, imageFile);
  }
}

void ImagingCondition::saveImage(const char *imageFile, float *image, float fz, float dz, int nz, float fx, float dx, int nx, float fy,
    float dy, int ny) {

  if(strstr(imageFile, ".fdm") || strstr(imageFile, ".FDM")) {
    saveFdmCube(image, fx, fy, fz, nx, ny, nz, dx, dy, dz, imageFile);
  } else if(strstr(imageFile, ".js") || strstr(imageFile, ".JS")) {
    if(dx == 0) dx = 10.0f;
    if(dy == 0) dy = 10.0f;
    int incx = 1;
    int incy = 1;
    int ix0 = roundf(fx / dx) + 1;
    int iy0 = roundf(fy / dy) + 1;
    if(nx == 1 && ny > 1) jseisUtil::save_zxy(imageFile, image, nz, ny, nx, dz, dy, dx, 0, fy, fx, iy0, incy, ix0, incx);
    else jseisUtil::save_zxy(imageFile, image, nz, nx, ny, dz, dx, dy, 0, fx, fy, ix0, incx, iy0, incy);
  } else if(strstr(imageFile, ".dat") || strstr(imageFile, ".DAT")) {
    saveDatFrame(imageFile, image, nz, ny * nx, dz, fz);
  } else {
    saveSuImage(fx, nx, dx, fy, ny, dy, fz, nz, dz, image, imageFile);
  }
}

void ImagingCondition::saveImage(string rawImageFile, float *regimg, float fz, float dz, int nz, float fx, float dx, int nx, float fy,
    float dy, int ny) {
  saveImage(rawImageFile.c_str(), regimg, fz, dz, nz, fx, dx, nx, fy, dy, ny);
}

void ImagingCondition::saveImage(string rawImageFile, float *regimg, float fz, float dz, int nz, float fx, float dx, int nx, float fy,
    float dy, int ny, float cdpdx, float cdpdy, float *dest, OutputGrid *g, int nThreads) {
  if(!dest) {
    saveImage(rawImageFile.c_str(), regimg, fz, dz, nz, fx, dx, nx, fy, dy, ny, cdpdx, cdpdy);
    return;
  }

  // resample and output to destGrid
  bool do_stack = getBool("do_stack", true);
  assertion(g->nz == nz, "Output nz mismatch! nzi=%d, nzo = %d", nz, g->nz);
  int incx = nearbyintf(dx / cdpdx), incy = nearbyintf(dy / cdpdy);
  assertion(incx == g->incx, "Input incx=%d does not match output g->xi=%d", incx, g->incx);
  assertion(incy == g->incy, "Input incy=%d does not match output g->yi=%d", incy, g->incy);

  // iy,ix etc are IL/XL numbers
  int ix0s = (int)nearbyintf(fx / cdpdx) + 1;
  int iy0s = (int)nearbyintf(fy / cdpdy) + 1;
  if(global_pars["qc_rtmapi_image"])
    jseisUtil::save_zxy(getJsFilename("qc_rtmapi_image", "_b4stk").c_str(), regimg, nz, nx, ny, dz, dx, dy, 0, fx, fy, ix0s, incx, iy0s,
                        incy);
  int ix1s = ix0s + (nx - 1) * incx;
  int iy1s = iy0s + (ny - 1) * incy;
  int ix0d = g->ix0;
  int iy0d = g->iy0;
  int ix1d = g->ix0 + (g->nx - 1) * g->incx;
  int iy1d = g->iy0 + (g->ny - 1) * g->incy;
  int ix0 = max(ix0s, ix0d);
  int iy0 = max(iy0s, iy0d);
  int ix1 = min(ix1s, ix1d); // inclusive
  int iy1 = min(iy1s, iy1d); // inclusive
  int nypartial = max(0, (iy1 - iy0) / incy + 1), nxpartial = max(0, (ix1 - ix0) / incx + 1);

  int qc_api_rtm = global_pars["qc_api_rtm"].as<int>(0);
  static int printed;
  if(qc_api_rtm || (ny == 1 && nypartial == 0) || !printed) {
    printed = 1;
    print1m("IC output s_ILXL range: [%d,%d],[%d,%d], dest_ILXL: [%d,%d],[%d,%d] => [%d,%d],[%d,%d]\n", iy0s, iy1s, ix0s, ix1s, iy0d, iy1d,
            ix0d, ix1d, iy0, iy1, ix0, ix1);
    print1m("IC: ny=%d, nypartial=%d, nxpartial=%d\n", ny, nypartial, nxpartial);
  }
  assertion((iy0s - iy0d) % incy == 0, "iy0s=%d, iy0d=%d, incy=%d, the input and output grid does not align to each other!", iy0s, iy0d,
            incy);
  assertion((ix0s - ix0d) % incx == 0, "ix0s=%d, ix0d=%d, incx=%d, the input and output grid does not align to each other!", ix0s, ix0d,
            incx);
  assertion(!(nx == 1 && nxpartial == 0),
            "IL output range does not match with data location! Undetected errors in XL direction is likely too!");

  int iy0off = do_stack ? iy0d : iy0;
  int ixoff = do_stack ? (ix0 - ix0d) / incx : 0;
  int nxoff = do_stack ? g->nx : nxpartial; // when do_stack, target is full outgrid than partial
  int nyoff = do_stack ? g->ny : nypartial;
#pragma omp parallel for num_threads(nThreads) if(iy1>iy0)
  for(int il = iy0; il <= iy1; il += incy) {
    float *__restrict in = regimg + ((size_t)(il - iy0s) / incy * nx + (ix0 - ix0s) / incx) * nz;
    float *__restrict out = dest + ((size_t)(il - iy0off) / incy * nxoff + ixoff) * nz;
    for(int xl = ix0; xl <= ix1; xl += incx, in += nz, out += nz) {
      //      if(il == iy0) print1m("xl=%d, in: ix0=%d, ix0s=%d, incx=%d, ix_in=%d; ix_out=%d\n", xl, ix0, ix0s, incx,
      //                           (ix0 - ix0s) / incx + (xl - ix0) / incx, ixoff + (xl - ix0) / incx);
      if(do_stack) for(int iz = 0; iz < nz; iz++)
        out[iz] += in[iz];
      else memcpy(out, in, sizeof(float) * nz);
    }
  }

  if(qc_api_rtm > 1) {
    Util::print_mem_crc(regimg, nz, nx * ny, "regimg");
    Util::print_mem_crc(dest, nz, nxoff * nyoff, "dest");
  }
  if(global_pars["qc_rtmapi_image"])
    jseisUtil::save_zxy(getJsFilename("qc_rtmapi_image", "_aftstk").c_str(), dest, nz, g->nx, g->ny, g->dz, g->dx, g->dy, 0, g->x0, g->y0,
                        ix0d, incx, iy0d, incy);
}

void ImagingCondition::saveImage(string file, float *data, Grid *grid, float *dest, OutputGrid *destGrid, int nThreads) {
  saveImage(file.c_str(), data, grid->z0, grid->dz, grid->nz, grid->x0, grid->dx, grid->nx, grid->y0, grid->dy, grid->ny,
            grid->dx / grid->incx, grid->dy / grid->incy, dest, destGrid, nThreads);
}
void ImagingCondition::saveDatFrame(const char *filename, float *image, int nz, int nxy, float dz, float z0) {
  FILE *fp = fopen(filename, "w");

  fprintf(fp, "#z f(z,ixy)[nxy=%d]\n", nxy);
  for(int iz = 0; iz < nz; iz++) {
    fprintf(fp, "%g", z0 + iz * dz);
    for(int ixy = 0; ixy < nxy; ixy++)
      fprintf(fp, ", %g", image[(size_t)ixy * nz + iz]);
    fprintf(fp, "\n");
  }
  fclose(fp);
}

string ImagingCondition::getWorkDir() const {
  if(!global_pars["_workDir"]) libCommon::Utl::fatal("local yaml file need to set '_workDir'.");

  return global_pars["_workDir"].as<string>();
}

string ImagingCondition::getLocalFileName(int idx) const {
  return getWorkDir() + "/shot_" + std::to_string(idx) + "_" + std::to_string(global_pars["sourceID"].as<int>(0)) + ".fdm";
}

void ImagingCondition::vectorInfo(float *vec, size_t n, string info) {
  double sum = 0;
  for(size_t i = 0; i < n; i++)
    sum += vec[i];

  std::cout << std::endl << "----------------" << info << "----------------------- " << std::endl;
  std::cout << "sum:  " << sum << std::endl;
  std::cout << n / 2 << ":  " << vec[n / 2] << endl;
}

void ImagingCondition::prepareQCImgs(ImageHolder &imageHolder) {
  doQC = global_pars["qcImagingPrefix"] ? true : false;
  if(!doQC) return;

  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  int nt = srcWaveField->nt;
  int ng = imageHolder.getGatherN();

  if(doOutput("RTM")) {
    qcImgs.push_back(imageHolder.getImgCube(ng / 2));
    qcImgNames.push_back("rtm");
  }

  if(doOutput("FWI")) {
    qcImgs.push_back(imageHolder.getFWICube(ng / 2));
    qcImgNames.push_back("fwi");
  }

  if(doOutput("RAW")) {
    qcImgs.push_back(imageHolder.getRawCube(ng / 2));
    qcImgNames.push_back("raw");
  }

  int nImg = qcImgs.size();
  js_wfs.resize(2);
  js_wfs_y.resize(2);
  js_imgs.resize(nImg);
  js_imgs_y.resize(nImg);
  string qcImagingPrefix;
  vector<string> qcImgNames_y(nImg);

  qcImagingPrefix = expEnvVars(global_pars["qcImagingPrefix"].as<string>());
  for(int i = 0; i < nImg; i++) {
    qcImgNames_y[i] = qcImagingPrefix + "_" + qcImgNames[i] + "_y.js";
    qcImgNames[i] = qcImagingPrefix + "_" + qcImgNames[i] + ".js";
  }

  vector<string> qcWfNames { qcImagingPrefix + "_wf_fwd.js", qcImagingPrefix + "_wf_bwd.js" };
  vector<string> qcWfNames_y { qcImagingPrefix + "_wf_fwd_y.js", qcImagingPrefix + "_wf_bwd_y.js" };
  for(int i = 0; i < 2; i++)
    js_wfs[i].reset(
        new oJseis3D(qcWfNames[i].c_str(), nz, nx, nt, srcWaveField->dz, srcWaveField->dx, srcWaveField->dt * 1000, 0, srcWaveField->x0,
                     0));
  for(int i = 0; i < nImg; i++)
    js_imgs[i].reset(
        new oJseis3D(qcImgNames[i].c_str(), nz, nx, nt, srcWaveField->dz, srcWaveField->dx, srcWaveField->dt * 1000, 0, srcWaveField->x0,
                     0));

  if(ny > 1) {
    for(int i = 0; i < 2; i++)
      js_wfs_y[i].reset(
          new oJseis3D(qcWfNames_y[i].c_str(), nz, ny, nt, srcWaveField->dz, srcWaveField->dy, srcWaveField->dt * 1000, 0, srcWaveField->y0,
                       0));
    for(int i = 0; i < nImg; i++)
      js_imgs_y[i].reset(
          new oJseis3D(qcImgNames_y[i].c_str(), nz, nx, nt, srcWaveField->dz, srcWaveField->dx, srcWaveField->dt * 1000, 0,
                       srcWaveField->y0, 0));
    buf_xslice.resize(nz * ny); // aliased version, without double imaging grid
  }
}

void ImagingCondition::outputSrcQC(float *srcBuf, int it) {
  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  js_wfs[0]->write_frame(srcBuf + (size_t)nz * nx * (ny / 2), it); // slice iy=ny/2
  if(ny > 1) {
    for(int iy = 0; iy < ny; iy++)
      memcpy(&buf_xslice[iy * nz], srcBuf + ((size_t)iy * nx + nx / 2) * nz, sizeof(float) * nz);
    js_wfs_y[0]->write_frame(&buf_xslice[0], it);
  }

}
void ImagingCondition::outputRecQC(float *recBuf, int it) {
  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  js_wfs[1]->write_frame(recBuf + (size_t)nz * nx * (ny / 2), it); // slice iy=ny/2
  if(ny > 1) {
    for(int iy = 0; iy < ny; iy++)
      memcpy(&buf_xslice[iy * nz], recBuf + ((size_t)iy * nx + nx / 2) * nz, sizeof(float) * nz);
    js_wfs_y[1]->write_frame(&buf_xslice[0], it);
  }
}

void ImagingCondition::outputImgQC(int it) {
  int nx = srcWaveField->nx;
  int ny = srcWaveField->ny;
  int nz = srcWaveField->nz;
  int nImg = qcImgs.size();
  for(int i = 0; i < nImg; i++) {
    js_imgs[i]->write_frame(qcImgs[i] + (size_t)nz * nx * (ny / 2), it); // slice iy=ny/2
    if(ny > 1) {
      for(int iy = 0; iy < ny; iy++)
        memcpy(&buf_xslice[iy * nz], qcImgs[i] + ((size_t)iy * nx + nx / 2) * nz, sizeof(float) * nz);
      js_imgs_y[i]->write_frame(&buf_xslice[0], it);
    }
  }
}

void ImagingCondition::interp(libfftv::FFTVFilter &fftv, float **waveBuf) {

  int ivz = 1, ivx = 2, ivy = 3, ivzx = 4, ivzy = 5, ivxy = 6, ivzxy = 7;

  if(nDbly == 1) {
    if(nDblz == 1) ivx = 1;
    else ivz = 1, ivx = 2, ivzx = 3;
  } else if(nDblx == 1) {
    if(nDblz == 1) ivy = 1; // only interp-y needed
    else ivz = 1, ivy = 2, ivzy = 3;
  } else if(nDblz == 1) ivx = 1, ivy = 2, ivxy = 3;

  // lines below: not used, set to 0
  if(nDbly == 1) ivy = ivxy = ivzy = ivzxy = 0;
  if(nDblx == 1) ivx = ivxy = ivzx = ivzxy = 0;
  if(nDblz == 1) ivz = ivzx = ivzy = ivzxy = 0;

  fftv.SetFilterType(libfftv::SHIFTHALF);
  if(ivz) fftv.run(waveBuf[0], waveBuf[ivz], NULL, 1); //z
  if(ivx) fftv.run(waveBuf[0], waveBuf[ivx], NULL, 2);  //x
  if(ivy) fftv.run(waveBuf[0], waveBuf[ivy], NULL, 3);  //y
  if(ivzx) fftv.run(waveBuf[ivz], waveBuf[ivzx], NULL, 2);  //zx
  if(ivzy) fftv.run(waveBuf[ivz], waveBuf[ivzy], NULL, 3);  //zy
  if(ivxy) fftv.run(waveBuf[ivx], waveBuf[ivxy], NULL, 3); //x-y
  if(ivzxy) fftv.run(waveBuf[ivzx], waveBuf[ivzxy], NULL, 3); //zx-y
}

