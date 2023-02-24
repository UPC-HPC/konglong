#include <xmmintrin.h>
#include <string.h>

#include "GetPar.h"
#include "AsyncIO.h"
#include "WaveFieldCompress.h"
#include "libCommon/Timer.hpp"
using libCommon::time_now;
#include "Util.h"
#include "Wavefield.h"
#include "Profile.h"
#include "ModelLoader.h"
#include "ModelPrepare.h"
#include "ModelRegrid.h"
#include "Q.h"
#include "Params.h"
#include "FreqUtil.h"
#include "Derivative_fftv.h"
#include "Derivative_cfd.h"
#include "Derivative_sincos.h"
#include "FdEngine_fd.h"
#include "FdEngine_cfd.h"
#include "Geometry.h"
#include "PhaseCorrection.h"
#include "Propagator.h"
#include "Source.h"
#include "Traces.h"
#include "Wavelet.h"
#include "KernelCPU.h"
#include "Receiver.h"
#include "ExtractReceiverData.h"
#include "Boundary.h"
#include "FdEngine.h"
#include "libSWIO/RecordIO.hpp"
#include "DomainRange.h"
#include "RecordUtilities.h"
#include "CacheFile.h"
#include "ImagingCondition.h"
#include "timing.h"
#include "stdio.h"
#include "MpiPrint.h"
#include "gpurtm.h"
using MpiPrint::print1m;

Propagator::Propagator(float zmin, float zmax, float maxFreq, float tmax) : zmin(zmin), zmax(zmax), maxFreq(maxFreq), tmax(tmax) {
  nxbnd = 0;
  nybnd = 0;
  nzbnd = 0;
  dxmax = 0;
  dymax = 0;
  dzmin = 0;
  drangex = 0;
  drangey = 0;
  volModel = NULL;
  wfComp = NULL;
  minFreq = maxFreq / 20.0f;
  myshot = NULL;
  receivers[0] = NULL, receivers[1] = NULL;
  bnd = NULL;
  bndDemig = NULL;
  derive = NULL;
  deriveDemig = NULL;
  kernelCPU = NULL;
  kernelDemig = NULL;
  engineType = FD::FFTV; // defualt is FFTV
  allocWx = true;
  sfBoundType[0] = Source::getSurfaceType(PROP::FORWARD);
  sfBoundType[1] = Source::getSurfaceType(PROP::BACKWARD);

  if(global_pars["minFreq"]) minFreq = global_pars["minFreq"].as<float>(minFreq);
  else print1m("Assuming minimum frequency of %f .\n", minFreq);

  nThreads = init_num_threads(true);

  dim = ModelPrepare::getDimension();

  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  posLogical = global_pars["_posLogical"].as<vector<int>>(vector<int> { 0, 0, 0 });
}

void Propagator::prepare(float xMinValid, float xMaxValid, float yMinValid, float yMaxValid, vector3 recvMin, vector3 recvMax) {
  this->recvMin = recvMin;
  this->recvMax = recvMax;

  //get the grid type from geometry file
  string geomFile = expEnvVars(global_pars["local"]["geometry"]["file"].as<string>(GEOM_LOCAL_DEFAULT));
  Geometry *geom = new Geometry();
  geom->read(geomFile);
  GeomHeader *gHeader = geom->getHeader();

  int gtype = gHeader->gridType;

  //print1m("gridType=%d \n", gtype);
  nThreads = init_num_threads();

  myModel = new Model();
  modelLoader = new ModelLoader(*myModel, "local", nThreads);

  // nvmodel is only for print the memory usage
  if(myModel->modeltype == ISO) {
    nvmodel = 1;
  }
  if(myModel->modeltype == VTI) {
    nvmodel = 3;
  }
  if(myModel->modeltype == TTI) {
    nvmodel = 5;
  }
  if(myModel->modeltype == ORT) {
    nvmodel = 5;
  }
  if(myModel->modeltype == TOR) {
    nvmodel = 8;
  }      // ignore del3 in this case

  nvmodel++; // needed for density even if not using it
  nvmodel += bool(myModel->useQ);
  nvmodel += bool(myModel->useReflectivity);

  nxbnd = gHeader->nxbnd; // default value unified in ModelRegrid
  nybnd = gHeader->nybnd;

  if(dim & OneD) nxbnd = 0;
  if(!(dim & ThreeD)) nybnd = 0;


  float gdx = gHeader->dx;
  float gdy = gHeader->dy;
  float gx0 = gHeader->x0;
  float gy0 = gHeader->y0;

  apertx = (dim & OneD) ? 0 : (xMaxValid - xMinValid);
  aperty = (dim & ThreeD) ? (yMaxValid - yMinValid) : 0;

  int nxt = (int)nearbyintf(apertx / gdx + 1);
  int nxu = (nxt == 1) ? 1 : libCommon::padfft(nxt);
  nxbnd = max(nxbnd, (nxu - nxt) / 2);

  int nyt = (int)nearbyintf(aperty / gdy + 1);
  int nyu = (nyt == 1) ? 1 : libCommon::padfft(nyt);
  nybnd = max(nybnd, (nyu - nyt) / 2);
  print1m("The XY boundary are  %d  %d\n", nxbnd, nybnd);
  nzbnd = gHeader->nzbnd;

  apertx = (nxu - 1) * gdx;
  aperty = (nyu - 1) * gdy;
  float midx = 0.5f * (xMinValid + xMaxValid);
  float midy = 0.5f * (yMinValid + yMaxValid);

  xMinValid = midx - 0.5f * apertx;
  yMinValid = midy - 0.5f * aperty;

  int ifx = roundf((xMinValid - gx0) / gdx) + 1;
  int ify = roundf((yMinValid - gy0) / gdy) + 1;

  if(ifx < 1 || ify < 1) {
    print1m("Warning: the aperture is out of model range! ifx=%d ify=%d \n", ifx, ify);
  }

  xMinValid = (ifx - 1) * gdx + gx0;
  xMaxValid = xMinValid + (nxu - 1) * gdx;

  if(aperty != 0) {
    yMinValid = (ify - 1) * gdy + gy0;
    yMaxValid = yMinValid + (nyu - 1) * gdy;
  }

  print1m("Round to computation grid, the compute region is Xmin=%f, Xmax=%f, Ymin=%f, Ymax=%f \n", xMinValid, xMaxValid, yMinValid,
          yMaxValid);

  int nzu = gHeader->nz;
  nzuppad = gHeader->nzuppad;
  float gdz = gHeader->dz;

  float *zgrid = NULL;
  float *dzgrid = NULL;
  if(gtype != RECTANGLE) {
    zgrid = geom->getZGrid();
    dzgrid = geom->getDzGrid();
  }

  myGrid = make_shared<Grid>(gtype, nxu, nyu, nzu, gdx, gdy, gdz, zmin, zmax, nThreads);

  myGrid->setupGrid(zgrid, dzgrid, nzbnd, nzuppad);

  myGrid->setOrigin(xMinValid, yMinValid);

  myGrid->setupJacob();

  delete geom;

  // to test whether allocate more than necessary memory for the volModel?
  allocvolume();

  //TODO:load model
  vmax = 0.0f;
  modelLoader->loadLocalModels(volModel, xMinValid, yMinValid, nxu, nyu, vmax);
  print1m("vmax=%f \n", vmax);

  if(global_pars["pml_vel_max"]) {
    vmax = global_pars["pml_vel_max"].as<float>();
    print1m("vmax for pml overridden by user: vmax=%f \n", vmax);
 }

  //get the vsurface here
  zsurf = -global_pars["zMin"].as<float>(0.0f);
  addVelSurface();

  ModelBoundaryTaper();

  // need get dt here
  dtpro = calcDeltaTime();
  Q::populateCoeff();

  nt = (int)(tmax / dtpro);
  it0 = 0;

  //(v*dt)^2
  Vel2(dtpro);

  //
  myBndTaper = new BndTaper(myGrid.get(), nxbnd, nybnd, nzbnd, 0.90);

  wfComp = new WaveFieldCompress(nThreads);

  myWavefield = new Wavefield(myGrid.get(), myModel);

  if(myModel->useReflectivity) {
    myDemigWavefield = new Wavefield(myGrid.get(), myModel);
  }

  string engineName = global_pars["Engine"].as<string>("FFTV");
  transform(engineName.begin(), engineName.end(), engineName.begin(), ::toupper);

  //Neptune March 2018

//  if(myModel->useRho && myModel->RhoCN* == 1) {
//    rhoCanonical();
//    model_bnd_smooth(RHO, nxbnd, nybnd, nzbnd);
//  }//removed by wolf

  if(myModel->useReflectivity) {
    //print1m("nx = %i, ny = %i, nz = %i, nxbnd = %i, nybnd = %i, nzbnd = %i\n", myGrid->nx, myGrid->ny, myGrid->nz, nxbnd, nybnd, nzbnd);
    model_bnd_smooth(REFLECTIVITY, nxbnd, nybnd, nzbnd);
  }

  engineType = FD::FFTWR2C;
  if(engineName.compare("FFTV") == 0) {
    engineType = FD::FFTV;
  } else if(engineName.compare("FFTWR2C") == 0) {
    engineType = FD::FFTWR2C;
  } else if(engineName.compare("FFTWR2R") == 0) {
    engineType = FD::FFTWR2R;
//  } else if(engineName.compare("HFD") == 0) {
//    engineType = FD::HFD;
  } else if(engineName.compare("CFD") == 0) {
    engineType = FD::CFD;
  } else if(engineName.compare("CFD_BATCH") == 0) {
    engineType = FD::CFD_BATCH;
  } else if(engineName.compare("SINCOS") == 0) {
    engineType = FD::SINCOS;
  } else if(engineName.rfind("FD", 0) == 0) {
    engineType = FD::FD;
  } else {
   print1m("Error: Unknown engine type! engineType=%s \n", engineName.c_str());
    exit(-1);
  }
  print1m("engineType=%d (%s) \n", engineType, FD::EngineNames[engineType].c_str());

  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  if((engineType == FD::FFTV) || (engineType == FD::CFD_BATCH) || (engineType == FD::SINCOS)) {
    allocWx = true;
    bnd = new Boundary(1, nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, dtpro, vmax, nThreads);
    bnd->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd);
    if(myModel->useReflectivity) {
      bndDemig = new Boundary(1, nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, dtpro, vmax, nThreads);
      bndDemig->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd);
    }

    if(engineType == FD::FFTV) {
      derive = new Derivative_fftv(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) {
        deriveDemig = new Derivative_fftv(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
      }
    } else if(engineType == FD::CFD_BATCH) {
      derive = new Derivative_cfd(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) {
        deriveDemig = new Derivative_cfd(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
      }
    } else if(engineType == FD::SINCOS) {
      derive = new Derivative_sincos(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) deriveDemig = new Derivative_sincos(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
    }
  } else {
    allocWx = false;
    kernelCPU = new KernelCPU(nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, engineType, nThreads); // removed RhoCN* by wolf
    kernelCPU->setJacob(myGrid.get(), &(myGrid->jacobx)[0], &(myGrid->jacoby)[0], &(myGrid->jacobz)[0]);
    kernelCPU->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd, dtpro, vmax);
    kernelCPU->setModel(volModel[VEL], volModel[RHO], volModel[DEL], volModel[EPS], volModel[PJX], volModel[PJY]);

    if(myModel->useReflectivity) {
      kernelDemig = new KernelCPU(nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, engineType, nThreads); // removed RhoCN* by wolf
      kernelDemig->setJacob(myGrid.get(), &(myGrid->jacobx)[0], &(myGrid->jacoby)[0], &(myGrid->jacobz)[0]);
      kernelDemig->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd, dtpro, vmax);
      kernelDemig->setModel(volModel[VEL], volModel[RHO], volModel[DEL], volModel[EPS], volModel[PJX], volModel[PJY]);
    }
  }

  // count_prop_memomry();
}

//<<<<<<< HEAD
void Propagator::reprepare() {
  if(!getBool(global_pars["dual_flood"], false)) return;
  vmax = 0.0f;
  modelLoader->reloadLocalModels(volModel, myGrid->x0, myGrid->y0, myGrid->nx, myGrid->ny, vmax, VEL2);
  print1m("vmax=%f \n", vmax);

  if(global_pars["pml_vel_max"]) {
    vmax = global_pars["pml_vel_max"].as<float>();
    print1m("vmax for pml overridden by user: vmax=%f \n", vmax);
  }

  addVelSurface();
Vel2(dtpro);

  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  if((engineType == FD::FFTV) || (engineType == FD::CFD_BATCH) || (engineType == FD::SINCOS)) {
    allocWx = true;
    if(bnd) {
      delete bnd;
      bnd = NULL;
    }
    if(derive) {
      delete derive;
      derive = NULL;
    }
    if(deriveDemig) {
      delete deriveDemig;
      deriveDemig = NULL;
    }
    if(bndDemig) {
      delete bndDemig;
      bndDemig = NULL;
    }
    bnd = new Boundary(1, nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, dtpro, vmax, nThreads);
    bnd->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd);
    if(myModel->useReflectivity) {
      bndDemig = new Boundary(1, nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, dtpro, vmax, nThreads);
      bndDemig->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd);
    }

    if(engineType == FD::FFTV) {
      derive = new Derivative_fftv(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) {
deriveDemig = new Derivative_fftv(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
      }
    } else if(engineType == FD::CFD_BATCH) {
      derive = new Derivative_cfd(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) {
        deriveDemig = new Derivative_cfd(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
      }
    } else if(engineType == FD::SINCOS) {
      derive = new Derivative_sincos(myGrid.get(), bnd, myModel->modeltype, nThreads);
      if(myModel->useReflectivity) deriveDemig = new Derivative_sincos(myGrid.get(), bndDemig, myModel->modeltype, nThreads);
    }
  } else {
    allocWx = false;
  if (kernelDemig) delete kernelDemig, kernelDemig = NULL;
    if (kernelCPU)  delete kernelCPU, kernelCPU = NULL;
    kernelCPU = new KernelCPU(nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, engineType, nThreads);// removed RhoCN* by wolf
    kernelCPU->setJacob(myGrid.get(), &(myGrid->jacobx)[0], &(myGrid->jacoby)[0], &(myGrid->jacobz)[0]);
    kernelCPU->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd, dtpro, vmax);
    kernelCPU->setModel(volModel[VEL], volModel[RHO], volModel[DEL], volModel[EPS], volModel[PJX], volModel[PJY]);

    if(myModel->useReflectivity) {
      kernelDemig = new KernelCPU(nx, ny, nz, myGrid->dx, myGrid->dy, myGrid->dz, engineType, nThreads);// removed RhoCN* by wolf
      kernelDemig->setJacob(myGrid.get(), &(myGrid->jacobx)[0], &(myGrid->jacoby)[0], &(myGrid->jacobz)[0]);
      kernelDemig->setBoundary(nxbnd, nxbnd, nybnd, nybnd, nzbnd + nzuppad, nzbnd, dtpro, vmax);
      kernelDemig->setModel(volModel[VEL], volModel[RHO], volModel[DEL], volModel[EPS], volModel[PJX], volModel[PJY]);
    }
  }
}

void Propagator::ic_prepare() {
//  if(!getBool(global_pars["dual_flood"], false)) return;
  vmax = 0.0f;
  modelLoader->reloadLocalModels(volModel, myGrid->x0, myGrid->y0, myGrid->nx, myGrid->ny, vmax, VEL);
  //always reload velocity, so no need to calculate the velocity in ic.getVel().
  print1m("vmax=%f \n", vmax);
}


void Propagator::count_prop_memomry() { // this function is no longer accurate, estimated models/wavefields seperately elsewhere

  int nvolumes = nvmodel;
  if(allocWx) nvolumes += 7;
  else nvolumes += 4;
  float total_memmory = (float)(gridsize * nvolumes * 4) / 1024.0 / 1024.0 / 1024;

  print1m("Estimated memory usage for propagation: %fGB, MemFree: %s\n", total_memmory, libCommon::Utl::free_memory().c_str());

}

void Propagator::freeVolume(int flag) {

  if(flag == 0) {   //keep vel for rtm
    if(volModel) {
      for(int i = 0; i < SIZE_ModelVolID; i++)
        if(volModel[i]) {
          _mm_free(volModel[i]);
          volModel[i] = NULL;
        }
      delete[] volModel, volModel = NULL;
    }
  } else {
    for(int i = 1; i < SIZE_ModelVolID; i++)
      if(volModel[i]) {
        _mm_free(volModel[i]);
        volModel[i] = NULL;
      }
  }

  count_mem3d = 0;
}
Propagator::~Propagator() {
  freeVolume(0);
  myWavefield->deallocateMemory();

  if(myModel->useReflectivity) {
    myDemigWavefield->deallocateMemory();
    delete kernelDemig, kernelDemig = NULL;
    if(deriveDemig) {
      delete deriveDemig;
      deriveDemig = NULL;
    }
    if(bndDemig) {
      delete bndDemig;
      bndDemig = NULL;
    }
    delete myDemigWavefield;
    myDemigWavefield = 0;
  }

  if(myBndTaper) {
    delete myBndTaper;
    myBndTaper = NULL;
  }
  if(wfComp) {
    delete wfComp;
    wfComp = NULL;
  }
  if(derive) {
    delete derive;
    derive = NULL;
  }
 if(bnd) {
    delete bnd;
    bnd = NULL;
  }

  delete myWavefield;
  myWavefield = 0;

  delete myModel;
  myModel = NULL;
  if(modelLoader) {
    delete modelLoader;
    modelLoader = NULL;
  }

  delete myshot;
  myshot = NULL;

  delete kernelCPU, kernelCPU = NULL;

}
float* Propagator::allocMem3d(int nx, int ny, int nz, int nThreads) {
  size_t gridsize = (size_t)nx * ny * nz;
  size_t nxz = (size_t)nx * nz;
  float *waveField = (float*)_mm_malloc(gridsize * sizeof(float) + 128, 16);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++)
    memset(&waveField[iy * nxz], 0, nxz * sizeof(float));

  count_mem3d++;
  return waveField;
}

void Propagator::allocModels() {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  volModel = new float*[SIZE_ModelVolID]();

  volModel[VEL] = allocMem3d(nx, ny, nz, nThreads);

  string g = "global", l = "local";
  if(global_pars[g]["rho"] || global_pars[l]["rho"]) volModel[RHO] = allocMem3d(nx, ny, nz, nThreads);
  if(global_pars[g]["reflectivity"] || global_pars[l]["reflectivity"]) volModel[REFLECTIVITY] = allocMem3d(nx, ny, nz, nThreads);
  if(global_pars[g]["1/Q"] || global_pars[l]["1/Q"]) volModel[Q] = allocMem3d(nx, ny, nz, nThreads);
  if(global_pars[g]["epsilon"] || global_pars[l]["epsilon"]) volModel[EPS] = allocMem3d(nx, ny, nz, nThreads);
  if(global_pars[g]["delta"] || global_pars[l]["delta"]) volModel[DEL] = allocMem3d(nx, ny, nz, nThreads);
  if(FdEngine::determineModelType() >= TTI) {
    volModel[PJX] = allocMem3d(nx, ny, nz, nThreads);
    volModel[PJY] = allocMem3d(nx, ny, nz, nThreads);
  }
}
void Propagator::allocvolume() {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  gridsize = myGrid->mysize;
  nxz = nx * nz;

  print1m("The Model volume = %d, Gridsize = %ld\n", nvmodel, gridsize);
  string mem_free0 = libCommon::Utl::free_memory();
  allocModels();
  print1m("Allocated %7.2fGB for models: nvol=%d [%dx%dx%d], MemFree: %s->%s\n",
          float(count_mem3d) * nxz * ny * sizeof(float) / 1024 / 1024 / 1024, count_mem3d, nz, nx, ny, mem_free0.c_str(),
          libCommon::Utl::free_memory().c_str());

}
void Propagator::addVelSurface() {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  print1m("scan model for surface velocity... \n");
  float *myVel = volModel[VEL];
  float vsurface = 0.0f;
  int iz0 = myGrid->getIDz(zsurf);
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      size_t id = (size_t)(iy * nx + ix) * (size_t)nz + iz0;
      vsurface += myVel[id];
    }
  }
  vsurface /= float(nx * ny);
  printf("scanned vsurface=%f \n", vsurface);

  if(!global_pars["vsurf"]) {
    global_pars["vsurf"] = vsurface;
  } else {
    vsurface = global_pars["vsurf"].as<float>();
//    print1m("warning: surface velocity was overridden by user input=%f \n", vsurface);
  }
 print1m("scan model for vmin ... \n");
  float vmin = 99999999.0f;
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      for(int iz = 0; iz < nz; iz++) {
        size_t id = (size_t)(iy * nx + ix) * (size_t)nz + iz;
        if(vmin > myVel[id]) vmin = myVel[id];
      }
    }
  }
  printf("scanned vmin=%f \n", vmin);

//  if(!global_pars["vMin"]) {
//    global_pars["vMin"] = vmin;
//  } else {
//    vmin = global_pars["vMin"].as<float>();
//    print1m("warning: vmin was overridden by user input=%f \n", vmin);
//  }// set global_pars["vMin"] in ModelRegrid::preModel by wolf

}
void Propagator::populateModelDARemval() {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;
  int iz0 = myGrid->getIDz(zsurf);

  copyValue(volModel[VEL], iz0);

  if((myModel->modeltype == VTI) || (myModel->modeltype == TTI)) {
    copyValue(volModel[EPS], iz0);
    copyValue(volModel[DEL], iz0);
  }
  if(myModel->modeltype == TTI) {
    copyValue(volModel[PJX], iz0);
    copyValue(volModel[PJY], iz0);
  }

  if(myModel->useRho) {  // myModel->RhoCN* == 0 is handled too
    copyValue(volModel[RHO], iz0);
  }
}
void Propagator::copyValue(float *vol, int iz0) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;
#pragma omp parallel for
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    size_t offset = ixy * nz;
    float v = vol[offset + iz0];
    for(int iz = 0; iz < nz; iz++)
      vol[offset + iz] = v;
  }
}

void Propagator::fillVolume(float *vol, float v) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;
#pragma omp parallel for
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    size_t offset = ixy * nz;
    for(int iz = 0; iz < nz; iz++)
      vol[offset + iz] = v;
  }
}
void Propagator::ModelBoundaryTaper(int thickness) {

  if((myModel->modeltype == VTI) || (myModel->modeltype == TTI)) {
    TaperVolume(volModel[EPS], thickness);
    TaperVolume(volModel[DEL], thickness);
  }

  /*
   if (myModel->modeltype == TTI) {
   TaperVolume(volModel[PJX], thickness);
   TaperVolume(volModel[PJY], thickness);
   }
   */
}
void Propagator::TaperVolume(float *vol, int thickness) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float tapsize = 1.0 / float(nxbnd + thickness);

  if(ny > 1) {
    int iy1 = 0;
    int iy2 = min(ny, (nybnd + thickness));

#pragma omp parallel num_threads(nThreads)
    {
#pragma omp for schedule(static)
      for(int iy = iy1; iy < iy2; iy++) {
        int iiy = iy - iy1;
        float weight = 0.5f * (1.0 - cos(PI * iiy * tapsize));
        size_t id = iy * nx;
        id *= nz;
        for(int ix = 0; ix < nx; ix++) {
          for(int iz = 0; iz < nz; iz++) {
            int idxz = ix * nz + iz;
            vol[id + idxz] *= weight;
          }
        }
      }
     iy1 = max(0, (ny - nybnd - thickness));
      iy2 = max(0, (ny));
#pragma omp for schedule(static)
      for(int iy = iy1; iy < iy2; iy++) {
        int iiy = iy - iy1;
        float weight = 0.5f * (1.0 + cos(PI * iiy * tapsize));
        for(int ix = 0; ix < nx; ix++) {
          for(int iz = 0; iz < nz; iz++) {
            int id = (iy * nx + ix) * nz + iz;
            vol[id] *= weight;
          }
        }
      }
    }

    if(nx > 1) {
      // Tapper the X boundary
      tapsize = 1.0 / float(nxbnd + thickness);
      int ix1 = min(nx, 0);
      int ix2 = min(nx, (nxbnd + thickness));
#pragma omp for schedule(static)
      for(int iy = 0; iy < ny; iy++) {
        for(int ix = ix1; ix < ix2; ix++) {
          int iix = ix - ix1;
          float weight = 0.5f * (1.0 - cos(PI * iix * tapsize));
          size_t id = iy * nx + ix;
          id *= nz;
          for(int iz = 0; iz < nz; iz++) {
            vol[id + iz] *= weight;
          }
        }
      }

      ix1 = max(0, (nx - nxbnd - thickness));
      ix2 = max(0, (nx));
#pragma omp for schedule(static)
      for(int iy = 0; iy < ny; iy++) {
        for(int ix = ix1; ix < ix2; ix++) {
          int iix = ix - ix1;
          float weight = 0.5f * (1.0 + cos(PI * iix * tapsize));
          size_t id = iy * nx + ix;
          id *= nz;
          for(int iz = 0; iz < nz; iz++) {
            vol[id + iz] *= weight;
          }
        }
      }
    }
tapsize = 1.0 / float(nzbnd + thickness);
    // Tapper the Low boundary only for Z
    int iz0 = min(nz, (nzbnd + thickness));
    int iz1 = max(0, (nz - nzbnd - thickness));
    int iz2 = max(0, (nz));
#pragma omp for schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        size_t id = iy * nx + ix;
        id *= nz;
        for(int iz = 0; iz < iz0; iz++) {
          float weight = 0.5f * (1.0 - cos(PI * iz * tapsize));
          vol[id + iz] *= weight;
        }
        for(int iz = iz1; iz < iz2; iz++) {
          int iiz = iz - iz1;
          float weight = 0.5f * (1.0 + cos(PI * iiz * tapsize));
          vol[id + iz] *= weight;
        }
      }
    }
  }
}
void Propagator::Vel2(float dt) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict vel = volModel[VEL];
  size_t nxz = nx * nz;
  if(!myModel->useQ) {
    __m128 dt2 = _mm_set1_ps((dt * dt));
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
        size_t i = iy * nxz + ixz;
        __m128 v2 = _mm_load_ps(vel + i);
        v2 = _mm_mul_ps(_mm_mul_ps(v2, v2), dt2);
        _mm_store_ps(vel + i, v2);
      }
    }
  } else {
    float *__restrict invQ = volModel[Q];
    float dt2 = dt * dt;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(size_t ixz = 0; ixz < nxz; ixz++) {
        size_t i = iy * nxz + ixz;
        float v = vel[i];
        float invq = invQ[i];
        float invq2 = invq * invq;
        float correction = 1 / (1 - Q::cq0 * invq);
        vel[i] = v * v * dt2 * (1 - 0.25f * invq2 + invq2 * invq2 / 48) * correction;
        invQ[i] = invq * correction;
      }
    }
  }
}

float Propagator::calcDeltaTime() {
  // dt was changed to calculate globally
  if(global_pars["_dt_prop"]) return global_pars["_dt_prop"].as<float>();

  // for stand-alone mod job, need calc dt
  MigParams migParams;
  ModelRegrid modelRegrid;

  string workDir = expEnvVars(global_pars["workDir"].as<string>("/tmp"));
  time_t tm = time(nullptr);
  workDir = workDir + "/" + "mod" + "_" + std::to_string(getpid()) + "_" + std::to_string(tm / 60);
  libCommon::Utl::mkpath(workDir.c_str());

  modelRegrid.setPath(workDir.c_str());
  modelRegrid.prepModel(&migParams, nullptr);
  float dt = migParams.dt;

  global_pars["_dt_prop"] = (double)dt; // YAML writing as float does not preserve precision
  return dt;
}
float Propagator::getDtSave(float recordingFreq) {
  float dtpro = calcDeltaTime();
  float dtNyquist = 0.5f / recordingFreq;
  int dtratio = int(dtNyquist / dtpro);
  assertion(dtratio > 0, "The Delta T has some problem, the propagation dt(%f) is larger than Nyquist DT(%f)", dtpro, dtNyquist);
  dtratio = global_pars["nprop_per_ic"].as<int>(dtratio);
  return dtratio * dtpro;
}

float Propagator::getDtSave() {
  assertion(global_pars["_dt_prop"].IsDefined(), "_dt_prop need to be known in advance!");
  float dtpro = global_pars["_dt_prop"].as<float>();
  float dtNyquist = 0.5f / global_pars["maxFreq"].as<float>();
  int dtratio = int(dtNyquist / dtpro);
  assertion(dtratio > 0, "The Delta T has some problem, the propagation dt(%f) is larger than Nyquist DT(%f)", dtpro, dtNyquist);
  dtratio = global_pars["nprop_per_ic"].as<int>(dtratio);
  return dtratio * dtpro;
}
int Propagator::setupReceiver(PROP::Operation oper, int spreadSize, int i_srcrcv, vector<float> &x, vector<float> &y, vector<float> &z,
    vector<float> &delay) {
  float slow0 = myModel->getModelValue(VEL, x[0], y[0], z[0]); // (vel*dt)^2
  slow0 = dtpro / sqrtf(slow0);
  myshot = new Source(myGrid.get()); // myshot's t0 is important
  float min_delay = *min_element(delay.begin(), delay.end()); // if it's negative, we need leave enough source rising time
  Wavelet *wavelet = myshot->setupWavelet(Source::getWaveletType(), Source::getWaveletPhaseType(), slow0, nt, dtpro, maxFreq, oper, dim, 0,
                                          0, min_delay);

  int nsrc = (int)x.size();
  it0 = myshot->it0;

  // int ntsrc = nt; //DEBUG
  int ntsrc = nt + it0; // myshot->nt = nt + it0

  receivers[i_srcrcv] = make_shared<Receiver>(myGrid, myModel, dtpro, ntsrc, maxFreq, minFreq, spreadSize, dim,
                                              wavelet->waveletType == USERWAVELET ? Receiver::USER_WAVELET : Receiver::SOURCE, oper);

  FreqUtil fft(ntsrc);
  unique_ptr<Traces> traces = make_unique<Traces>(ntsrc, nullptr, 0, dtpro);
  vector<float> qc((size_t)ntsrc * (nsrc + 1), 0);
  if(global_pars["qc_nsrc"]) memcpy(&qc[0], wavelet->mysource, sizeof(float) * ntsrc);
  fft.timeshift_in(wavelet->mysource);
  for(int i = 0; i < nsrc; i++) {
    vector3 xyz(x[i], y[i], z[i]);
    traces->addReceiver(xyz);
    print1m("delay=%f, delay_over_dt=%f\n", delay[i], delay[i] / dtpro);
    fft.timeshift_out(traces->data[i], delay[i] / dtpro);
    if(global_pars["qc_nsrc"]) memcpy(&qc[(i + 1) * ntsrc], traces->data[i], sizeof(float) * ntsrc);
  }
  if(global_pars["qc_nsrc"])
  jseisUtil::save_zxy(getJsFilename("qc_nsrc", i_srcrcv ? "_bwd" : "").c_str(), &qc[0], ntsrc, nsrc + 1, 1, dtpro * 1000);
  receivers[i_srcrcv]->traces = move(traces);
  receivers[i_srcrcv]->update_zrange();
  receivers[i_srcrcv]->spread();

  delete wavelet;
  return 1;
}

unique_ptr<Traces> Propagator::setupReceiver(int spreadSize, int i_srcrcv, unique_ptr<Traces> traces, bool keepTraces) {
  unique_ptr<Traces> inTraces;
  receivers[i_srcrcv] = make_shared<Receiver>(myGrid, myModel, dtpro, nt, maxFreq, minFreq, spreadSize, dim, i_srcrcv, PROP::RTM);
  if(traces) {
    float dt = traces->dt;
    auto vec_traces = receivers[i_srcrcv]->recordLoader->filterTraces(move(traces), dt, 0, false, keepTraces);
    receivers[i_srcrcv]->traces = move(vec_traces[0]);
    inTraces = move(vec_traces[1]);
    receivers[i_srcrcv]->update_zrange(false);
  } else {
    string key = i_srcrcv ? "RTMInputFile" : "SourceInputFile";
    assertion(global_pars[key].IsDefined(), "Key '%s' is required to call setupReceiver(spreadSize, %s)", key.c_str(),
              i_srcrcv ? "Receiver" : "Source");

    string fileName = expEnvVars(global_pars[key].as<string>());
    print1m("RTMInput: %s \n", fileName.c_str());
    if(!receivers[i_srcrcv]->loadData(fileName.c_str())) return nullptr;
  }
  receivers[i_srcrcv]->spread();

  return inTraces;
}
shared_ptr<Receiver> Propagator::setupReceiverForModeling(int spreadSize, unique_ptr<Traces> traces) {
  bool doTemplate = global_pars["doTemplate"].as<int>(1); // set to 0 to use modeling code based on grid
  if(!doTemplate && !global_pars["ReceiverTemplateFile"]) return NULL;

  receivers[1] = make_shared<Receiver>(myGrid, myModel, dtpro, nt, maxFreq, minFreq, spreadSize, dim, Receiver::RECEIVER, PROP::MOD);
  if(traces) {
    receivers[1]->traces = move(traces);
    receivers[1]->update_zrange(false);
  } else if(!global_pars["ReceiverTemplateFile"]) receivers[1]->createHdrFromGrid(global_pars);
  else {
    string fileName = expEnvVars(global_pars["ReceiverTemplateFile"].as<string>());
    print1m("ReceiverTemplate: %s \n", fileName.c_str());
    assertion(receivers[1]->loadHdr(fileName.c_str()), "Failed to load hdr from ReceiverTemplate!");
  }

  int do_dipole = Receiver::isDipole(PROP::BACKWARD, PROP::MOD);
  print1m("# setupReceiverForModeling: do_dipole=%d\n", do_dipole);
  receivers[1]->spreadCoeffs(0, do_dipole);
  return receivers[1];
}

void Propagator::copySnapshotGrid(float *w0, float *ws, int nElements) const {
  memcpy(ws, w0, sizeof(float) * nElements);
}
void Propagator::printTime(int it, PROP::Direction direct, float *w) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  int show1 = static_cast<int>(nt / 100. + 0.5);
  int show2 = static_cast<int>(nt / 10. + 0.5);

  int it01 = 0;
  if(direct == PROP::BACKWARD) it01 = nt - 1;

  if(it == it01 || it == show1 || !(it % show2)) {
    int perc = (100 * it) / nt;
    if(direct == PROP::BACKWARD) {
      perc = 100 - perc;
    }
    time_t elapsed = time(0) - time_start;
    time_t remaining_estimate = (perc > 0) ? elapsed * (100 - perc) / perc : 0;
    print1m("[%s %d] %s> %s: Time step: %5d %3.1d%% done. MemFree: %s.", hostname, posLogical[2], time_now().c_str(),
            direct == PROP::FORWARD ? "Fwd" : "Bwd", it, perc, libCommon::Utl::free_memory().c_str());
    if(remaining_estimate > 0) {
      print1m(" Estimated remaining: %luh%lum%lus\n", remaining_estimate / 3600, (remaining_estimate / 60) % 60, remaining_estimate % 60);
    } else {
      print1m("\n");
    }
    float ampmax = libCommon::maxfabs(w, nz, nx * ny, nThreads);
    print1m("amplitude:%g \n", ampmax);

  }
}
void Propagator::kernel() {

    printf("Start Propagator::Kernel0128\n");

   // gpu_test();

    if(engineType == FD::FFTV || engineType == FD::CFD_BATCH || engineType == FD::SINCOS)
    {
        std::cout<<"if engineType="<<engineType<<std::endl;
        derive->getGradient(myWavefield); //w1 -> wx, wy, wz (+bc)

        // March 2018
        if(myModel->useRho) divideRho(myWavefield);  // wx, wy, wz // removed RhoCN* == 0 by wolf

        if(myModel->modeltype == TTI)
        {
            std::cout<<"modeltype=TTI"<<std::endl;
            CalTTIscaler(myWavefield);
        }
        else if(myModel->modeltype == VTI)
        {
            std::cout<<"modeltype=VTI"<<std::endl;
            CalVTIscaler(myWavefield);
        }
        derive->getDiverge(myWavefield); // wx, wy, wz-> wb (+bc)
        // March 2018
        if(myModel->useRho)
        {
//      if(myModel->RhoCN* == 0) {
            multiplyRho(myWavefield); // wb
//      } else if(myModel->RhoCN* == 1) {
//        multiplycnnRho(myWavefield);
//      } // revised by wolf to remove the case RhoCN* == 1
        }
        applyVel(myWavefield);
        int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
        size_t nxz = nx * nz;
        //if(!myModel->useQ) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
        for(int iy = 0; iy < ny; iy++) {
            for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
                size_t i = iy * nxz + ixz;
                __m128 ww0 = _mm_load_ps(myWavefield->w0 + i);
                __m128 wwb = _mm_load_ps(myWavefield->wb + i);
                _mm_store_ps(myWavefield->w0 + i, _mm_sub_ps(ww0, wwb));
            }
      // }//revised by wolf on Sep 21. 2022.
        }

    }
  else
    {
        int tprlen = nzbnd;
        if(myModel->modeltype == TTI)
            kernelCPU->TTI(myWavefield->w1, myWavefield->w0, myWavefield->wr);
       //second argument changed from wb to w0 by wolf
        else if(myModel->modeltype == VTI)
            kernelCPU->VTI(myWavefield->w1, myWavefield->w0, myWavefield->wr);
        //second argument changed from wb to w0 by wolf
        else if(myModel->modeltype == ISO)
            kernelCPU->ISO(myWavefield->w1, myWavefield->w0, myWavefield->wr, sfBoundType[prop_dir], myGrid->getIDz(zsurf), tprlen);
        //second argument changed from wb to w0 by wolf
    }
    printf("End Propagator::Kernel\n");
}

void Propagator::kernel(Wavefield *myLocalWavefield, Derivative *myLocalDerive, KernelCPU *myLocalKernel) {
  if(engineType == FD::FFTV || engineType == FD::CFD_BATCH || engineType == FD::SINCOS) {
    myLocalDerive->getGradient(myLocalWavefield); //w1 -> wx, wy, wz (+bc)
    if(myModel->useRho) divideRho(myLocalWavefield);  // wx, wy, wz // revised by wolf to remove  && myModel->RhoCN* == 0

    if(myModel->modeltype == TTI) {
      CalTTIscaler(myLocalWavefield);
    } else if(myModel->modeltype == VTI) {
      CalVTIscaler(myLocalWavefield);
    }
    myLocalDerive->getDiverge(myLocalWavefield); // wx, wy, wz-> wb (+bc)
    // March 2018
    if(myModel->useRho) {
//      if(myModel->RhoCN* == 0) {
      multiplyRho(myLocalWavefield); // wb
//      } else if(myModel->RhoCN* == 1) {
//        multiplycnnRho(myLocalWavefield);
//      }//revised by wolf to remove RhoCN*
    }
    applyVel(myLocalWavefield);
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    size_t nxz = nx * nz;
//      if(!myModel->useQ) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
        size_t i = iy * nxz + ixz;
        __m128 ww0 = _mm_load_ps(myLocalWavefield->w0 + i);
        __m128 wwb = _mm_load_ps(myLocalWavefield->wb + i);
        _mm_store_ps(myLocalWavefield->w0 + i, _mm_sub_ps(ww0, wwb));
      }
//        }//revised by wolf on Sep 21. 2022.
    }
  } else {
    int tprlen = nzbnd;
    if(myModel->modeltype == TTI) myLocalKernel->TTI(myLocalWavefield->w1, myLocalWavefield->w0, myLocalWavefield->wr);
    else if(myModel->modeltype == VTI) myLocalKernel->VTI(myLocalWavefield->w1, myLocalWavefield->w0, myLocalWavefield->wr);
    else if(myModel->modeltype == ISO)
      myLocalKernel->ISO(myLocalWavefield->w1, myLocalWavefield->w0, myLocalWavefield->wr, sfBoundType[prop_dir], myGrid->getIDz(zsurf),
                         tprlen);
  }
}
bool Propagator::prepSnapJS(bool &snap3D, unique_ptr<oJseis3D> &js_snap, unique_ptr<oJseis3D> &js_snap_y, unique_ptr<oJseisND> &js_snap3D,
    vector<float> &buf_xslice, int nsnap, int isav, PROP::Direction direct) {
  bool snapJS = false;
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;

  string snapShotPrefix;
  if(global_pars["snapShotPrefix"]) {
    snapShotPrefix = expEnvVars(global_pars["snapShotPrefix"].as<string>());
    // doSnapshot = true;
    snapJS = global_pars["snapShotFormat"].as<string>("js").compare("js") == 0;
    snap3D = getBool(global_pars["snapshot3D"], false) && ny > 1;
  }
  if(snapJS) {
    string ext = direct == PROP::FORWARD ? "_fwd.js" : direct == PROP::BACKWARD ? "_bwd.js" : ".js";
    qcVel(ext);
    string fname = snapShotPrefix + ext;
    string fname2 = snapShotPrefix + "_y" + ext;
    string fname3D = snapShotPrefix + "_3D" + ext;
    js_snap.reset(
        new oJseis3D(fname.c_str(), nz, nx, nsnap, myGrid->dz, myGrid->dx, dtpro * isav * 1000, -myGrid->getIDz(0.0), myGrid->x0, it0));
    if(ny > 1) {
      js_snap_y.reset(
          new oJseis3D(fname2.c_str(), nz, ny, nsnap, myGrid->dz, myGrid->dy, dtpro * isav * 1000, -myGrid->getIDz(0.0), myGrid->y0, it0));
      buf_xslice.resize(nz * ny);
    }
    if(snap3D)
      js_snap3D.reset(
          new oJseisND(fname3D, vector<int> { nz, nx, ny, nsnap }, vector<int> { -myGrid->getIDz(0.0), 1, 1, -it0 }, vector<int> { 1, 1, 1,
                           isav },
                       vector<double> { myGrid->z0, myGrid->x0, myGrid->y0, -it0 * dtpro * isav * 1000 }, vector<double> { myGrid->dz,
                           myGrid->dx, myGrid->dy, dtpro * isav * 1000 }, //
                       vector<string> { "DEPTH", "XLINE_NO", "ILINE_NO", "TIMESTEP" }));
  }
  return snapJS;
}

void Propagator::saveSnap(int i_snap, bool snap3D, unique_ptr<oJseis3D> &js_snap, unique_ptr<oJseis3D> &js_snap_y,
    unique_ptr<oJseisND> &js_snap3D, vector<float> &buf_xslice) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;

  if(snap3D) js_snap3D->write_volume(myWavefield->w1, i_snap); // revised from wb to w1
  js_snap->write_frame(myWavefield->w1 + nxz * (ny / 2), i_snap);  // slice iy=ny/2 //revised from wb to w1
  if(ny > 1) {
    for(int iy = 0; iy < ny; iy++)
      memcpy(&buf_xslice[iy * nz], myWavefield->w1 + ((size_t)iy * nx + nx / 2) * nz, sizeof(float) * nz); //revised from wb to w1
    js_snap_y->write_frame(&buf_xslice[0], i_snap);
  }
//    char fname[256];
//    sprintf(fname, "%s_%03d_wave%04d.fdm", snapShotPrefix.c_str(), direct, it);
//    myGrid->savefdm(fname, myWavefield->wb);
}
void Propagator::modelling(unique_ptr<ExtractReceiverData> &erd, PROP::Direction direct) { // save all the wave fields
  prop_dir = direct;
  bool ghost = determine_ghost(direct);
  myWavefield->allocateMemory(allocWx, nThreads);

    printf("Start Propagator::modelling\n");

    if(engineType != FD::FFTV && engineType != FD::CFD_BATCH && engineType != FD::SINCOS)
    {
        kernelCPU->cleanPMLMemory();
    }
    else
    {
        if(derive) derive->cleanMemory();  // in case derive was commented out
    }

    float tmaxratio = global_pars["tmaxratio"].as<float>(1.05f);

    int nts = nt * tmaxratio + it0;  //source time steps
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;

    int itsav = 0;

    int isav = global_pars["snapshotInterval"].as<int>(100); // save interval for qc
    // print1m("The nt and isav is %d   %d\n", nt, isav);
    int nsnap = global_pars["snapshotLimit"].as<int>(9999);
    nsnap = min(nsnap, nts / isav);
    int i_snap = 0;
    bool snap3D = false;
    unique_ptr<oJseis3D> js_snap, js_snap_y;
    unique_ptr<oJseisND> js_snap3D;
    vector<float> buf_xslice;
    bool snapJS = prepSnapJS(snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice, nsnap, isav);
   int it_shift_force = global_pars["it_shift_force"].as<int>(-1);
    int it01 = 0;
    int it02 = nts;
    int itinc = 1;
    int sym;

    myWavefield->cleanMemory(nThreads);

    int it = it01, ita = 0;

    time_start = time(0);

    int jt = 0;
    int ntratio = (int)nearbyintf(erd->dt / dtpro);
    while(it != it02)
    {

        int its = (it - it0);

        printf("its=%d,it=%d\n",its,it);

        float mytimerun = its * dtpro;

        //run the kernel
        kernel(); // w1 -> (w0 - v^2* dt^2* divergence), w0 <-w0 - v^2 * dt^2* divergence

        //applyVel(myWavefield); // wb *= (vdt)^2 // changed from code to text by wolf

        update2nd(myWavefield); // w0 <- 2*w1-w0. Later w0 <-> w1

        receivers[0]->apply(myWavefield, ita = it + it_shift_force, ghost, volModel[VEL], 1);

        if(sfBoundType[0] == FREESURFACE)
        {
 sym = -1;
          //      apply_symmetry(myWavefield->w1, sym);
          //      apply_symmetry(myWavefield->w0, sym);
            apply_symmetry(myWavefield->w0, sym); //changed from code to text by wolf
        }

        // extract receiver data
        if(it >= 0 && its % ntratio == 0)
        {
      //        print1m("Modeling: it0=%d, it=%d, its=%d, ntratio=%d, jt=%d, outvol_nt=%d, outvol_t0=%f, outvol_dt=%f\n", it0, it,
      //            its, ntratio, jt, outvol->nt, outvol->t0, outvol->dt), fflush(stdout);
            if(jt == 0)
                assertion(fabsf(mytimerun - erd->t0) / dtpro < 0.01f, "Inconsistent ExtractReceiverData.t0! mytime=%f, erd->t0=%f", mytimerun,
                  erd->t0);
            erd->extract(jt, myWavefield->w0); //changed from code to text by wolf
            jt++;
        }

        //print & snapshot
        this->printTime(it, direct, myWavefield->w0); //changed from code to text by wolf

        myWavefield->swapPointers(); //  w1<->w0, revised by wolf

        if(snapJS && ita >= 0 && (!(ita % isav)) && i_snap < nsnap)
        {
            saveSnap(i_snap++, snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice);
        }

        // Switch pointers here

        it += itinc;

    } // end of it loop
 printf("End Propagator::modelling\n");
}

void Propagator::migration(CacheFile *cachefile, PROP::Direction direct) {                // save all the wave fields
  prop_dir = direct;
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  bool ghost = determine_ghost(direct);
  if(direct == PROP::FORWARD) {
    myWavefield->allocateMemory(allocWx, nThreads);
  }

  if(engineType != FD::FFTV && engineType != FD::CFD_BATCH && engineType != FD::SINCOS) {
    kernelCPU->cleanPMLMemory();
  } else {
    if(derive) derive->cleanMemory();  // in case derive was commented out
  }

  size_t compSize = WaveFieldCompress::nshort_volume(nz, nx, ny);
  ushort *compBuf = new ushort[compSize]();
  AsyncIO *aio = new AsyncIO(cachefile, compSize * sizeof(ushort));

  float tmaxratio = global_pars["tmaxratio"].as<float>(0.7f);
  if(tmaxratio > 1.0f) tmaxratio = 1.0f;

  int nts = nt * tmaxratio + it0;  //source time steps, always include source rising time
  int ntr = nt - nearbyintf(cachefile->t0 / dtpro);  //receiver time steps, depends on cachefile->t0 which defaults to 0
  int nskip_ic = (int)nearbyintf(cachefile->dt / dtpro); // imaging condition
  size_t nbytes;

  int isav = global_pars["snapshotInterval"].as<int>(100); // save interval for qc
  // print1m("The nt and isav is %d   %d\n", nt, isav);
  int nsnap = global_pars["snapshotLimit"].as<int>(9999);
  nsnap = min(nsnap, (nts - it0) / isav);
  int i_snap = 0;
  bool snap3D = false;
  unique_ptr<oJseis3D> js_snap, js_snap_y;
  unique_ptr<oJseisND> js_snap3D;
  vector<float> buf_xslice;
  bool snapJS = prepSnapJS(snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice, nsnap, isav, direct);

  int it_shift_force = global_pars["it_shift_force"].as<int>(-1);
  int it01 = 0;
  int it02 = nts;
  int itinc = 1;
  int it0 = this->it0;
  if(direct == PROP::BACKWARD) {
    it01 = nt - 1;
    it02 = (nt - ntr) - 1;
    itinc = -1;
    it0 = 0;
  }

  myWavefield->cleanMemory(nThreads);

  time_start = time(0);
  bool do_join = false;
  for(int it = it01; it != it02; it += itinc) {

    float t = (it - it0) * dtpro;

    //myBndTaper->apply(myWavefield);

    //run the kernel
    this->kernel();
    // Util::print_mem_crc(myWavefield->wb, nz, nx * ny, "wb", 1); // skipzero=1

    //applyVel(myWavefield); changed from code to comment by wolf


    update2nd(myWavefield);

    int itapply = it + it_shift_force * (direct == PROP::FORWARD ? 1 : -1);
    receivers[direct]->apply(myWavefield, itapply, ghost, volModel[VEL], 1);

    if(sfBoundType[direct] == FREESURFACE) {
      apply_symmetry(myWavefield->w0, -1); //changed from wb to w0 by wolf
    }

    // record the wavefield
    if(cachefile) {
      int itsav = (int)nearbyintf((t - cachefile->t0) / cachefile->dt);
      float mytimesav = itsav * cachefile->dt + cachefile->t0;

      if((fabsf(mytimesav - t) < 0.5f * dtpro) && (itsav >= 0 && itsav < cachefile->nt)) {
        if(do_join) aio->join();

        wfComp->compress(myWavefield->w0, compBuf, nx, ny, nz); //changed from wb to w0 by wolf
        aio->pwrite(compBuf, itsav, &nbytes), do_join = true;
      }
    }

    myWavefield->swapPointers(); //move from line 1156 to here by wolf

    if(snapJS && itapply >= 0 && (!(itapply % isav)) && i_snap < nsnap) {
      saveSnap(direct == PROP::FORWARD ? i_snap : nsnap - i_snap - 1, snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice);
      i_snap++;
    }

    // Switch pointers here

    //print the time
    printTime(it, direct, myWavefield->w0); //changed from wb to w0 by wolf } // end of it loop

  if(do_join) aio->join();
  delete aio;
  delete[] compBuf;
}

void Propagator::demigration(unique_ptr<ExtractReceiverData> &erd, PROP::Direction direct) { // save all the wave fields
  prop_dir = direct;
  bool ghost = determine_ghost(direct);
  myWavefield->allocateMemory(allocWx, nThreads);
  myDemigWavefield->allocateMemory(allocWx, nThreads);

  //  myGrid->savefdm("image_grid.fdm", myWavefield->image);

  if(engineType != FD::FFTV && engineType != FD::CFD_BATCH && engineType != FD::SINCOS) {
    kernelCPU->cleanPMLMemory();
    kernelDemig->cleanPMLMemory();
  } else {
    if(derive) derive->cleanMemory();  // in case derive was commented out
    if(deriveDemig) deriveDemig->cleanMemory();  // in case deriveDemig was commented out
  }

  float tmaxratio = global_pars["tmaxratio"].as<float>(1.05f);

  int nts = nt * tmaxratio + it0;  //source time steps
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;

  int itsav = 0;

  int isav = global_pars["snapshotInterval"].as<int>(100); // save interval for qc
  // print1m("The nt and isav is %d   %d\n", nt, isav);
  int nsnap = global_pars["snapshotLimit"].as<int>(9999);
  nsnap = min(nsnap, (nts - it0) / isav);
 int i_snap = 0;
  bool snap3D = false;
  unique_ptr<oJseis3D> js_snap, js_snap_y;
  unique_ptr<oJseisND> js_snap3D;
  vector<float> buf_xslice;
  bool snapJS = prepSnapJS(snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice, nsnap, isav);

  int it_shift_force = global_pars["it_shift_force"].as<int>(-1);
  int it01 = 0;
  int it02 = nts;
  int itinc = 1;
  int sym;

  myWavefield->cleanMemory(nThreads);
  myDemigWavefield->cleanMemory(nThreads);

  int it = it01, ita = 0;

  time_start = time(0);

  int jt = 0;
  int ntratio = (int)nearbyintf(erd->dt / dtpro);
  while(it != it02) {

    int its = (it - it0);

    float mytimerun = its * dtpro;

    // 1. propagation with source
    kernel(myWavefield, derive, kernelCPU);  // w1 -> (w0 - v^2* dt^2* divergence), w0 <-w0 - v^2 * dt^2* divergence
//    applyVel(myWavefield);//by wolf
    update2nd(myWavefield);  // w0 <- 2*w1-w0. Later w0 <-> w1
    receivers[0]->apply(myWavefield, ita = it + it_shift_force, ghost, volModel[VEL], 1);

    // 2. propagation with reflector
    kernel(myDemigWavefield, deriveDemig, kernelDemig);  // w1 -> (w0 - v^2* dt^2* divergence), w0 <-w0 - v^2 * dt^2* divergence
//    applyVel(myDemigWavefield); // by wolf
    update2nd(myDemigWavefield); // w0 <- 2*w1-w0. Later w0 <-> w1
    applyReflectivity(myWavefield, myDemigWavefield);

    // extract receiver data
    if(it >= 0 && its % ntratio == 0) {
      // print1m("Modeling saving: it0=%d, it=%d, its=%d, jt=%d,  mytimerun=%f, myvalue = %f \n", it0, it, its, jt, mytimerun, myWavefield->w1[100]), fflush(stdout);
      // print1m("Modeling saving: it0=%d, it=%d, its=%d, jt=%d,  mytimerun=%f, myvalue = %f \n", it0, it, its, jt, mytimerun, myWavefield->w0[100]), fflush(stdout);
      // print1m("Modeling saving: it0=%d, it=%d, its=%d, jt=%d,  mytimerun=%f, myvalue = %f \n", it0, it, its, jt, mytimerun, myWavefield->wb[100]), fflush(stdout);
      // print1m("Modeling saving: it0=%d, it=%d, its=%d, jt=%d,  mytimerun=%f, myvalue = %f \n", it0, it, its, jt, mytimerun, myDemigWavefield->w1[100]), fflush(stdout);
      if(jt == 0)
        assertion(fabsf(mytimerun - erd->t0) / dtpro < 0.01f, "Inconsistent ExtractReceiverData.t0! mytime=%f, erd->t0=%f", mytimerun,
                  erd->t0);
      //erd->extract(jt, myWavefield->wb);
      erd->extract(jt, myDemigWavefield->w1);
      jt++;
    }

    //print & snapshot
    this->printTime(it, direct, myDemigWavefield->w0);

    myWavefield->swapPointers(); // moved from line 1258 to here by wolf
    myDemigWavefield->swapPointers(); // moved from line 1263 to here by wolf

    if(snapJS && ita >= 0 && (!(ita % isav)) && i_snap < nsnap) {
      // this saves wb, if wants w1, add more parameters ...
      saveSnap(i_snap++, snap3D, js_snap, js_snap_y, js_snap3D, buf_xslice);    }

    // Switch pointers here

    // Switch pointers here

    it += itinc;

  } // end of it loop
}

void Propagator::CalVTIscaler(Wavefield *myWavefield) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  if(ny == 1) {
    CalVTIscaler2D(myWavefield);
    return;
  }

  float *__restrict eps = volModel[EPS];
  float *__restrict del = volModel[DEL];
  size_t nxz = nx * nz;
  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 myeps = _mm_load_ps(eps + i);
      __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
      myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
      __m128 vectx = _mm_load_ps(myWavefield->wx + i);
      __m128 vecty = _mm_load_ps(myWavefield->wy + i);
      __m128 vectz = _mm_load_ps(myWavefield->wz + i);
      __m128 vecx2 = _mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vecty, vecty));
      __m128 vecz2 = _mm_mul_ps(vectz, vectz);
      __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vecx2), vecz2);
      __m128 parts = _mm_sqrt_ps(
          _mm_sub_ps(
              one,
              _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vecx2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
      __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
      __m128 scale2 = _mm_mul_ps(myeps, _mm_min_ps(scale, one));

      _mm_store_ps(myWavefield->wx + i, _mm_mul_ps(vectx, scale2));
      _mm_store_ps(myWavefield->wy + i, _mm_mul_ps(vecty, scale2));
    }
  }
}
void Propagator::CalVTIscaler2D(Wavefield *myWavefield) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict eps = volModel[EPS];
  float *__restrict del = volModel[DEL];
  assert(nx > 1); // 1D not needed for TTI

  size_t nxz = nx * nz;
  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 myeps = _mm_load_ps(eps + i);
      __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
      myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
      __m128 vectx = _mm_load_ps(myWavefield->wx + i);
      __m128 vectz = _mm_load_ps(myWavefield->wz + i);
      __m128 vecx2 = _mm_mul_ps(vectx, vectx);
      __m128 vecz2 = _mm_mul_ps(vectz, vectz);
      __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vecx2), vecz2);
      __m128 parts = _mm_sqrt_ps(
          _mm_sub_ps(
              one,
              _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vecx2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
      __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
      __m128 scale2 = _mm_mul_ps(myeps, _mm_min_ps(scale, one));

      _mm_store_ps(myWavefield->wx + i, _mm_mul_ps(vectx, scale2));
    }
  }
}

void Propagator::CalTTIscaler(Wavefield *myWavefield) {
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    printf("Start Propagator::CalTTIscaler %d %d %d\n",nx,ny,nz);
    if(ny == 1)
    {
        CalTTIscaler2D(myWavefield);
        return;
    }

  float *__restrict eps = volModel[EPS];
  float *__restrict del = volModel[DEL];
  float *__restrict pjx = volModel[PJX];
  float *__restrict pjy = volModel[PJY];
  size_t nxz = nx * nz;
  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);
#pragma omp parallel for num_threads(nThreads) schedule(static) 
    for(int iy = 0; iy < ny; iy++)
        {
        for(size_t ixz = 0; ixz < nxz; ixz += SSEsize)
        {
            size_t i = iy * nxz + ixz;
            __m128 myeps = _mm_load_ps(eps + i);
            __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
            myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
            __m128 axisx = _mm_load_ps(pjx + i);
            __m128 axisy = _mm_load_ps(pjy + i);
            __m128 axisz = _mm_sqrt_ps(_mm_sub_ps(one, _mm_add_ps(_mm_mul_ps(axisx, axisx), _mm_mul_ps(axisy, axisy))));
            __m128 vectx = _mm_load_ps(myWavefield->wx + i);
            __m128 vecty = _mm_load_ps(myWavefield->wy + i);
            __m128 vectz = _mm_load_ps(myWavefield->wz + i);

            __m128 vect2 = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vecty, vecty)), _mm_mul_ps(vectz, vectz));
            __m128 veczz = _mm_add_ps(_mm_add_ps(_mm_mul_ps(vectx, axisx), _mm_mul_ps(vecty, axisy)), _mm_mul_ps(vectz, axisz));
            __m128 vecz2 = _mm_mul_ps(veczz, veczz);
            __m128 vech2 = _mm_sub_ps(vect2, vecz2);
            __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vech2), vecz2);

            __m128 parts = _mm_sqrt_ps(
                             _mm_sub_ps(one,
                            _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vech2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
            __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
            scale = _mm_min_ps(scale, one);
            axisx = _mm_mul_ps(veczz, axisx);
            axisy = _mm_mul_ps(veczz, axisy);
            axisz = _mm_mul_ps(veczz, axisz);
            _mm_store_ps(myWavefield->wx + i, _mm_mul_ps(_mm_add_ps(axisx, _mm_mul_ps(_mm_sub_ps(vectx, axisx), myeps)), scale));
            _mm_store_ps(myWavefield->wy + i, _mm_mul_ps(_mm_add_ps(axisy, _mm_mul_ps(_mm_sub_ps(vecty, axisy), myeps)), scale));
            _mm_store_ps(myWavefield->wz + i, _mm_mul_ps(_mm_add_ps(axisz, _mm_mul_ps(_mm_sub_ps(vectz, axisz), myeps)), scale));    }   
    }   
    printf("End Propagator::CalTTIscaler %d %d %d\n",nx,ny,nz);
}

void Propagator::CalTTIscaler2D(Wavefield *myWavefield) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict eps = volModel[EPS];
  float *__restrict del = volModel[DEL];
  float *__restrict pjx = volModel[PJX];
  assert(nx > 1); // 1D not needed for TTI

  size_t nxz = nx * nz;
  __m128 one = _mm_set1_ps(1.0f);
  __m128 two = _mm_set1_ps(2.0f);
  __m128 eight = _mm_set1_ps(8.0f);
  __m128 half = _mm_set1_ps(0.5f);
  __m128 small = _mm_set1_ps(1e-32);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 myeps = _mm_load_ps(eps + i);
      __m128 mydel = _mm_sub_ps(myeps, _mm_load_ps(del + i));
      myeps = _mm_add_ps(one, _mm_mul_ps(two, myeps));
      __m128 axisx = _mm_load_ps(pjx + i);
      __m128 axisz = _mm_sqrt_ps(_mm_sub_ps(one, _mm_mul_ps(axisx, axisx)));
      __m128 vectx = _mm_load_ps(myWavefield->wx + i);
      __m128 vectz = _mm_load_ps(myWavefield->wz + i);

      __m128 vect2 = _mm_add_ps(_mm_mul_ps(vectx, vectx), _mm_mul_ps(vectz, vectz));
      __m128 veczz = _mm_add_ps(_mm_mul_ps(vectx, axisx), _mm_mul_ps(vectz, axisz));
      __m128 vecz2 = _mm_mul_ps(veczz, veczz);
      __m128 vech2 = _mm_sub_ps(vect2, vecz2);
    __m128 eclip = _mm_add_ps(_mm_mul_ps(myeps, vech2), vecz2);

      __m128 parts = _mm_sqrt_ps(
          _mm_sub_ps(
              one,
              _mm_div_ps(_mm_mul_ps(eight, _mm_mul_ps(mydel, _mm_mul_ps(vech2, vecz2))), _mm_add_ps(_mm_mul_ps(eclip, eclip), small))));
      __m128 scale = _mm_mul_ps(half, _mm_add_ps(one, parts));
      scale = _mm_min_ps(scale, one);
      axisx = _mm_mul_ps(veczz, axisx);
      axisz = _mm_mul_ps(veczz, axisz);
      _mm_store_ps(myWavefield->wx + i, _mm_mul_ps(_mm_add_ps(axisx, _mm_mul_ps(_mm_sub_ps(vectx, axisx), myeps)), scale));
      _mm_store_ps(myWavefield->wz + i, _mm_mul_ps(_mm_add_ps(axisz, _mm_mul_ps(_mm_sub_ps(vectz, axisz), myeps)), scale));
    }
  }
}
bool Propagator::determine_ghost(PROP::Direction direct) {
  if(direct == PROP::BACKWARD) return sfBoundType[1] == FREESURFACE || sfBoundType[1] == RECEIVER_GHOST || sfBoundType[1] == GHOST;
  return sfBoundType[0] == FREESURFACE || sfBoundType[0] == SOURCE_GHOST || sfBoundType[0] == GHOST;
}
float* Propagator::SmoothingVol(float *volin) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *volout = allocMem3d(nx, ny, nz, nThreads);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      for(int iz = 0; iz < nz; iz++) {
        int iz1 = max(iz - 1, 0);
        int iz2 = min(iz + 1, nz - 1);
        size_t id1 = size_t(iy * nx + ix) * nz + iz1;
        size_t id = size_t(iy * nx + ix) * nz + iz;
        size_t id2 = size_t(iy * nx + ix) * nz + iz2;
        volout[id] = 0.25 * volin[id1] + 0.5 * volin[id] + 0.25 * volin[id2];
      }
    }
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      int ix1 = max(ix - 1, 0);
      int ix2 = min(ix + 1, nx - 1);
      for(int iz = 0; iz < nz; iz++) {
        size_t id1 = size_t(iy * nx + ix1) * nz + iz;
        size_t id = size_t(iy * nx + ix) * nz + iz;
        size_t id2 = size_t(iy * nx + ix2) * nz + iz;
        volin[id] = 0.25 * volout[id1] + 0.5 * volout[id] + 0.25 * volout[id2];
      }
    }
  }
  // Specifically or velocity
  if(ny == 1) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id = size_t(iy * nx + ix) * nz + iz;
          volout[id] = volin[id];
        }
      }
    }
  } else {
    for(int iy = 0; iy < ny; iy++) {
      int iy1 = max(iy - 1, 0);
      int iy2 = min(iy + 1, ny - 1);
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id1 = size_t(iy1 * nx + ix) * nz + iz;
          size_t id = size_t(iy * nx + ix) * nz + iz;
          size_t id2 = size_t(iy2 * nx + ix) * nz + iz;
          volout[id] = 0.25 * volin[id1] + 0.5 * volin[id] + 0.25 * volin[id2];
        }
      }
    }
  }

  return volout;

}

float* Propagator::SqrtVol(float *volin) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *volout = allocMem3d(nx, ny, nz, nThreads);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      for(int iz = 0; iz < nz; iz++) {
        size_t id = size_t(iy * nx + ix) * nz + iz;
        volout[id] = sqrt(volin[id]);
      }
    }
  }

  return volout;
}
void Propagator::model_bnd_smooth(int ModelVolID, int nxbnd, int nybnd, int nzbnd) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;

  float *__restrict rho = volModel[ModelVolID];

  int nbndm = max(nxbnd, nybnd);
  nbndm = max(nbndm, nzbnd);

  vector<float> taper(nbndm);

  for(int iz = 0; iz < nzbnd; iz++) {

    taper[iz] = 0.5f * (1 - cosf((float) M_PI * (iz + 0.5) / (nzbnd - 1)));
  }


#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    for(int iz = 0; iz < nzbnd; iz++) {
      size_t id1 = ixy * nz + iz;
      size_t id2 = ixy * nz + nz - iz - 1;
      size_t id1b = ixy * nz + nzbnd - 1;
      size_t id2b = ixy * nz + nz - nzbnd;

      float tpr = taper[iz];
   rho[id1] = rho[id1] * tpr;
      rho[id2] = rho[id2] * tpr;
      //        rho[id1] = rho[id1b];
      //        rho[id2] = rho[id2b];

    }
  }

  for(int ix = 0; ix < nxbnd; ix++) {
    taper[ix] = 0.5f * (1 - cosf((float) M_PI * (ix + 0.5) / (nxbnd - 1)));
  }

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nxbnd; ix++) {
      float tpr = taper[ix];
      for(int iz = 0; iz < nz; iz++) {
        size_t id1 = size_t(iy * nx + ix) * nz + iz;
        size_t id2 = size_t(iy * nx + nx - ix - 1) * nz + iz;
        size_t id1b = size_t(iy * nx + nxbnd - 1) * nz + iz;
        size_t id2b = size_t(iy * nx + nx - nxbnd) * nz + iz;
rho[id1] = rho[id1] * tpr;
        rho[id2] = rho[id2] * tpr;
        //          rho[id1] = rho[id1b];
        //          rho[id2] = rho[id2b];

      }
    }
  }

  if(ny > 1) {
    for(int iy = 0; iy < nybnd; iy++) {
      taper[iy] = 0.5f * (1 - cosf((float) M_PI * (iy + 0.5) / (nybnd - 1)));
    }

#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < nybnd; iy++) {
      float tpr = taper[iy];
      for(int ix = 0; ix < nx; ix++) {
        for(int iz = 0; iz < nz; iz++) {
          size_t id1 = size_t(iy * nx + ix) * nz + iz;
          size_t id2 = size_t((ny - iy - 1) * nx + ix) * nz + iz;
          size_t id1b = size_t((nybnd - 1) * nx + ix) * nz + iz;
          size_t id2b = size_t((ny - nybnd) * nx + ix) * nz + iz;
rho[id1] = rho[id1] * tpr;
          rho[id2] = rho[id2] * tpr;
          //          rho[id1] = rho[id1b];
          //          rho[id2] = rho[id2b];

        }
      }
    }

  }

}
void Propagator::divideRho(Wavefield *wf) { // ATTN: check the vectorization
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;

  float *__restrict rho = volModel[RHO];
  float *__restrict wx = wf->wx;
  float *__restrict wy = wf->wy;
  float *__restrict wz = wf->wz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    for(int iz = 0; iz < nz; iz++) {
      size_t i = ixy * nz + iz;
      if(nx > 1) wx[i] /= rho[i];
      if(ny > 1) wy[i] /= rho[i];
      wz[i] /= rho[i];
    }
  }

}
void Propagator::multiplyRho(Wavefield *wf) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t nxy = (size_t)nx * ny;

  float *__restrict rho = volModel[RHO];
  float *__restrict wb = wf->wb;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    for(int iz = 0; iz < nz; iz++) {
      size_t i = ixy * nz + iz;
      wb[i] *= rho[i];
    }
  }
}
void Propagator::applyVel(Wavefield *myLocalWavefield) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict vel = volModel[VEL];
  size_t nxz = nx * nz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 v2 = _mm_load_ps(vel + i);
      __m128 vectx = _mm_load_ps(myLocalWavefield->wb + i);
      _mm_store_ps(myLocalWavefield->wb + i, _mm_mul_ps(vectx, v2));
    }
  }
}
void Propagator::applyReflectivity(Wavefield *mySrcWavefield, Wavefield *myRecWavefield) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict reflectivity = volModel[REFLECTIVITY];
  size_t nxz = nx * nz;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 vecref = _mm_load_ps(reflectivity + i);
      __m128 vecsrc = _mm_load_ps(mySrcWavefield->w0 + i); //changed from wb to w0 by wolf
      __m128 vecrec = _mm_load_ps(myRecWavefield->w0 + i); //changed from wb to w0 by wolf
      _mm_store_ps(myRecWavefield->w0 + i, _mm_add_ps(vecrec, _mm_mul_ps(vecsrc, vecref))); //changed from wb to w0 by wolf
    }
  }
}

void Propagator::update2nd(Wavefield *myLocalWavefield) {
  update2nd(myLocalWavefield->w0, myLocalWavefield->w1, myLocalWavefield->wb);
}
void Propagator::update2nd(float *__restrict w0, float *__restrict w1, float *__restrict wb) {
    printf("Start Propagator::update2nd\n");
    int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
    size_t nxz = nx * nz;
    if(!myModel->useQ)
    {
#pragma omp parallel for num_threads(nThreads) schedule(static)
        for(int iy = 0; iy < ny; iy++)
        {
            for(size_t ixz = 0; ixz < nxz; ixz += SSEsize)
            {
                size_t i = iy * nxz + ixz;
                __m128 ww0 = _mm_load_ps(w0 + i);
                __m128 ww1 = _mm_load_ps(w1 + i);
                //  __m128 wwb = _mm_load_ps(wb + i);
                //  wwb = _mm_add_ps(_mm_add_ps(ww1, ww1), _mm_sub_ps(wwb, ww0));
                //  _mm_store_ps(wb + i, wwb);
                __m128 wwb = _mm_sub_ps(_mm_add_ps(ww1, ww1), ww0);
                _mm_store_ps(w0 + i, wwb);
            }
        } //revised by wolf on Sep 21. 2022. IMPORTANT: This part is not consistent with original code, so it must be modified to go with different scenario.
    }
else
    {
        float *__restrict invQ = volModel[Q];
        Model *m = myModel;
#pragma omp parallel for num_threads(nThreads) schedule(static)
        for(int iy = 0; iy < ny; iy++)
        {
            for(size_t ixz = 0; ixz < nxz; ixz++)
            {
                size_t i = iy * nxz + ixz;
                w0[i] = w1[i] * (2 + invQ[i] * Q::cqsum) - w0[i];
            }
            for(int l = 0; l < Q::order; l++)
            {
                // ToOWL: this seems to be a bug to me
                float *__restrict d0 = &myWavefield->wq[myWavefield->iq0][l][0];
                // ToOWL: this seems to be a bug to me
                float *__restrict d1 = &myWavefield->wq[myWavefield->iq1][l][0];
                float cl = Q::cq[l], el = Q::wq[l];
                for(size_t ixz = 0; ixz < nxz; ixz++)
                {
                    size_t i = iy * nxz + ixz;
                    w0[i] += invQ[i] * ((1 + el) * d0[i] - d1[i] * 2);
                    d0[i] = cl * w1[i] + el * d0[i];
                }
            }
        }
    }
    printf("End Propagator::update2nd\n");
}
void Propagator::ApplyScaler(float *wb, float *wr) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float *__restrict vel = volModel[VEL];
  float *__restrict wrr = wr;
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(size_t ixz = 0; ixz < nxz; ixz += SSEsize) {
      size_t i = iy * nxz + ixz;
      __m128 scale = _mm_load_ps(wrr + i);
      __m128 v2 = _mm_load_ps(vel + i);
      __m128 v2scale = _mm_mul_ps(v2, scale);
      _mm_store_ps(wb + i, _mm_mul_ps(_mm_load_ps(wb + i), v2scale));
    }
  }
}
void Propagator::tapersource(float *ws, int nx, int ny, int nz, int ix0, int iy0, int iz0) {
  int tapersize = 5;
  int ix1 = max(0, (ix0 - tapersize));
  int ix2 = min((nx - 1), (ix0 + tapersize));
  int iy1 = max(0, (iy0 - tapersize));
  int iy2 = min((ny - 1), (iy0 + tapersize));
  int iz1 = max(0, (iz0 - tapersize));
  int iz2 = min((nz - 1), (iz0 + tapersize));
  for(int iy = iy1; iy <= iy2; iy++) {
    float yy = (iy - iy0) * (iy - iy0);
    for(int ix = ix1; ix <= ix2; ix++) {
      float xx = (ix - ix0) * (ix - ix0);
      for(int iz = iz1; iz <= iz2; iz++) {
        float zz = (iz - iz0) * (iz - iz0);
        float dist = sqrt(xx + yy + zz);
        if(dist > tapersize) continue;
        float weight = 0.5 * (1.0 - cosf((float) M_PI * dist / tapersize));
        size_t id = size_t(iy * nx + ix) * nz + iz;
        ws[id] *= weight;
      }
    }
  }
}
void Propagator::apply_symmetry(float *wf0, int sym) {

  int iz0 = myGrid->getIDz(zsurf);
  //    print1m("symmetry = %d, iz0=%d \n", sym, iz0);
  int iz2 = iz0 * 2; // symmetric point iz0 * 2
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  int tprlen = nzbnd;

  int taper_apply_symmetry = global_pars["taper_apply_symmetry"].as<int>(1);

  if(taper_apply_symmetry) {
    float constant = logf(0.001);
    vector<float> taper(tprlen);
    float a = 0.00;
    for(int iz = 0; iz < tprlen; iz++) {
      taper[iz] = 0.5f * (1 - cosf((float) M_PI * (iz + 0.5) / tprlen));
      //          print1m("iz=%d, taper=%f\n",iz, taper[iz]);

      /*
       float tmp = float ( iz) / float (tprlen );
       float a3 = 10*(1-a);
       float a4 = -15*(1-a);
       float a5 = 6*(1-a);
       taper[iz] = a + (a3 +a4*tmp+ a5*tmp*tmp)* tmp*tmp*tmp;
       */
    }
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        float *__restrict wf = wf0 + ((size_t)iy * nx + ix) * nz;
        /*
         for (int iz = 0; iz < nzbnd; iz++) {
         float f = taper[iz];
         wf[iz] =  sym * wf[iz2 - iz] * f;
         }
         */
        for(int iz = tprlen; iz < iz0; iz++) {
          wf[iz] = sym * wf[iz2 - iz];
        }
        if(sym == -1) wf[iz0] = 0;
        for(int iz = 0; iz < tprlen; iz++) {
          float f = taper[iz];
          wf[iz] = sym
              * (wf[iz2 - iz] * f
                  + (1 - f) * (wf[iz2 - iz - 2] + wf[iz2 - iz + 2] + 4 * (wf[iz2 - iz - 1] + wf[iz2 - iz + 1]) + 6 * wf[iz2 - iz]) / 16.)
              * f;
        }
      }
    }
  } else {
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        float *__restrict wf = wf0 + ((size_t)iy * nx + ix) * nz;
        for(int iz = 0; iz < iz0; iz++)
          wf[iz] = sym * wf[iz2 - iz];
        if(sym == -1) wf[iz0] = 0;
        wf[0] = 0.5 * (wf[0] + wf[1]);
      }
    }
  }
}
void Propagator::boundary_attenuation(float *wf, float *wf0, float dt) {

  int iz0 = myGrid->getIDz(zsurf);
  //    print1m("symmetry = %d, iz0=%d \n", sym, iz0);
  int iz2 = iz0 * 2;  // symmetric point iz0 * 2
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  int tprlen = nzbnd;

  vector<float> taper(tprlen);
  float a = 0.00;
  for(int iz = 0; iz < tprlen; iz++) {
    float d = (tprlen - iz) / ((float)tprlen);
    taper[iz] = 0.05f * d * d * exp(1.0 - 1.0 / d);
  }
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      float *__restrict pf0 = wf0 + ((size_t)iy * nx + ix) * nz;
      float *__restrict pf = wf + ((size_t)iy * nx + ix) * nz;
      for(int iz = 0; iz < tprlen; iz++) {
        float f = taper[iz];
        pf[iz] = (pf[iz] - pf0[iz] * f) / (1 + f);
      }
    }
  }
}
float Propagator::getDiskSize(int ntsav) {
  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  size_t mysize = WaveFieldCompress::nshort_pack(nz) * nx * ny;
  float diskSize = (float)mysize * sizeof(ushort) * float(ntsav) / (1024.0f) / (1024.f) / (1024.0f);
  return 2.0f * diskSize;
}

vector<float> Propagator::getIcLiveFull(float tmax) {
  float tmutePerc = global_pars["tmutePerc"].as<float>(0.1);
  float waterDepth = global_pars["waterDepth"].as<float>(0.0);
  float vmin = global_pars["vMin"].as<float>(1492);
  float tmute = max(waterDepth / vmin, tmutePerc * tmax);

  float tmin0 = 0.0f, tmin1 = tmute;
  float t0 = it0 * dtpro;
  if(global_pars["ic_live"]) {
    tmin0 = -t0 + global_pars["ic_live"].as<float>();
    tmin1 = -t0 + global_pars["ic_full"].as<float>(tmin0 + t0 + tmute);
  } else if(global_pars["ic_full"]) {
    tmin1 = -t0 + global_pars["ic_full"].as<float>();
    tmin0 = -t0 + max(0.0f, global_pars["ic_live"].as<float>(tmin1 + t0 - tmute));
  } else if(global_pars["ic_live_relative"]) {
    tmin0 = global_pars["ic_live_relative"].as<float>();
    tmin1 = global_pars["ic_full_relative"].as<float>(tmin0 + tmute);
  } else if(global_pars["ic_full_relative"]) {
    tmin1 = global_pars["ic_full_relative"].as<float>();
    tmin0 = max(-t0, global_pars["ic_live_relative"].as<float>(tmin1 - tmute));
  }
  return vector<float> { tmin0, tmin1 };
}
vector<float> Propagator::sourceDelayUpdateLoc(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID,
    const Traces *traces) {
  vector<float> sourceXmod(1, 0.0f);
  vector<float> sourceYmod(1, 0.0f);
  vector<float> sourceZmod(1, 0.0f);
  vector<float> sourceDelay(1, 0.0f);

  if(global_pars["RTMInputFile"]) {
    string fileName = expEnvVars(global_pars["RTMInputFile"].as<string>());
    RecordUtilities::getRecordSourceLoc(fileName.c_str(), sourceX[0], sourceY[0], sourceZ[0], sourceID);
    print1m("Source location (from trace header): sourceX=%g sourceY=%g sourceZ=%g .\n", sourceX[0], sourceY[0], sourceZ[0]);
  }
//override by user input
  if(global_pars["sourceID"]) {
    sourceID = global_pars["sourceID"].as<int>();
    print1m("Source location (overridden by user): sourceID=%d\n", sourceID);
  }

  int nsrc = 1;
  if(global_pars["sourceX"]) {
    assertion(global_pars["sourceX"].IsScalar(), "Please convert sourceX from array to string before save to yaml!");
    sourceX = str2floats(global_pars["sourceX"].as<string>());
    nsrc = max(nsrc, (int)sourceX.size());
    print1m("Source location (overridden by user): sourceX=%s\n", floats2str(sourceX).c_str());
  }
  if(global_pars["sourceY"]) {
    sourceY = str2floats(global_pars["sourceY"].as<string>());
    nsrc = max(nsrc, (int)sourceY.size());
    print1m("Source location (overridden by user): sourceY=%s\n", floats2str(sourceY).c_str());
  }
  if(global_pars["sourceZ"]) {
    sourceZ = str2floats(global_pars["sourceZ"].as<string>());
    nsrc = max(nsrc, (int)sourceZ.size());
    print1m("Source location (overridden by user): sourceZ=%s\n", floats2str(sourceZ).c_str());
  }
  if(global_pars["sourceXmod"]) {
    sourceXmod = str2floats(global_pars["sourceXmod"].as<string>());
    nsrc = max(nsrc, (int)sourceXmod.size());
    print1m("Additional modification: sourceXmod=%s\n", floats2str(sourceXmod).c_str());
  }
  if(global_pars["sourceYmod"]) {
    sourceYmod = str2floats(global_pars["sourceYmod"].as<string>());
    nsrc = max(nsrc, (int)sourceYmod.size());
    print1m("Additional modification: sourceYmod=%s\n", floats2str(sourceYmod).c_str());
  }
  if(global_pars["sourceZmod"]) {
sourceZmod = str2floats(global_pars["sourceZmod"].as<string>());
    nsrc = max(nsrc, (int)sourceZmod.size());
    print1m("Additional modification: sourceZmod=%s\n", floats2str(sourceZmod).c_str());
  }
  if(global_pars["sourceDelay"]) {
    sourceDelay = str2floats(global_pars["sourceDelay"].as<string>());
    nsrc = max(nsrc, (int)sourceDelay.size());
    print1m("Source delay (overridden by user): sourceDelay=%s\n", floats2str(sourceDelay).c_str());
  }
  Util::pad_vector(sourceX, nsrc);
  Util::pad_vector(sourceY, nsrc);
  Util::pad_vector(sourceZ, nsrc);
  Util::pad_vector(sourceDelay, nsrc);
  Util::pad_vector(sourceXmod, nsrc, 0);
  Util::pad_vector(sourceYmod, nsrc, 0);
  Util::pad_vector(sourceZmod, nsrc, 0);
  for(int i = 0; i < nsrc; i++) {
    sourceX[i] += sourceXmod[i];
    sourceY[i] += sourceYmod[i];
    sourceZ[i] += sourceZmod[i];
  }

  {   // for writing to SEGY, so only the first value is needed. TODO: also for IC cone mute, shall we handle multiple sources?
    global_pars["_sourceID"] = sourceID;
    global_pars["_sourceX"] = sourceX[0];
    global_pars["_sourceY"] = sourceY[0];
    global_pars["_sourceZ"] = sourceZ[0];
  }

  return sourceDelay;
}
// static method, moved from mod.cpp
unique_ptr<ExtractReceiverData> Propagator::mod(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID,
    vector3 &recvMin, vector3 &recvMax, Traces *traces0) {
  bool traces_was_null = traces0 == nullptr;
  unique_ptr<Traces> traces(traces0);
  int nThreads = init_num_threads();

  // 0. check demig or modeling
  bool isDemig = bool(global_pars["global"]["reflectivity"]);

  // 1. get source location
  vector<float> sourceDelay = sourceDelayUpdateLoc(sourceX, sourceY, sourceZ, sourceID, traces0);

  // 2. define the aperture
  assertion(global_pars["zMax"].IsDefined(), "zMax parameter is required in jobdeck!");
  assertion(global_pars["maxFreq"].IsDefined(), "maxFreq parameter is required in jobdeck!");
  float zmin = 0; // hack, shift these to zero (also shift source/receiver elevations
  float zmax = global_pars["zMax"].as<float>() - global_pars["zMin"].as<float>(0.0f);
  float mfreq = global_pars["maxFreq"].as<float>();
  print1m("zMax=%g maxFreq=%g \n", zmax, mfreq);

  // 4 get propagate time
  assertion(global_pars["tmax"].IsDefined(), "tmax keyword is required for forward modeling!");
  float tmax0 = global_pars["tmax"].as<float>();

  Propagator *pProp = new Propagator(zmin, zmax, mfreq, tmax0);

  float xMinValid = FLT_MAX, yMinValid = FLT_MAX;
  float xMaxValid = -FLT_MAX, yMaxValid = -FLT_MAX;
 DomainRange::getComputingRange(sourceX, sourceY, xMinValid, xMaxValid, yMinValid, yMaxValid, recvMin, recvMax, pProp->dim,
                                 traces_was_null);
  print1m("xMinValid=%f, xMaxValid=%f, yMinValid=%f, yMaxValid=%f, recvZmin=%f, recvZmax=%f \n", xMinValid, xMaxValid, yMinValid, yMaxValid,
          recvMin.z, recvMax.z);

  //override by user input
  if(global_pars["receiverZmin"] || global_pars["receiverZmax"]) {
    recvMin.z = global_pars["receiverZmin"].as<float>(recvMin.z);
    recvMax.z = global_pars["receiverZmax"].as<float>(recvMax.z);
    print1m("User overrides: recvZmin=%f, recvZmax=%f \n", recvMin.z, recvMax.z);
  }

  int bndType = Source::getSurfaceType(PROP::FORWARD);

  // 5. set up wave propagator
  pProp->prepare(xMinValid, xMaxValid, yMinValid, yMaxValid, recvMin, recvMax);

  int da_removal = global_pars["da_removal"].as<int>(0);
  assertion(!(da_removal && isDemig), "da_removal is not needed for demigration!");

//  int fd_order = FdEngine_fd::getOrder();//by wolf
  int srcSpread = global_pars["sourceSpreadSize"].as<int>(30); // modeling
  int spreadSize = global_pars["receiverSpreadSize"].as<int>(30);
  global_pars["receiverSpreadSize"] = spreadSize; // unify the defaults for later usage

  // if demigtation, rotate -90 degree. now computed inside Source.cpp

  bool mmod = !traces_was_null && traces->nt > 0;
  if(mmod) traces = pProp->setupReceiver(spreadSize, 0, move(traces), true);
  else pProp->setupReceiver(PROP::MOD, srcSpread, 0, sourceX, sourceY, sourceZ, sourceDelay);

  float recordingFreq = global_pars["modRecordFreq"].as<float>(mfreq);
  pProp->dtsav = pProp->getDtSave(recordingFreq);
  int ntratio = nearbyintf(pProp->dtsav / pProp->dtpro);
  float t0_save = mmod ? 0 : -(pProp->myshot->it0 / ntratio) * pProp->dtsav;
  print1m("mod: dtpro=%f, dtNyquist(from modRecordFreq)=%f, ntratio=%d, dtsave=%f\n", pProp->dtpro, 0.5f / recordingFreq, ntratio,
          pProp->dtsav);
  int ntsav = (int)ceil((tmax0 - t0_save) / pProp->dtsav) + 1;

  // 5.5. do modelling propagation
  //Receiver building
  auto erd = std::unique_ptr<ExtractReceiverData>(new ExtractReceiverData(pProp->myGrid, ntsav, pProp->dtsav, t0_save, bndType, nThreads));

  erd->receivers = pProp->setupReceiverForModeling(spreadSize, move(traces));
  erd->specifyReceivers();

  // 6. forward propagation
  string waveFile = expEnvVars(global_pars["wavefieldCachePrefix"].as<string>("/tmp/wavefield"));

  if(isDemig) pProp->demigration(erd, PROP::FORWARD);
  else pProp->modelling(erd, PROP::FORWARD);

  if(da_removal) {
    erd->saveToRecData();
    pProp->populateModelDARemval();
    pProp->modelling(erd, PROP::FORWARD);
  }

  //free memory used in wave propagation
  pProp->freeVolume(0);

  // 7. save to output
  if(traces_was_null) {
    string recFile = expEnvVars(global_pars["recFile"].as<string>("./data.js"));
    erd->saveRecFile(recFile.c_str());
  }
delete pProp; // erd was using receivers[1] allocated from pProp, so do not delete before erd->saveRecFile

  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  printf((isDemig ? "[%s] %s > Demigration finished. \n" : "[%s] %s > Modelling finished. MemFree: %s\n"), hostname, time_now().c_str(),
         libCommon::Utl::free_memory().c_str());
  MpiPrint::printed1 = 1;

  return erd;
}
shared_ptr<Grid> Propagator::rtm(vector<float> &sourceX, vector<float> &sourceY, vector<float> &sourceZ, int sourceID, vector3 &recvMin,
    vector3 &recvMax, vector<Traces*> vec_traces, vector<float*> destImages, OutputGrid *destGrid) {
  // 1. get source location
  int ncapIn = vec_traces.size();
  vector<float> sourceDelay = Propagator::sourceDelayUpdateLoc(sourceX, sourceY, sourceZ, sourceID, vec_traces[ncapIn - 1]);

  // 2. define the aperture
  assertion(global_pars["zMax"].IsDefined(), "zMax parameter is required in jobdeck!");
  assertion(global_pars["maxFreq"].IsDefined(), "maxFreq parameter is required in jobdeck!");
  float zmin = 0; // hack, shift these to zero (also shift source/receiver elevations
  float zmax = global_pars["zMax"].as<float>() - global_pars["zMin"].as<float>(0.0f);
  float mfreq = global_pars["maxFreq"].as<float>();
  // mfreq          /=0.9f;
  print1m("zMax=%g maxFreq=%g \n", zmax, mfreq);

  // 4. get the receiver grid and propagate time
  //     4b. get propagate time
  float tmax0 = global_pars["tmax"].as<float>();
  float tmaxratio = global_pars["tmaxratio"].as<float>(0.7f);
  if(tmaxratio > 1.0f) tmaxratio = 1.0f;
  print1m("tmax=%f, tmaxratio=%f \n", tmax0, tmaxratio);

  // 5. set up wave propagator
  Propagator myprop(zmin, zmax, mfreq, tmax0);

  float xMinValid = FLT_MAX, yMinValid = FLT_MAX;
  float xMaxValid = -FLT_MAX, yMaxValid = -FLT_MAX;

  DomainRange::getComputingRange(sourceX, sourceY, xMinValid, xMaxValid, yMinValid, yMaxValid, recvMin, recvMax, myprop.dim,
                                 vec_traces[0] == nullptr);  //revised from DomainRangeRTM to DomainRange
  print1m("xMinValid=%f, xMaxValid=%f, yMinValid=%f, yMaxValid=%f, recvZmin=%f, recvZmax=%f \n", xMinValid, xMaxValid, yMinValid, yMaxValid,
          recvMin.z, recvMax.z);
 //override by user input
  if(global_pars["receiverZ0"]) {
    recvMin.z = recvMax.z = global_pars["receiverZ0"].as<float>();
    print1m("User overrides: recvZmin=recvZmax=receiverZ0=%g\n", recvMin.z);
  }

  myprop.prepare(xMinValid, xMaxValid, yMinValid, yMaxValid, recvMin, recvMax);

  int srcSpread = global_pars["sourceSpreadSize"].as<int>(10); // RTM
  int spreadSize = global_pars["receiverSpreadSize"].as<int>(10);
  global_pars["receiverSpreadSize"] = spreadSize; // unify the defaults for later usage
  if(ncapIn == 1) myprop.setupReceiver(PROP::RTM, srcSpread, 0, sourceX, sourceY, sourceZ, sourceDelay); // it0 is now updated
  else {
    timeRecorder.start(RECEIVER_TIME);
    myprop.setupReceiver(spreadSize, 0, unique_ptr<Traces> { vec_traces[0] });
    timeRecorder.end(RECEIVER_TIME);
  }

  vector<float> tlivefull = myprop.getIcLiveFull(tmax0 * tmaxratio);
  myprop.dtsav = myprop.getDtSave(mfreq);
  int ntratio = nearbyintf(myprop.dtsav / myprop.dtpro);
  print1m("dt_pro=%f, nprop_per_ic=%d, dt_ic=%f\n", myprop.dtpro, ntratio, myprop.dtsav);
  int ntsav = nearbyintf((tmax0 * tmaxratio - tlivefull[0]) / myprop.dtsav);

  float diskSize = myprop.getDiskSize(ntsav);
  print1m("Cache needed for this shot: %7.2f GB (ntsave=%d), MemFree: %s\n", diskSize, ntsav, libCommon::Utl::free_memory().c_str());

#if 1
// 6. forward propagation
  string waveFile = expEnvVars(global_pars["wavefieldCachePrefix"].as<string>("/tmp/wavefield"));
  CacheFile shotwaves(myprop.myGrid->nx, myprop.myGrid->ny, myprop.myGrid->nz, ntsav,
  SEQ_ZXYT,
                      waveFile.c_str());
  shotwaves.setparas(myprop.myGrid->x0, myprop.myGrid->y0, myprop.myGrid->z0, tlivefull[0], tlivefull[1], myprop.myGrid->dx,
                     myprop.myGrid->dy, myprop.myGrid->dz, myprop.dtsav);
  shotwaves.open();

  timeRecorder.start(FORWARD_TIME);
  if(!global_pars["qc_receiver_only"].as<int>(0)) myprop.migration(&shotwaves, PROP::FORWARD);
  timeRecorder.end(FORWARD_TIME);
#endif
  myprop.receivers[0].reset(); // free vol4d (and FWD/BWD uses same filename so far)



#if 1
  // 7. backward propagation
  myprop.reprepare();// added by wolf on Nov 7, 2022
  timeRecorder.start(RECEIVER_TIME);
  myprop.setupReceiver(spreadSize, 1, unique_ptr<Traces> { vec_traces[ncapIn - 1] }); // do it before new CacheFile, receiver traces are free'd after this
  timeRecorder.end(RECEIVER_TIME);

  CacheFile recvwaves(myprop.myGrid->nx, myprop.myGrid->ny, myprop.myGrid->nz, ntsav,
  SEQ_ZXYT,
                      waveFile.c_str());
  recvwaves.setparas(myprop.myGrid->x0, myprop.myGrid->y0, myprop.myGrid->z0, tlivefull[0], tlivefull[1], myprop.myGrid->dx,
                     myprop.myGrid->dy, myprop.myGrid->dz, myprop.dtsav);
  recvwaves.open();

  timeRecorder.start(BACKWARD_TIME);
  myprop.migration(&recvwaves, PROP::BACKWARD);
timeRecorder.end(BACKWARD_TIME);

#endif

#if 0
  std::string srcFile("wavefield_p4202_g0_0.cache");
  std::string recFile("wavefield_p4202_g0_1.cache");
  CacheFile *shotwaves = new CacheFile(myprop.nx, myprop.ny, myprop.nz, ntsav,
                                       SEQ_ZXYT, srcFile);
  CacheFile *recvwaves = new CacheFile(myprop.nx, myprop.ny, myprop.nz, ntsav,
                                       SEQ_ZXYT, recFile);
#endif

  // 8. image condition
  myprop.ic_prepare();
  myprop.myWavefield->deallocateMemory();
  ImagingCondition ic(&shotwaves, &recvwaves, STACK);

  ic.getVel(myprop.volModel[VEL], myprop.dtpro);
  //free memory used in wave propagation, after this myprop.volModel is no longer usable
  myprop.freeVolume(1);

  timeRecorder.start(IMAGING_TIME);
  shared_ptr<Grid> grid = ic.genImage(myprop.myGrid.get(), myprop.wfComp, xMinValid, xMaxValid, yMinValid, yMaxValid, myprop.dtsav,
                                      destImages, destGrid);
  timeRecorder.end(IMAGING_TIME);

  MpiPrint::printed1 = 1;

  return grid;
}
void Propagator::qcVel(string ext) {
  if(!getBool(global_pars["snapshotVel"], false)) return;

  int nx = myGrid->nx, ny = myGrid->ny, nz = myGrid->nz;
  float invdt = 1 / dtpro;

  vector<string> vext0 { "vel", "eps", "del", "dipx", "dipy" }, vext;
  vector<int> idxs0 { VEL, EPS, DEL, PJX, PJY }, idxs;
  int nmodel = 1;

  if(myModel->modeltype == VTI) nmodel = 3;
  else if(myModel->modeltype == TTI) nmodel = ny > 1 ? 5 : 4;

  for(int j = 0; j < nmodel; j++)
    vext.push_back(vext0[j]), idxs.push_back(idxs0[j]);
  if(myModel->useRho) vext.push_back("rho"), idxs.push_back(RHO);
  if(myModel->useReflectivity) vext.push_back("refl"), idxs.push_back(REFLECTIVITY);
  int nmodel_ext = vext.size();

  vector<float> vel((size_t)nz * max(nx, ny));

  int ix0 = nearbyintf(global_pars["_IDxf"].as<float>(0));
  int iy0 = nearbyintf(global_pars["_IDyf"].as<float>(0));
  ix0 = min(nx - 1, max(0, ix0));
  iy0 = min(ny - 1, max(0, iy0));
  string snapShotPrefix = expEnvVars(global_pars["snapShotPrefix"].as<string>());

  for(int j = 0; j < nmodel_ext; j++) {
    int idx = idxs[j];
    string fnameV = snapShotPrefix + "_" + vext[j] + ext;
    string fnameV2 = snapShotPrefix + "_" + vext[j] + "_y" + ext;
// iy = iy0 slice
    for(int ix = 0; ix < nx; ix++) {
      size_t off0 = ((size_t)iy0 * nx + ix) * nz;
      size_t off = (size_t)ix * nz;
      for(int iz = 0; iz < nz; iz++, off0++, off++) {
        if(idx == VEL) vel[off] = sqrtf(volModel[idx][off0]) * invdt;
        else vel[off] = volModel[idx][off0];
      }
    }
    jseisUtil::save_zxy(fnameV.c_str(), &vel[0], nz, nx, 1, myGrid->dz, myGrid->dx, myGrid->dy, -myGrid->getIDz(0.0), // purposely not getIDz(zsurf)
                        global_pars["_IDxf"].as<float>(0) * myGrid->dx, 0, ix0);

    if(ny > 1) {
      for(int iy = 0; iy < ny; iy++) {
        size_t off0 = ((size_t)iy * nx + ix0) * nz;
        size_t off = (size_t)iy * nz;
        for(int iz = 0; iz < nz; iz++, off0++, off++) {
          if(idx == VEL) vel[off] = sqrtf(volModel[idx][off0]) * invdt;
          else vel[off] = volModel[idx][off0];
        }
      }
      jseisUtil::save_zxy(fnameV2.c_str(), &vel[0], nz, ny, 1, myGrid->dz, myGrid->dy, myGrid->dx, -myGrid->getIDz(0.0), // purposely not getIDz(zsurf)
                          global_pars["_IDyf"].as<float>(0) * myGrid->dy, 0, iy0);
    }
  }
}



