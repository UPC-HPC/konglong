/*
 * ModelRegrid.cpp
 *
 */

#include <xmmintrin.h>    // defines SSE intrinsics
#include <sys/times.h>
#include <sys/utsname.h>
#include <time.h>
#include <omp.h>
#include <unistd.h>
#include <cfloat>
#include <stdlib.h>
#include <mpi.h>
#include "stdio.h"
#include "Params.h"
#include "fdm.hpp"
#include "GetPar.h"
#include "ModelLoader.h"
#include "Profile.h"
#include "libFFTV/numbertype.h"
#include "libCommon/padfft.h"
#include "Grid.h"
#include "Geometry.h"
#include "ModelPrepare.h"
#include "ModelRegrid.h"
#include "Q.h"
#include "FdEngine_cfd.h"
#include "FdEngine_fd.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

ModelRegrid::ModelRegrid() {

}

ModelRegrid::~ModelRegrid() {

}

void ModelRegrid::setPath(const char *path) { // end with '/' if not empty
  int l = strlen(path);
  this->path = path;
  if(l > 0 && path[l - 1] != '/') this->path += '/';
}

void ModelRegrid::prepModel(MigParams *migParams, NodeParams *nodeParams) {

  Model *myModel = new Model();
  int nThreads = global_pars["nThreads"].as<int>(omp_get_max_threads());

  //getLowCutFilterType();

  //get grid type
  int gtype = IRREGULAR;
  string gridType = global_pars["gridType"].as<string>("IRREGULAR");
  transform(gridType.begin(), gridType.end(), gridType.begin(), ::toupper);

  if(gridType.compare("RECTANGLE") == 0) {
    gtype = RECTANGLE;
  } else if(gridType.compare("IRREGULAR") == 0) {
    gtype = IRREGULAR;
  } else if(gridType.compare("XPYRAMID") == 0) {
    gtype = XPYRAMID;
  } else if(gridType.compare("YPYRAMID") == 0) {
    gtype = YPYRAMID;
  } else if(gridType.compare("XYPYRAMID") == 0) {
    gtype = XYPYRAMID;
  } else {
    print1m("Unknown grid type! gridType=%s \n", gridType.c_str());
    exit(-1);
  }
  //print1m("gridType=%d \n", gtype);

  assertion(global_pars["zMax"].IsDefined(), "zMax parameter is required in jobdeck!");
  assertion(global_pars["maxFreq"].IsDefined(), "maxFreq parameter is required in jobdeck!");
  float zmin = 0; // hack, shift these to zero (also shift source/receiver elevations
  float zmax = global_pars["zMax"].as<float>() - global_pars["zMin"].as<float>(0.0f);
  float mfreq = global_pars["maxFreq"].as<float>();
  print1m("zMin=%g, zMax=%g, maxFreq=%g \n", zmin, zmax, mfreq);

  ModelLoader modelLoader(*myModel, "global", nThreads);

  float dispersion = FdEngine::getFdDispersion();
  float khpass = global_pars["src_apply_khpass"].as<float>(dispersion <= 0.8f ? dispersion * 0.5f : 0.40f);
  float khcut = global_pars["src_apply_khcut"].as<float>(dispersion <= 0.8f ? dispersion * 0.5f : 0.48f);
  global_pars["src_apply_khpass"] = khpass; // save it, the above is the only place to determine these values
  global_pars["src_apply_khcut"] = khcut;

  if(nodeParams == NULL || nodeParams->mpiRank == 0) {

    modelLoader.loadModelVolume(VEL);

    Profile *vprof = new Profile(myModel->fdms[VEL], mfreq);

    float velmin = vprof->getVmin(zmin, zmax);


    bool do_dual_flood = getBool(global_pars["dual_flood"], false);
    if (do_dual_flood) {
      modelLoader.loadModelVolume(VEL2);
      Profile *vprof2 = new Profile(myModel->fdms[VEL2], mfreq);
      float velmin2 = vprof2->getVmin(zmin, zmax);
      if (velmin2 < velmin) {
        velmin = velmin2;
        delete vprof;
        vprof = vprof2;
      } else {
        delete vprof2;
      }
    } //In case of irregular gridtype, set the myGrid based on the velocity file from which the minimum velocity comes.
    if(!global_pars["vMin"]) {
      global_pars["vMin"] = velmin;
    }
    // by wolf on Nov 7, 2022

    float xyGridFactor = global_pars["xyGridFactor"].as<float>(1.5) * dispersion;
    float zGridFactor = global_pars["zGridFactor"].as<float>(1.0) * dispersion;

    float half_wavelength = velmin / (2.0f * mfreq);
    float gdz = half_wavelength * zGridFactor;
    float gdx = half_wavelength * xyGridFactor;
    float gdy = half_wavelength * xyGridFactor;

    gdz = global_pars["prop"]["dz"].as<float>(gdz);
    gdx = global_pars["prop"]["dx"].as<float>(gdx);
    gdy = global_pars["prop"]["dy"].as<float>(gdy);

    bool isDerive2nd = FdEngine::isDerive2nd();
    // pure iso engine needs more padding, finer grid too (should be relative to wavelength)
    // these values are to be sent to slave in geom->header
    int nxbnd = global_pars["nxBoundary"].as<int>(nearbyintf((isDerive2nd ? 30 : 10) * half_wavelength / gdx));
    int nybnd = global_pars["nyBoundary"].as<int>(nearbyintf((isDerive2nd ? 30 : 10) * half_wavelength / gdy));
    int nzbnd0 = global_pars["nzBoundary"].as<int>(nearbyintf((isDerive2nd ? 60 : 20) * half_wavelength / gdz));
    int nzuppad = global_pars["nzUpPad"].as<int>(10 * half_wavelength / gdz);

    //
    int nzt = ZnumGrid(vprof, zmin, zmax, gdz);
    if(gtype == RECTANGLE) nzt = nearbyintf((zmax - zmin) / gdz) + 1;
    nzt += nzuppad;
    int nzo = nzt + 2 * nzbnd0;
    int nzu = libCommon::padfft(nzo);
    int nzbnd = (nzu - nzt) / 2;
    static int printed;
    if(!printed)
      print1m("### HALF_WAVELENGTH: %f, dispersion=%f, gdz=%f, gdx=%f\n"
                        "nzBoundary: %d, nzUpPad: %d, final nzbnd: %d\n",
                        half_wavelength, dispersion, gdz, gdx, nzbnd0, nzuppad, nzbnd), printed = 1;

    //
    float xVelStart = modelLoader.getXMinValid();
    float yVelStart = modelLoader.getYMinValid();
    float xVelEnd = modelLoader.getXMaxValid();
    float yVelEnd = modelLoader.getYMaxValid();

    int nxu = int((xVelEnd - xVelStart) / gdx + 0.5f) + 1;
    int nyu = int((yVelEnd - yVelStart) / gdy + 0.5f) + 1;

    Grid *myGrid = new Grid(gtype, nxu, nyu, nzu, gdx, gdy, gdz, zmin, zmax, nThreads);

    myGrid->setupGrid(vprof, nzbnd, nzuppad);

    myGrid->setOrigin(xVelStart, yVelStart);

    //preprocess models
    ModelPrepare *prep = new ModelPrepare(myGrid, myModel, &modelLoader, nThreads);

    //prepare  velocity
    string velFdmFile = expEnvVars(global_pars["local"]["vel"]["file"].as<string>(VEL_LOCAL_DEFAULT));
    velFdmFile = path + velFdmFile;
    if(gtype == RECTANGLE || gtype == IRREGULAR) {
      prep->velPrepare(velFdmFile);
    } else {
      prep->velPrepare_backup(velFdmFile);
    }

    if (do_dual_flood){
      string vel2FdmFile = expEnvVars(global_pars["local"]["vel_dual"]["file"].as<string>(VEL2_LOCAL_DEFAULT));
          vel2FdmFile = path + vel2FdmFile;
          if(gtype == RECTANGLE || gtype == IRREGULAR) {
            prep->vel2Prepare(vel2FdmFile);
          } else {
            prep->vel2Prepare_backup(vel2FdmFile);
          }
    }

    if(global_pars["global"]["rho"]) {
      string rhoFdmFile = expEnvVars(global_pars["local"]["rho"]["file"].as<string>(RHO_LOCAL_DEFAULT));
      rhoFdmFile = path + rhoFdmFile;
      if(gtype == RECTANGLE || gtype == IRREGULAR) {
        prep->rhoPrepare(rhoFdmFile);
      } else {
        prep->rhoPrepare_backup(rhoFdmFile);
      }
    }

    if(global_pars["global"]["reflectivity"]) {
      string reflectivityFdmFile = expEnvVars(global_pars["local"]["reflectivity"]["file"].as<string>(REFLECTIVITY_LOCAL_DEFAULT));
      reflectivityFdmFile = path + reflectivityFdmFile;
      prep->reflectivityPrepare(reflectivityFdmFile);
    }

    if(global_pars["global"]["1/Q"]) {
      string qFdmFile = expEnvVars(global_pars["local"]["1/Q"]["file"].as<string>(Q_LOCAL_DEFAULT));
      qFdmFile = path + qFdmFile;
      assertion(gtype == RECTANGLE || gtype == IRREGULAR, "Q option is only implemented for RECTANGLE or IRREGULAR!");
      prep->qPrepare(qFdmFile);
    }

    //load and prepare eps and del
    if(myModel->modeltype == VTI || myModel->modeltype == TTI) {
      string epsFdmFile = expEnvVars(global_pars["local"]["epsilon"]["file"].as<string>(EPS_LOCAL_DEFAULT));
      string delFdmFile = expEnvVars(global_pars["local"]["delta"]["file"].as<string>(DEL_LOCAL_DEFAULT));
      epsFdmFile = path + epsFdmFile;
      delFdmFile = path + delFdmFile;
      if(gtype == RECTANGLE || gtype == IRREGULAR) {
        prep->vtiPrepare(epsFdmFile, delFdmFile);
      } else {
        prep->vtiPrepare_backup(epsFdmFile, delFdmFile);
      }
    }

    //load and prepare pjx and pjy
    if(myModel->modeltype == TTI) {
      string pjxFdmFile = expEnvVars(global_pars["local"]["pjx"]["file"].as<string>(PJX_LOCAL_DEFAULT));
      string pjyFdmFile = expEnvVars(global_pars["local"]["pjy"]["file"].as<string>(PJY_LOCAL_DEFAULT));
      pjxFdmFile = path + pjxFdmFile;
      pjyFdmFile = path + pjyFdmFile;
      if(gtype == RECTANGLE || gtype == IRREGULAR) {
        prep->ttiPrepare(pjxFdmFile, pjyFdmFile);
      } else {
        prep->ttiPrepare_backup(pjxFdmFile, pjyFdmFile);
      }

    }

    //calculate dt and as a parameter
    migParams->dt = prep->calcDeltaTime();

    //save the geometry and profile
    string geomFile = expEnvVars(global_pars["local"]["geometry"]["file"].as<string>(GEOM_LOCAL_DEFAULT));
    geomFile = path + geomFile;
    Geometry *geom = new Geometry();
    geom->setHeader(myGrid->x0, myGrid->y0, myGrid->z0, myGrid->nx, myGrid->ny, myGrid->nz, myGrid->dx, myGrid->dy, myGrid->dz, nzbnd,
                    nzuppad, nxbnd, nybnd, gtype);

    if(gtype != RECTANGLE) {
      geom->setZGrid(&(myGrid->zgrid[0]));
      geom->setDzGrid(&(myGrid->dzgrid[0]));
    }

    geom->write(geomFile);

    delete myGrid;
    delete prep;
    delete geom;
    delete vprof;
  }
  ////////////////////////////////////
  migParams->modType = myModel->modeltype;
  migParams->gridType = gtype;

  ///////////////////////////////////
  //print1m("model_prep done! \n");

  delete myModel;
}

void ModelRegrid::broadcastGrid(MigParams *migParams, NodeParams *nodeParams) {

  //struct timeval t1, t2;
  int geomhdr_floatsize = sizeof(GeomHeader) / sizeof(float);
  if(nodeParams->mpiRank == 0) {
    //gettimeofday(&t1, NULL);
    string geomFile = expEnvVars(global_pars["local"]["geometry"]["file"].as<string>(GEOM_LOCAL_DEFAULT));
    geomFile = path + geomFile;
    Geometry *geom = new Geometry();
    geom->read(geomFile);
    //float* zgrid = & (geom->zgrid[0]);

        MPI_Bcast(geom->header, geomhdr_floatsize, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(geom->header->gridType != RECTANGLE) {
            MPI_Bcast(&(geom->zgrid[0]), geom->header->nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&(geom->zgrid[0]), geom->header->nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    delete geom;

    //broadcast dt
        MPI_Bcast(&migParams->dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);


    print1m("  Geometry send completed!\n");
  } else if(nodeParams->superslave) {
    string geomFile = expEnvVars(global_pars["local"]["geometry"]["file"].as<string>(GEOM_LOCAL_DEFAULT));
    geomFile = path + geomFile;
    global_pars["local"]["geometry"]["file"] = geomFile;

    Geometry *geom = new Geometry();
    geom->header = (GeomHeader*)malloc(sizeof(GeomHeader));
    MPI_Bcast(geom->header, geomhdr_floatsize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    if(geom->header->gridType != RECTANGLE) {
      geom->zgrid.resize(geom->header->nz);
      geom->dzgrid.resize(geom->header->nz);
            MPI_Bcast(&(geom->zgrid[0]), geom->header->nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&(geom->zgrid[0]), geom->header->nz, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
        MPI_Bcast(&migParams->dt, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if(nodeParams->superslave) {
      global_pars["_dt_prop"] = (double)migParams->dt;
    }

    geom->write(geomFile);
    delete geom;

    //global_pars["_dt_prop"] = (double) mydt; // more precisions ...

  }

}
void ModelRegrid::broadcastModels(MigParams *migParams, NodeParams *nodeParams) {

  string velFile1 = expEnvVars(global_pars["local"]["vel"]["file"].as<string>(VEL_LOCAL_DEFAULT));
  string velFile2 = expEnvVars(global_pars["local"]["vel"]["file"].as<string>(VEL_LOCAL_DEFAULT));
  //broadcast velocity
  broadcastModel(nodeParams, velFile1, velFile2); // fixed me
  if(nodeParams->superslave) {
    global_pars["local"]["vel"]["file"] = velFile2;
  }
  if(nodeParams->mpiRank == 0) print1m("  Velocity broadcast completed!\n");

  if(global_pars["global"]["vel_dual"]) {
    string vel2File1 = expEnvVars(global_pars["local"]["vel_dual"]["file"].as<string>(VEL2_LOCAL_DEFAULT));
    string vel2File2 = expEnvVars(global_pars["local"]["vel_dual"]["file"].as<string>(VEL2_LOCAL_DEFAULT));
    broadcastModel(nodeParams, vel2File1, vel2File2);
    if(nodeParams->superslave) {
      global_pars["local"]["vel_dual"]["file"] = vel2File2;
    }
    if(nodeParams->mpiRank == 0) print1m("  Velocity_Dual broadcast completed!\n");
  }

  if(global_pars["global"]["rho"]) {
    string rhoFile1 = expEnvVars(global_pars["local"]["rho"]["file"].as<string>(RHO_LOCAL_DEFAULT));
    string rhoFile2 = expEnvVars(global_pars["local"]["rho"]["file"].as<string>(RHO_LOCAL_DEFAULT));
    broadcastModel(nodeParams, rhoFile1, rhoFile2);
    if(nodeParams->superslave) {
      global_pars["local"]["rho"]["file"] = rhoFile2;
    }
  }

  if(global_pars["global"]["reflectivity"]) {
    string reflectivityFile1 = expEnvVars(global_pars["local"]["reflectivity"]["file"].as<string>(REFLECTIVITY_LOCAL_DEFAULT));
    string reflectivityFile2 = expEnvVars(global_pars["local"]["reflectivity"]["file"].as<string>(REFLECTIVITY_LOCAL_DEFAULT));
    broadcastModel(nodeParams, reflectivityFile1, reflectivityFile2);
    if(nodeParams->superslave) {
      global_pars["local"]["reflectivity"]["file"] = reflectivityFile2;
    }
  }

  if(global_pars["global"]["1/Q"]) {
    string QFile1 = expEnvVars(global_pars["local"]["1/Q"]["file"].as<string>(Q_LOCAL_DEFAULT));
    string QFile2 = expEnvVars(global_pars["local"]["1/Q"]["file"].as<string>(Q_LOCAL_DEFAULT));
    broadcastModel(nodeParams, QFile1, QFile2);
    if(nodeParams->superslave) {
      global_pars["local"]["1/Q"]["file"] = QFile2;
    }

    Q::mpibcast(nodeParams->mpiRank); // TODO: will empty nodes be a problem?
  }

  //broadcast epsilon and delta
  if(migParams->modType == VTI || migParams->modType == TTI) {
    string epsFile1 = expEnvVars(global_pars["local"]["epsilon"]["file"].as<string>(EPS_LOCAL_DEFAULT));
    string epsFile2 = expEnvVars(global_pars["local"]["epsilon"]["file"].as<string>(EPS_LOCAL_DEFAULT));
    broadcastModel(nodeParams, epsFile1, epsFile2); // fixed me
    if(nodeParams->mpiRank == 0) print1m("  Epsilon broadcast completed!\n");

    string delFile1 = expEnvVars(global_pars["local"]["delta"]["file"].as<string>(DEL_LOCAL_DEFAULT));
    string delFile2 = expEnvVars(global_pars["local"]["delta"]["file"].as<string>(DEL_LOCAL_DEFAULT));
    broadcastModel(nodeParams, delFile1, delFile2); // fixed me
    if(nodeParams->mpiRank == 0) print1m("  Delta broadcast completed!\n");

    if(nodeParams->superslave) {
      global_pars["local"]["epsilon"]["file"] = epsFile2;
      global_pars["local"]["delta"]["file"] = delFile2;
    }
  }

  //broadcast pjx and pjy
  if(migParams->modType == TTI) {
    string pjxFile1 = expEnvVars(global_pars["local"]["pjx"]["file"].as<string>(PJX_LOCAL_DEFAULT));
    string pjxFile2 = expEnvVars(global_pars["local"]["pjx"]["file"].as<string>(PJX_LOCAL_DEFAULT));
    broadcastModel(nodeParams, pjxFile1, pjxFile2); // fixed me
    if(nodeParams->mpiRank == 0) print1m("  Pjx broadcast completed!\n");

    string pjyFile1 = expEnvVars(global_pars["local"]["pjy"]["file"].as<string>(PJY_LOCAL_DEFAULT));
    string pjyFile2 = expEnvVars(global_pars["local"]["pjy"]["file"].as<string>(PJY_LOCAL_DEFAULT));
    broadcastModel(nodeParams, pjyFile1, pjyFile2); // fixed me
    if(nodeParams->mpiRank == 0) print1m("  Pjy broadcast completed!\n");

    if(nodeParams->superslave) {
      global_pars["local"]["pjx"]["file"] = pjxFile2;
      global_pars["local"]["pjy"]["file"] = pjyFile2;
    }
  }
}

void ModelRegrid::broadcastModel(NodeParams *nodeParams, string &fileName1, string &fileName2) {
    cout<<"enter ModelRegrid::broadcastModel"<<endl;
  if(nodeParams->mpiRank == 0) {
    //gettimeofday(&t1, NULL);
    fileName1 = path + fileName1;
    Fdm *vel = new Fdm();
    vel->read(fileName1);
    FdmHeader head = vel->getHeader();
    //vel->info();
    float *data = vel->getdata();
    size_t data_size = (size_t)head.nx * (size_t)head.ny * (size_t)head.nz * sizeof(float);
        MPI_Bcast((char*)&head, sizeof(FdmHeader), MPI_CHAR, 0, MPI_COMM_WORLD);
        libCommon::Utl::bcastBigVol(MPI_COMM_WORLD, (char*)data, data_size, 0);
    delete vel;

    } else if(nodeParams->superslave) {

    fileName2 = path + fileName2;

    Fdm *vel = new Fdm();
    FdmHeader hdr = vel->getHeader();
        MPI_Bcast((char*)&hdr, sizeof(FdmHeader), MPI_CHAR, 0, MPI_COMM_WORLD);

    size_t nxy = (size_t)hdr.nx * (size_t)hdr.ny;
    size_t nxyz = nxy * (size_t)hdr.nz;
    float *data = (float*)malloc(nxyz * sizeof(float));
    vel->setdata(data);
        libCommon::Utl::bcastBigVol(MPI_COMM_WORLD, (char*)data, nxyz*sizeof(float), 0);

        vel->fillHeader(hdr);
    vel->savecube(fileName2.c_str());

    delete vel;
  }
}

