#include "ModelLoader.h"
#include "Q.h"
#include "GetPar.h"
#include "libCommon/io_util.h"
using libCommon::read_all;
using libCommon::write_all;
#include "libFFTV/transpose.h"
#include "GlobalTranspose.h"
#include "libSWIO/RecordIO.hpp"
#include "FdEngine.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

#include <boost/algorithm/string/predicate.hpp>
using boost::algorithm::iends_with;

ModelLoader::ModelLoader(Model &_model, const char *_key, int _nThreads) : model(_model), key(_key), nThreads(_nThreads) {
  model.modeltype = FdEngine::determineModelType();
  model.useRho = ((bool)global_pars[GLOBAL]["rho"]) ? 1 : 0;
  assertion(!bool(global_pars[GLOBAL]["Q"]) && !bool(global_pars[LOCAL]["Q"]), "Please use the key '1/Q' instead of 'Q' for Q-model!");
  model.useQ = ((bool)global_pars[GLOBAL]["1/Q"]) ? 1 : 0;
  model.useReflectivity = ((bool)global_pars[GLOBAL]["reflectivity"]) ? 1 : 0;
  Q::order = model.useQ ? global_pars["q_order"].as<int>(4) : 0; // num of aux field

  // set default values for local ones ...
  if(strcmp(_key, LOCAL) == 0) {
    if(!global_pars[key]["vel"]["file"]) global_pars[key]["vel"]["file"] = VEL_LOCAL_DEFAULT;
    if(!global_pars[key]["geometry"]["file"]) global_pars[key]["geometry"]["file"] = GEOM_LOCAL_DEFAULT;

    // optional ones below:
    if(!global_pars[key]["vel_dual"]["file"] && global_pars[GLOBAL]["vel_dual"]["file"]) global_pars[key]["vel_dual"]["file"] =
        VEL2_LOCAL_DEFAULT;
    if(!global_pars[key]["rho"]["file"] && global_pars[GLOBAL]["rho"]["file"]) global_pars[key]["rho"]["file"] =
        VEL2_LOCAL_DEFAULT;
    if(!global_pars[key]["rho"]["file"] && global_pars[GLOBAL]["rho"]["file"]) global_pars[key]["rho"]["file"] =
    RHO_LOCAL_DEFAULT;
    if(!global_pars[key]["reflectivity"]["file"] && global_pars[GLOBAL]["reflectivity"]["file"]) global_pars[key]["reflectivity"]["file"] =
    REFLECTIVITY_LOCAL_DEFAULT;
    if(!global_pars[key]["1/Q"]["file"] && global_pars[GLOBAL]["1/Q"]["file"]) global_pars[key]["1/Q"]["file"] =
    Q_LOCAL_DEFAULT;
    if(!global_pars[key]["epsilon"]["file"] && global_pars[GLOBAL]["epsilon"]["file"]) global_pars[key]["epsilon"]["file"] =
    EPS_LOCAL_DEFAULT;
    if(!global_pars[key]["delta"]["file"] && global_pars[GLOBAL]["delta"]["file"]) global_pars[key]["delta"]["file"] =
    DEL_LOCAL_DEFAULT;
    if(!global_pars[key]["pjx"]["file"] && global_pars[GLOBAL]["pjx"]["file"]) global_pars[key]["pjx"]["file"] =
    PJX_LOCAL_DEFAULT;
    if(!global_pars[key]["pjy"]["file"] && global_pars[GLOBAL]["pjy"]["file"]) global_pars[key]["pjy"]["file"] =
    PJY_LOCAL_DEFAULT;
    if(!global_pars[key]["dipx"]["file"] && global_pars[GLOBAL]["dipx"]["file"]) global_pars[key]["dipx"]["file"] =
    DIPX_LOCAL_DEFAULT;
    if(!global_pars[key]["dipy"]["file"] && global_pars[GLOBAL]["dipy"]["file"]) global_pars[key]["dipy"]["file"] =
    DIPY_LOCAL_DEFAULT;
    if(!global_pars[key]["dip"]["file"] && global_pars[GLOBAL]["dip"]["file"]) global_pars[key]["dip"]["file"] =
    DIP_LOCAL_DEFAULT;
    if(!global_pars[key]["azimuth"]["file"] && global_pars[GLOBAL]["azimuth"]["file"]) global_pars[key]["azimuth"]["file"] =
    AZIMUTH_LOCAL_DEFAULT;
  }
//  model.RhoCN* = global_pars["RhoCN*"].as<int>(0);
//  if(!model.useRho) model.RhoCN* = 0;
//  else if(!model.RhoCN*) global_pars["force_tti"] = 1; // important to determine isDerive2nd()

  if(model.useRho) global_pars["force_tti"] = 1; // added according to Lines 55-57; by wolf

  print1m("useRho=%d\n", model.useRho);

  assertion((bool)global_pars[GLOBAL]["vel"], "['%s']['vel'] tree need to be defined!", GLOBAL);

  global_pars[GLOBAL]["vel"] = read_grid(global_pars[GLOBAL], global_pars[GLOBAL]["vel"], nz, nx, ny, dz, dx, dy, z0, x0, y0);

  xMinValid = x0;
  xMaxValid = x0 + (nx - 1) * dx;

  yMinValid = y0;
  yMaxValid = y0 + (ny - 1) * dy;

  xratio = yratio = zratio = 1;

  elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);
}

Node ModelLoader::read_grid(Node trunk, Node branch, int &nz, int &nx, int &ny, float &dz, float &dx, float &dy, float &z0, float &x0,
    float &y0) {

  string velFile = expEnvVars(branch["file"].as<string>(""));
  if(iends_with(velFile, ".fdm")) {  // read grid info from FDM
    Fdm vel;
    vel.readheader(velFile.c_str());
    FdmHeader head = vel.getHeader();
    nz = head.nz, nx = head.nx, ny = head.ny;
    dz = head.dz, dx = head.dx, dy = head.dy;
    z0 = head.z0, x0 = head.x0, y0 = head.y0;
  } else if(iends_with(velFile, ".js")) {
    libSeismicFileIO::JSDataReader reader(velFile);
    nz = reader.getAxisLen(0);
    z0 = reader.getAxisPhysicalOrigin(0);
    dz = reader.getAxisPhysicalDelta(0);
    nx = reader.getAxisLen(1);
    x0 = reader.getAxisPhysicalOrigin(1);
    dx = reader.getAxisPhysicalDelta(1);
    ny = reader.getAxisLen(2);
    y0 = reader.getAxisPhysicalOrigin(2);
    dy = reader.getAxisPhysicalDelta(2);
  } else {
    int xl0, xl_inc, il0, il_inc;
    branch = model_builder::read_grid(trunk, branch, nz, nx, ny, dz, dx, dy, z0, x0, y0, xl0, xl_inc, il0, il_inc);
  }

  if(global_pars["geometry"]) {
    GlobalTranspose gtrans;
    gtrans.worldToLocal(x0, y0);
  }

  return branch;
}

ModelLoader::~ModelLoader() {
}

void ModelLoader::saveModels() {
  model.fdms[VEL]->savecube("v1.fdm");
  //model.fdms[EPS]->savecube("v2.fdm");
  //  model.fdms[DEL]->savecube("v3.fdm");
  //model.fdms[PJX]->savecube("v4.fdm");
  //  model.fdms[PJY]->savecube("v5.fdm");
}

void ModelLoader::loadLocalModels(float **volModel, float x0, float y0, int nx, int ny, float &vmax) {
  string g = "global", l = "local";

  // loading velocity model
  string velFile = expEnvVars(global_pars[l]["vel"]["file"].as<string>(VEL_LOCAL_DEFAULT));
  print1m("loading velocity file %s ...\n\n", velFile.c_str());
  Fdm *vel = new Fdm();
  model.fdms[VEL] = vel;
  vel->read(velFile, volModel[VEL], x0, y0, nx, ny);

  FdmHeader head = vel->getHeader();
  vmax = head.vmax;
  print1m("The velocity volume dimension is %d  %d  %d  %f  %f %f  %f  %f  %f\n", head.nx, head.ny, head.nz, head.dx, head.dy, head.dz,
          head.x0, head.y0, head.z0);

  if(global_pars[g]["rho"] || global_pars[l]["rho"]) {
    string rhoFile = expEnvVars(global_pars[l]["rho"]["file"].as<string>(RHO_LOCAL_DEFAULT));
    print1m("loading rho file %s ...\n\n", rhoFile.c_str());
    Fdm *rho = new Fdm();
    model.fdms[RHO] = rho;
    rho->read(rhoFile, volModel[RHO], x0, y0, nx, ny);
  }

  if(global_pars[g]["reflectivity"] || global_pars[l]["reflectivity"]) {
    string reflectivityFile = expEnvVars(global_pars[l]["reflectivity"]["file"].as<string>(REFLECTIVITY_LOCAL_DEFAULT));
    print1m("loading reflectivity file %s ...\n\n", reflectivityFile.c_str());
    Fdm *reflectivity = new Fdm();
    model.fdms[REFLECTIVITY] = reflectivity;
    reflectivity->read(reflectivityFile, volModel[REFLECTIVITY], x0, y0, nx, ny);
  }

  if(global_pars[g]["1/Q"] || global_pars[l]["1/Q"]) {
    string QFile = expEnvVars(global_pars[l]["1/Q"]["file"].as<string>(Q_LOCAL_DEFAULT));
    print1m("loading Q file %s ...\n\n", QFile.c_str());
    Fdm *invq = new Fdm();
    model.fdms[Q] = invq;
    invq->read(QFile, volModel[Q], x0, y0, nx, ny);
  }

  //loading epsilon
  if(global_pars[g]["epsilon"] || global_pars[l]["epsilon"]) {
    string epsilonFile = expEnvVars(global_pars[l]["epsilon"]["file"].as<string>(EPS_LOCAL_DEFAULT));
    print1m("loading epsilon file %s ...\n\n", epsilonFile.c_str());
    Fdm *eps = new Fdm();
    model.fdms[EPS] = eps;
    eps->read(epsilonFile, volModel[EPS], x0, y0, nx, ny);
  }

  //loading delta
  if(global_pars[g]["delta"] || global_pars[l]["delta"]) {
    string deltaFile = expEnvVars(global_pars[l]["delta"]["file"].as<string>(DEL_LOCAL_DEFAULT));
    print1m("loading delta file %s ...\n\n", deltaFile.c_str());
    Fdm *del = new Fdm();
    model.fdms[DEL] = del;
    del->read(deltaFile, volModel[DEL], x0, y0, nx, ny);
  }

  if(FdEngine::determineModelType() >= TTI) {
    string pjxFile = expEnvVars(global_pars[l]["pjx"]["file"].as<string>(PJX_LOCAL_DEFAULT));
    print1m("loading dipx file %s ...\n\n", pjxFile.c_str());
    Fdm *pjx = new Fdm();
    model.fdms[PJX] = pjx;
    pjx->read(pjxFile, volModel[PJX], x0, y0, nx, ny);

    string pjyFile = expEnvVars(global_pars[l]["pjy"]["file"].as<string>(PJY_LOCAL_DEFAULT));
    print1m("loading dipyFile file %s ...\n\n", pjyFile.c_str());
    Fdm *pjy = new Fdm();
    model.fdms[PJY] = pjy;
    pjy->read(pjyFile, volModel[PJY], x0, y0, nx, ny);
  }

}

void ModelLoader::reloadLocalModels(float **volModel, float x0, float y0, int nx, int ny, float &vmax, ModelVolID id) {
  string g = "global", l = "local";
  if(id == VEL2) {
    string vel2File = expEnvVars(global_pars[l]["vel_dual"]["file"].as<string>(VEL2_LOCAL_DEFAULT));
    print1m("loading vel_dual file %s for backward propagation...\n\n", vel2File.c_str());
    Fdm *vel2 = new Fdm();
    model.fdms[VEL] = vel2;
    vel2->read(vel2File, volModel[VEL], x0, y0, nx, ny);
    FdmHeader head2 = vel2->getHeader();
    vmax = head2.vmax;
  }
  else if(id == VEL) {
    string velFile = expEnvVars(global_pars[l]["vel"]["file"].as<string>(VEL_LOCAL_DEFAULT));
    print1m("loading vel file %s for image condition...\n\n", velFile.c_str());
    Fdm *vel = new Fdm();
    model.fdms[VEL] = vel;
    vel->read(velFile, volModel[VEL], x0, y0, nx, ny);
    FdmHeader head = vel->getHeader();
    vmax = head.vmax;
  }
}


void ModelLoader::loadModelVolume(ModelVolID id) {
  if(id == VEL) {
    loadFile("vel", id);
  } else if(id == RHO) {
    loadFile("rho", id);
  } else if(id == REFLECTIVITY) {
    loadFile("reflectivity", id);
  } else if(id == Q) {
    loadFile("1/Q", id);
  } else if(id == EPS) {
    loadFile("epsilon", id);
  } else if(id == DEL) {
    loadFile("delta", id);
  } else if(id == PJX) {
    // Both PJX and PJY is loaded when PJX is requested
    loadTTIFile();
  } else if (id == VEL2) {
    loadFile("vel_dual", id);
  } else assertion(false, "ModelVolID %d not implemented!", id);
}

Fdm* ModelLoader::loadFile(string modKey, ModelVolID id) {
  //print1m("ModelLoader::loadFile modKey=%s, id=%d \n", modKey.c_str(), id);
  Fdm *fdm = new Fdm(); // do not delete this pointer
  model.fdms[id] = fdm;

  int nz, nx, ny;
  float dz, dx, dy, z0, x0, y0;
  global_pars[key][modKey] = read_grid(global_pars[key], global_pars[key][modKey], nz, nx, ny, dz, dx, dy, z0, x0, y0);

  int fx = (int)nearbyintf(x0 / dx) + 1;
  int fy = (int)nearbyintf(y0 / dy) + 1;
  int fz = 0;
  fdm->sethead(x0, y0, z0, nx, ny, nz, dx, dy, dz, fx, fy, fz, 1, 1, 1);
  float *vol = (float*)malloc(sizeof(float) * nx * ny * nz + 128);
  fdm->setdata(vol);

  string modFile = expEnvVars(global_pars[key][modKey]["file"].as<string>(""));
  if(modFile.length() > 0) {
    print1m("loading %s file %s ...\n\n", modKey.c_str(), modFile.c_str());
    loadFile(modFile, vol);
  } else {
    print1m("using model_builder for [%s][%s] ...\n\n", key.c_str(), modKey.c_str());
    assertion(global_pars["zMin"].as<float>(0) >= 0, "zMin<0 is not supported by model_builder::load_model() yet!");
    model_builder::load_model(vol, global_pars[key], global_pars[key][modKey]);
  }

  // check fdm values
  if(fdm->haveNanInf()) libCommon::Utl::fatal(modFile + "have Nan or Inf");
  if(modKey == "vel" || modKey == "vel_dual" ) if(fdm->haveSmallValues(100.0)) libCommon::Utl::fatal(
      string("velocity file ") + modFile + string(" has small value which is less than 100.0"));

  return fdm;
}

void ModelLoader::loadTTIFile() {
  if(global_pars[key]["pjx"] && global_pars[key]["pjy"]) {
    loadFile("pjx", PJX);
    loadFile("pjy", PJY);
  } else if(global_pars[key]["dipx"] && global_pars[key]["dipy"]) {
    Fdm *pjx = loadFile("dipx", PJX);
    Fdm *pjy = loadFile("dipy", PJY);
    // convert dipxy into jxy
    FdmHeader hdr = pjx->getHeader();
    float *dipx = pjx->getdata();
    float *dipy = pjy->getdata();
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < hdr.ny; iy++) {
      for(int ix = 0; ix < hdr.nx; ix++) {
        for(int iz = 0; iz < hdr.nz; iz++) {
          size_t id = (size_t)((iy) * hdr.nx + ix) * (size_t)hdr.nz + iz;
          float dpx = dipx[id];
          float dpy = dipy[id];
          float scaler = -1.0f / sqrtf(dpx * dpx + dpy * dpy + 1.0);
          dipx[id] = dpx * scaler;
          dipy[id] = dpy * scaler;
        }
      }
    }
  } else if(global_pars[key]["dip"] && global_pars[key]["azimuth"]) {
    Fdm *pjx = loadFile("dip", PJX);
    Fdm *pjy = loadFile("azimuth", PJY);

    // convert into jxy
    FdmHeader hdr = pjx->getHeader();
    float *theta = pjx->getdata();
    float *phi = pjy->getdata();
    float ang2rad = M_PI / 180.0f;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int iy = 0; iy < hdr.ny; iy++) {
      for(int ix = 0; ix < hdr.nx; ix++) {
        for(int iz = 0; iz < hdr.nz; iz++) {
          size_t id = (size_t)((iy) * hdr.nx + ix) * (size_t)hdr.nz + iz;
          float sin_theta = sinf(theta[id] * ang2rad);
          float sin_phi = sinf(phi[id] * ang2rad);
          float cos_phi = cosf(phi[id] * ang2rad);
          theta[id] = -sin_theta * cos_phi;
          phi[id] = -sin_theta * sin_phi;
        }
      }
    }
  }
}

void ModelLoader::loadFile(string &file, float *vol) {
  if(iends_with(file, ".fdm")) {
    //print1m("loadFdmFile \n");
    loadFdmFile(file, vol);
  } else if(iends_with(file, "bin")) {
    //print1m("loadBinFile \n");
    loadBinFile(file, vol);
  } else {
    //print1m("loadRecordFile \n");
    loadRecordFile(file, vol);
  }
}

void ModelLoader::loadFdmFile(string &file, float *vol) {
  if(file.length() > 4) {
    print1m("loading model file %s ...\n\n", file.c_str());
    Fdm fdm;
    fdm.read(file, vol);
  }
}

void ModelLoader::loadBinFile(string &file, float *vol) {
  if(file.length() > 0) {
    print1m("loading model bin format file %s ...\n\n", file.c_str());

    int fd = open64(file.c_str(), O_RDONLY);
    if(fd < 0) {
      fprintf(stderr, "\n**** Unable to open file '%s' ! Cannot continue.\n", file.c_str());
      perror("Error");
      exit(-1);
    }
    read_all(fd, vol, sizeof(float) * nx * ny * nz);
    close(fd);
  }
}

void ModelLoader::loadRecordFile(string &fileName, float *dataBuf) {
  vector<libCommon::Trace*> traces;
  libSeismicFileIO::RecordReader reader(fileName);
  int nt = reader.getNSample();
  while(reader.readNextFrame(traces)) {
    for(size_t itrace = 0; itrace < traces.size(); itrace++) {
      memcpy((void*)dataBuf, (void*)(traces[itrace]->getData()), nt * sizeof(float));
      dataBuf += nt;
    }
    libCommon::Utl::deletePtrVect(traces);
  }
}

