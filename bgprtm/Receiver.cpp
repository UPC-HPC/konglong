/*
 * Receiver.cpp
 *
 */

#include <stdlib.h>
#include <math.h>
#include <unistd.h>

#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;

#include "Grid.h"
#include "Source.h" // TODO: later combine source/receiver (helper)
#include "Receiver.h"
#include "Propagator.h"
#include "Vector3.h"
#include "Model.h"
#include "fdm.hpp"
#include "GetPar.h"
#include "RecordLoader.h"
#include <omp.h>
#include <complex.h>
#include "Traces.h"
#include "Wavefield.h"
#include "libCommon/Horizon.hpp"
#include "libCommon/CommonMath.h"
#include "libCommon/padfft.h"
#include "libCommon/DftFilter.h"
using libCommon::DftFilter;

#include "MpiPrint.h"
using MpiPrint::print1m;

#define KDOMAIN 1
#define SDOMAIN 2

#define DEBUG_RECV 0
const int CACHED_IN_MEMORY = 100;
typedef std::complex<float> Complex;

Receiver::Receiver(shared_ptr<Grid> grid, Model *model, float dt, int nt, float maxFreq, float minFreq, int spreadSize, int dim, int id0,
    PROP::Operation op, bool bufferToDisk) : grid(grid), model(model), nt(nt), dt(dt), fhcut(maxFreq), spreadSize(spreadSize), dim_bits(
    dim), id(id0), oper(op), iz0(0), iz1(0), nsz(1), vol4d(NULL), buffer(NULL), size4d(0), bufferToDisk(bufferToDisk), scaleFactor(1.0f) {
  spreadSizeZ = global_pars["sourceSpreadSizeZ"].as<int>(spreadSize); // this will also control the receiver extraction, see: Propagator::setupReceiverForModeling()
  use_src4d = getBool(global_pars["use_src4d"], false);
  transpose_fk = getBool(global_pars["transpose_fk"], false); // this default saves memory when FK is involved
  khpass = global_pars["src_apply_khpass"].as<float>(); // default values unified in ModelRegrid::prepModel
  khcut = global_pars["src_apply_khcut"].as<float>();
  print1m("Receiver use_src4d=%d, khpass=%f, khcut=%f\n", use_src4d, khpass, khcut);
  //this->bufferToDisk = getBool(global_pars["buffer_to_disk "], false);
  print1m("Receiver bufferToDisk=%s\n", this->bufferToDisk ? "true" : "false");

  //20201031 to improve only turn off for usewavelet option under fwi.
 dt_dxyz = 1.0f / grid->dz;
  if(!(dim & OneD)) dt_dxyz /= grid->dx;
  if(dim & ThreeD) dt_dxyz /= grid->dy;

  if(id == USER_WAVELET) {
    id = SOURCE;
  }
  // print1m("Receiver::dt_dxyz=%g(dt=%f,dz=%f,dx=%f)\n", dt_dxyz, dt, grid->dz, grid->dx);

  PROP::Direction direction = id == RECEIVER ? PROP::BACKWARD : PROP::FORWARD;

  recordLoader = new RecordLoader(grid.get(), dt, nt, maxFreq, minFreq, dim, direction);

  nThreads = init_num_threads();

  nx = grid->nx, ny = grid->ny, nz = grid->nz;
  ix0 = 0;
  ix1 = nx - 1;

  iy0 = 0;
  iy1 = ny - 1;

  nz_grid = nz;

  myInterp = false;
  sfBoundType = Source::getSurfaceType(direction);
  mirror_ghost = ((id == SOURCE) && ((sfBoundType == SOURCE_GHOST) || (sfBoundType == GHOST)))
      || ((id == RECEIVER) && ((sfBoundType == RECEIVER_GHOST) || (sfBoundType == GHOST)));

  trueamp = getTrueAmp(oper);

  string str = global_pars["deghost"].as<string>("NONE");
  deghost = getSideBits(str);
  zsurf = -global_pars["zMin"].as<float>(0.0f);

  if(id != RECEIVER && oper != PROP::RTMM) { // recLowCut et al handled in RecordLoader ...
    float defaultLowPass = ((deghost & SIDE_SOURCE) && Source::getWaveletType() == SPIKY) ? 1.5f : 0.0f; // deghost: taper the low end of spiky
    flcut = global_pars["souLowCut"].as<float>(0.0f);
    flpass = global_pars["souLowPass"].as<float>(defaultLowPass); // TODO: currently only applied when go through FK operation
  }
}

Receiver::SIDE_BITS Receiver::getSideBits(string sou_rec_side) {
  std::transform(sou_rec_side.begin(), sou_rec_side.end(), sou_rec_side.begin(), ::toupper);
  return (sou_rec_side == "BOTH") ? SIDE_BOTH : (sou_rec_side == "SOURCE") ? SIDE_SOURCE :
         (sou_rec_side == "RECEIVER") ? SIDE_RECEIVER : SIDE_NONE;
}

int Receiver::getTrueAmp(PROP::Operation oper) {
  string str = global_pars["true_amplitude"].as<string>(oper == PROP::RTM ? "RECEIVER" : oper == PROP::RTMM ? "BOTH" : "NONE");
  std::transform(str.begin(), str.end(), str.begin(), ::toupper);
  assertion(str.find("DOMAIN") == string::npos, "Please change legacy jobdeck!\n"
            "  true_amplitude now accepts: NONE(default for FWD), SOURCE, RECEIVER(default for BWD), BOTH.\n"
            "  Use 'true_amplitude_dipole: false' if you want to choose FK domain true amp.\n"
            "  Use 'fk_domain' if you want to choose KDOMAIN or SDOMAIN(default).\n");
  int trueamp = getSideBits(str);
  return trueamp;
}

bool Receiver::isDipole() {
  return isDipole(id == RECEIVER ? PROP::BACKWARD : PROP::FORWARD, oper);
}

bool Receiver::isDipole(PROP::Direction direction, PROP::Operation oper) {
  string str = global_pars["deghost"].as<string>("NONE");
  int deghost = Receiver::getSideBits(str);
  int trueamp = getTrueAmp(oper);

  int do_trueamp = trueamp & (direction + 1);
  int do_deghost = deghost & (direction + 1);
  int true_amp_dipole = getBool(global_pars["true_amplitude_dipole"], direction == PROP::BACKWARD && oper == PROP::MOD); // !do_deghost);
  return do_trueamp && true_amp_dipole;
}

void Receiver::update_zrange(int do_alloc) {
  if(!traces) return;

  assertion(traces->xyzMax.z >= traces->xyzMin.z, "Receiver zmax=%f is less than zmin=%f. ntraces=%d", traces->xyzMax.z, traces->xyzMin.z,
            traces->getNReceivers());
  // note: cannot handle ghosts here, probably no need too
  int izmin = (int)floorf(grid->getIDzf(traces->xyzMin.z));
  int izmax = (int)floorf(grid->getIDzf(traces->xyzMax.z));
  int spreadZlimit = min(izmin + 1 - iz0, nz - 1 - izmax);
  if(spreadZlimit < spreadSizeZ) {
    print1m(
        "### zmin=%f, zmax=%f ==> izmin=%d, izmax=%d, nz=%d\n" //
        "### trying to set sourceSpreadSizeZ to spreadZLimit=%d but no less than 5\n",
        traces->xyzMin.z, traces->xyzMax.z, izmin, izmax, nz,
        spreadZlimit);
    spreadSizeZ = max(5, spreadZlimit);
  }
  iz0 = izmin - spreadSizeZ + 1; // >=0, i.e. spreadSizeZ<= izmin+1-iz0
  iz1 = izmax + spreadSizeZ; // inclusive, < nz, i.e. spreadSizeZ <= nz-1-izmax
  assertion(iz0 >= 0, "receiver zmin is out of the grid range iz0=%d, recvzMin=%f (iz:%d), spreadZ=%d.", iz0, traces->xyzMin.z, izmin,
            spreadSizeZ);
  assertion(iz1 < nz, "receiver zmax is out of the grid range iz1=%d, nz=%d, recvzMax=%f (iz:%d), spreadZ=%d.", iz1, nz, traces->xyzMax.z,
            izmax, spreadSizeZ);
  if(mirror_ghost) {
    int iz = floorf(grid->getIDzf(-traces->xyzMax.z)) - spreadSizeZ + 1;
    int nzuppad = global_pars["nzUpPad"].as<int>(10);
    assertion(
        iz >= 0,
        "receiver zmax for mirror ghost is out of the grid range iz=%d, recvz=%f, spread=%d, nzUpPad=%d [increase to %d (and re-run model_prep if running non-mpi version)].",
        iz, traces->xyzMax.z, spreadSizeZ, nzuppad, nzuppad - iz);
     }

  nsz = max(1, iz1 - iz0 + 1); // 2*spread if zmin==zmax

  print1m("\n");
  (id == SOURCE) ? print1m("Source spread: \n") : print1m("Receiver spread: \n");
  print1m("                ix0=%d, iy0=%d, iz0=%d \n", ix0, iy0, iz0);
  print1m("                nx=%d, ny=%d, nsz=%d \n", nx, ny, nsz);

  if(!do_alloc || !use_src4d) return;

  size4d = (size_t)nx * ny * nsz * nt;

  if(bufferToDisk) {
    print1m("Buffering preprocessed receivers to disk.\n");
    size_t nxyz = (size_t)nx * nsz * ny;
    delete[] buffer;
    buffer = new float[nxyz * CACHED_IN_MEMORY];

    bufferFileName = generateFileName();
    vol4dFile.open(bufferFileName.c_str(), std::ios::in | std::ios::out | std::ios::trunc | std::ios::binary);
    if(!vol4dFile.is_open()) {
      print1m("Unable to open file for buffering receiver data: %s\n", bufferFileName.c_str());
      exit(-1);
    }
  } else {
    delete[] vol4d;
    vol4d = new float[size4d];
    if(!vol4d) {
      print1m("Error: Cannot allocate memory for receiver %ld\n", size4d);
      exit(-1);
    }
    size_t nxz = (size_t)nx * nsz;
    size_t nxyz = nxz * ny;
    for(int it = 0; it < nt; it++) {
#pragma omp parallel for num_threads(nThreads) schedule(static)
      for(int iy = 0; iy < ny; iy++) {
        memset(vol4d + (size_t)it * nxyz + (size_t)iy * nxz, 0, nxz * sizeof(float));
      }
    }
  }
}

//
Receiver::~Receiver() {
  if(bufferToDisk) {
    vol4dFile.close();
    // Delete the file from disk
    remove(bufferFileName.c_str());
    delete[] buffer, buffer = NULL;
  } else {
    delete[] vol4d, vol4d = NULL;
  }

  delete recordLoader, recordLoader = NULL;
}

int Receiver::loadData(const char *fileName) {
  traces = recordLoader->readRecord(fileName, false);
  if(traces == nullptr) return 0;
  update_zrange();
  return 1;
}

int Receiver::loadHdr(const char *fileName) {
  traces = recordLoader->readRecord(fileName, false, true);
  update_zrange(false);
  return traces == nullptr ? 0 : 1;
}

int Receiver::createHdrFromGrid(Node &node) {
#if 0 // TODO: need to be consistent with RecordLoader::readRecord() ?
  float maxOffset = global_pars["maxOffset"].as<float>(-1.0f);
  int ixMin = -1;
  int ixMax = nx - 1;
  int iyMin = -1;
  int iyMax = ny - 1;
  if(getBool(global_pars["skipTraceOutsideOfModel"], false)) {
    RecordLoader::getModelRange(grid, ixMin, ixMax, iyMin, iyMax);
  }
#endif

  traces = make_unique<Traces>(0); // nt=0: no data
  float frx = global_pars["receiverX0"].as<float>();
  float fry = global_pars["receiverY0"].as<float>(0);
  float frz = global_pars["receiverZ0"].as<float>(0.0f);
  float dxr = global_pars["receiverXinc"].as<float>();
  float dyr = global_pars["receiverYinc"].as<float>(0);
  float dzr = global_pars["receiverZinc"].as<float>(0);

  int nxr = global_pars["nxreceivers"].as<int>(1);
  int nyr = global_pars["nyreceivers"].as<int>(1);
  int nzr = global_pars["nzreceivers"].as<int>(1);
  int nr = nxr * nyr * nzr;
  float rzmin = grid->getmyz(spreadSizeZ);
  float rzmax = grid->getmyz(nz - 1 - spreadSizeZ);
  int iz0 = (int)floorf(grid->getIDzf(rzmin)) - spreadSizeZ + 1;
  int iz1 = (int)floorf(grid->getIDzf(rzmax)) + spreadSizeZ; // inclusive
  assertion(iz0 >= 0, "rzmin=%f calculated is still too small! iz0=%d, spread=%d.", rzmin, iz0, spreadSizeZ);
  assertion(iz1 < nz, "rzmax=%f calculated is still too big! iz1=%d, nz=%d, spread=%d.", rzmax, iz1, nz, spreadSizeZ);

  for(int ir = 0; ir < nr; ir++) {
    int iz = ir / (nxr * nyr);
    int iy = ir / nxr - iz * nyr;
    int ix = ir - iz * nxr * nyr - iy * nxr;
    float rx = frx + ix * dxr;
    float ry = fry + iy * dyr;
    float rz = frz + iz * dzr;
    if(rz >= rzmin && rz <= rzmax) traces->addReceiver(vector3(rx, ry, rz));
  }
  if((int)traces->coord.size() < nr) {
    fprintf(stderr,
            "WARNING: createHdrFromGrid(): added trace number=%ld is less than input nr=%d! (rzmin=%f, rzmax=%f, spreadZ=%d, nz=%d)\n",
            traces->coord.size(), nr, rzmin, rzmax, spreadSizeZ, nz);
  }
  assertion(!traces->coord.empty(), "createHdrFromGrid(): NO traces added!");
  update_zrange(false);
  return traces == nullptr ? 0 : 1;
}

void Receiver::setData(unique_ptr<Traces> traces) {
  this->traces = move(traces);
  update_zrange();
}

std::string Receiver::generateFileName() {
  string prefix = expEnvVars(global_pars["receiverBufferPrefix"].as<string>("/tmp/"));
  std::stringstream strm;
  strm << prefix << "preprorecvs_p" << getpid() << ".bin";
  return strm.str();
}

static void taper_kernel(vector<float> &kernel, int l, int off, vector<float> &taper, int ntap, int n) {
  if(n <= ntap * 2) return;
  if(off < ntap) {  // i-off >=0 && i-off<l, i.e., i>=off, i<l+off
    int nlim = min(ntap, l + off), i0 = max(0, off);
    for(int i = i0; i < nlim; i++)
      kernel[i - off] *= taper[i];
  }
  if(off + l > n - ntap) {  // iTaper = n-1-off-iKernel, n-(off+i+1) >=0 && n-(off+i+1) < l, i.e., i<n-off, i>=n-off-l
    int nlim = min(ntap, n - off), i0 = max(0, n - off - l);
    for(int i = i0; i < nlim; i++)
      kernel[n - (off + i + 1)] *= taper[i];
  }
}

// returns average z (relative to water surface, i.e., -zMin) and stddev
vector<float> Receiver::spreadCoeffs(int ntaperXY, bool do_dipole) {
  did_dipole = do_dipole;

  int debug_apply = global_pars["debug_apply_force"].as<int>(0);
  assertion(spreadSize > 0 || debug_apply, "spreadSize must >0 to use receiver template (unless debug_apply_force is set)!");

  int spread_limx = nx == 1 ? 1 : max(1, spreadSize * 2);
  int spread_limy = ny == 1 ? 1 : max(1, spreadSize * 2);
  int spread_limz = max(1, iz1 - iz0 + 1);

  vector<float> wtaperXY(ntaperXY);
  for(int i = 0; i < ntaperXY; i++)
    wtaperXY[i] = 0.5f * (1.0f - cosf(M_PI * i / ntaperXY));

  // allocate kernels and offx, offy, offz for each receiver
  int nr = traces->getNReceivers();
  offxs.resize(nr), offys.resize(nr), offzs.resize(nr);
  lxs.resize(nr, spread_limx), lys.resize(nr, spread_limy), lzs.resize(nr, spread_limz);
  kernels_x.resize(nr), kernels_y.resize(nr), kernels_z.resize(nr);
  float zsum = 0;
  int do_deghost = deghost & (id + 1);
  float zsurf = -global_pars["zMin"].as<float>(0.0f);
#pragma omp parallel for num_threads(nThreads) schedule(static) reduction(+:zsum)
  for(int ir = 0; ir < nr; ir++) {
    float x = grid->getIDxf(traces->coord[ir].x, traces->coord[ir].z);
    float y = grid->getIDyf(traces->coord[ir].y, traces->coord[ir].z);
    zsum += traces->coord[ir].z - zsurf;
    float z = grid->getIDzf(do_deghost ? zsurf : traces->coord[ir].z); // deghost moves the source to surface
    if(ir == 0) { // for QC
      global_pars["_IDxf"] = x;
      global_pars["_IDyf"] = y;
      global_pars["_IDzf"] = z;
    }
    kernels_x[ir].resize(spread_limx);
    kernels_y[ir].resize(spread_limy);
    kernels_z[ir].resize(spread_limz);
    // offsets are absolute to the prop grid, so input z was not shifted, first argument must be whole grid size
    offxs[ir] = Util::CompactOrmsbySpreadCoeff(nx, kernels_x[ir], x, khpass, khcut, &lxs[ir]);
    offys[ir] = Util::CompactOrmsbySpreadCoeff(ny, kernels_y[ir], y, khpass, khcut, &lys[ir]);
    offzs[ir] = Util::CompactOrmsbySpreadCoeff(nz_grid, kernels_z[ir], z, khpass, khcut, &lzs[ir], do_dipole);
    taper_kernel(kernels_x[ir], lxs[ir], offxs[ir], wtaperXY, ntaperXY, nx);
    taper_kernel(kernels_y[ir], lys[ir], offys[ir], wtaperXY, ntaperXY, ny);
//    if(ir == nr / 2) print1m("z0:%f, nsz=%d, lz=%d, offz=%d, coeffz: ", z, nsz, lzs[ir], offzs[ir]);
//    if(ir == nr / 2) Util::print_vector(kernels_z[ir], spread_limz);
    //    exit(1);
    //    if (ir == 0 || ir == nr / 2 || ir == nr - 1) print1m("x[%d]=%f, nx=%d, lx=%d, offx=%d\n", ir, x, nx, lxs[ir],
    //        offxs[ir]), fflush(stdout);
    //    if (ir == nr / 2) print1m("z=%f, nsz=%d, lz=%d, offz=%d\n", z, nsz, lzs[ir], offzs[ir]);
  }

  float zmean = zsum / nr;
  zsum = 0;
  for(int ir = 0; ir < nr; ir++) {
    float dev = traces->coord[ir].z - zsurf - zmean;
    zsum += dev * dev;
  }
  return vector<float> { zmean, sqrtf(zsum / nr) };
}

// data had no coordinates
void Receiver::resetTraces(float *dat, int nt, float dt, int nr) {
  traces = make_unique<Traces>(nt, dat, nr, dt);
  traces->coord.resize(nr); // this is the actual num of traces
}

void Receiver::flipCoeff(float *mycoeffz, int lz) {

  if(mirror_ghost) {

    vector<float> mycoeffz1(lz);

    int izloc = grid->getIDz(zsurf) - iz0;

    for(int iz = 0; iz < lz; iz++) {
      if(2 * izloc - iz >= 0 && 2 * izloc - iz < lz) {
        mycoeffz1[iz] = mycoeffz[2 * izloc - iz];
      } else {
        mycoeffz1[iz] = 0.0;
      }
      //      print1m("receiver ghost: iz=%d, %f, %f \n", iz, mycoeffz[iz], mycoeffz1[iz]);
    }
    //    OrmsbySpreadCoeff(iz0, nsz, mycoeffz1, zloc, khpass, khcut);
    for(int iz = 0; iz < lz; iz++) {
      mycoeffz[iz] = mycoeffz[iz] - mycoeffz1[iz];
    }
  }
}

float Receiver::extract_value(float *wf, int ir, bool mirror_ghost) {
  float val = 0;
  float *__restrict kx = &(kernels_x[ir][0]);
  float *__restrict ky = &(kernels_y[ir][0]);
  float *__restrict kz = &(kernels_z[ir][0]);
  int lx = lxs[ir], ly = lys[ir], lz = lzs[ir];
  int offx = offxs[ir], offy = offys[ir], offz = offzs[ir];
  int iz_mirror = grid->getIDz(zsurf);
  int nr = traces->getNReceivers();
  // if(ir == nr / 2) print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d, mirror_ghost=%d\n", offz, lz, iz_mirror, iz0, mirror_ghost);
  for(int iy = 0; iy < ly; iy++) {
    float cy = ky[iy];
    for(int ix = 0; ix < lx; ix++) {
      float cx = cy * kx[ix];
      {
        size_t off = (((size_t)iy + offy) * nx + ix + offx) * nz + offz;
        for(int iz = 0; iz < lz; iz++)
          val += cx * kz[iz] * wf[off + iz];
      }

      if(mirror_ghost) {
        size_t off2 = (((size_t)iy + offy) * nx + ix + offx) * nz + 2 * iz_mirror - offz;
        // requires 2 * iz_mirror - offz - (lz2-1) >=0, i.e., lz2 <= 2 * iz_mirror - offz + 1
        int lz2 = min(lz, 2 * iz_mirror - offz + 1);
        for(int iz = 0; iz < lz2; iz++)
          val -= cx * kz[iz] * wf[off2 - iz];
      }

    }
  }

  return val;
}

void Receiver::apply_src_omp(int it, float scaler) {
  int nr = traces->getNReceivers();
  int tid = omp_get_thread_num();
  float *__restrict buf = &buf_omp[tid][0];

  memset(buf, 0, sizeof(float) * buf_omp[tid].size());
#pragma omp for
  for(int ir = 0; ir < nr; ir++) {
    float val = traces->data[ir][it] * scaler;
    float *__restrict kx = &(kernels_x[ir][0]);
    float *__restrict ky = &(kernels_y[ir][0]);
    float *__restrict kz = &(kernels_z[ir][0]);
    int lx = lxs[ir], ly = lys[ir], lz = lzs[ir];
    int offx = offxs[ir], offy = offys[ir], offz = offzs[ir];
    for(int iy = 0; iy < ly; iy++) {
      float cy = val * ky[iy];
      for(int ix = 0; ix < lx; ix++) {
        float cx = cy * kx[ix];
        {
          size_t off = (((size_t)iy + offy) * nx + ix + offx) * nz + offz;
          size_t offb = (((size_t)iy + offy) * nx + ix + offx) * nsz + offz - iz0;
          for(int iz = 0; iz < lz; iz++)
            buf[offb + iz] += cx * kz[iz];
        }
      }
    }
  }
}

void Receiver::combine_src_omp(float *wf, float *vdt2, bool mirror_ghost) {
  int iz_mirror = grid->getIDz(zsurf);
  int offz = mirror_ghost ? 2 * iz_mirror - iz0 : iz0;
  int nsz = this->nsz;
  if(mirror_ghost && 2 * iz_mirror - iz1 < 0) nsz += 2 * iz_mirror - iz1;
#pragma omp for collapse(2)
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      size_t off = ((size_t)iy * nx + ix) * nz + offz;
      size_t offb = ((size_t)iy * nx + ix) * nsz;
      for(int tid = 0; tid < nThreads; tid++) {
        if(!mirror_ghost) for(int iz = 0; iz < nsz; iz++)
          wf[off + iz] += buf_omp[tid][offb + iz] * vdt2[off + iz];
        else for(int iz = 0; iz < nsz; iz++)
          wf[off - iz] -= buf_omp[tid][offb + iz] * vdt2[off - iz];
      }
    }
  }
}

void Receiver::apply_src(float *wf, int ir, float val, bool mirror_ghost, float *vdt2) {
  float *__restrict kx = &(kernels_x[ir][0]);
  float *__restrict ky = &(kernels_y[ir][0]);
  float *__restrict kz = &(kernels_z[ir][0]);
  int lx = lxs[ir], ly = lys[ir], lz = lzs[ir];
  int offx = offxs[ir], offy = offys[ir], offz = offzs[ir];
  int iz_mirror = grid->getIDz(zsurf);

  int nr = traces->getNReceivers();
  //  if (ir == nr / 2) print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d\n", offz, lz, iz_mirror, iz0);
  for(int iy = 0; iy < ly; iy++) {
    float cy = val * ky[iy];
    for(int ix = 0; ix < lx; ix++) {
      float cx = cy * kx[ix];
      {
        size_t off = (((size_t)iy + offy) * nx + ix + offx) * nz + offz;
        for(int iz = 0; iz < lz; iz++)
          wf[off + iz] += cx * kz[iz] * vdt2[off + iz];
      }

      if(mirror_ghost) {
        size_t off2 = (((size_t)iy + offy) * nx + ix + offx) * nz + 2 * iz_mirror - offz;
        // requires 2 * iz_mirror - offz - (lz2-1) >=0, i.e., lz2 <= 2 * iz_mirror - offz + 1
        int lz2 = min(lz, 2 * iz_mirror - offz + 1);
        for(int iz = 0; iz < lz2; iz++)
          wf[off2 - iz] -= cx * kz[iz] * vdt2[off2 - iz];
      }

    }
  }
}

void Receiver::apply_src_coefxy(float *wf, float *trace, int ir, int nt, int nw) {
  float *__restrict kx = &(kernels_x[ir][0]);
  float *__restrict ky = &(kernels_y[ir][0]);
  int lx = lxs[ir], ly = lys[ir];
  int offx = offxs[ir], offy = offys[ir];

//  print1m("lx=%d, ly=%d, kx_size=%ld, ky_size=%ld, kx=%p, ky=%p, wf=%p, offx=%d, offy=%d\n", lx, ly, kernels_x[ir].size(),
//         kernels_y[ir].size(), kx, ky, wf, offx, offy);

  int nr = traces->getNReceivers();
  //  if (ir == nr / 2) print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d\n", offz, lz, iz_mirror, iz0);
  for(int iy = 0; iy < ly; iy++) {
    for(int ix = 0; ix < lx; ix++) {
      {
        size_t off = (((size_t)iy + offy) * nx + ix + offx) * nw;
        for(int it = 0; it < nt; it++)
          wf[off + it] += trace[it] * kx[ix] * ky[iy];
      }
    }
  }
}
// do not put additional omp loops here
void Receiver::spreadReceiver(const vector3 &coord, const float *data, int itMin, int itMax, float *vol, bool do_dipole) {
  if(spreadSize > 0) {
    if(myInterp) {
      float z = grid->getIDzf(coord.z);
      int spread_limz = spreadSizeZ * 2;
      vector<float> coeffz(spread_limz);
      int offz = Util::CompactOrmsbySpreadCoeff(nsz, coeffz, z - iz0, khpass, khcut, &spread_limz, do_dipole);
      int ixp = grid->getIDx(coord.x, coord.z);
      int iyp = grid->getIDy(coord.y, coord.z);
      for(int iz = 0; iz < spread_limz; iz++) {
        putTrace(ixp, iyp, iz + offz, data, itMin, itMax, vol, coeffz[iz]);
      }
    } else {
      float x = grid->getIDxf(coord.x, coord.z);
      float y = grid->getIDyf(coord.y, coord.z);
      float z = grid->getIDzf(coord.z);
      int spread_limx = nx == 1 ? 1 : spreadSize * 2;
      int spread_limy = ny == 1 ? 1 : spreadSize * 2;
      int spread_limz = spreadSizeZ * 2;
      vector<float> coeffx(spread_limx);
      vector<float> coeffy(spread_limy);
      vector<float> coeffz(spread_limz); // Note: ghosts not handled, probably no need either
      // offsets are relative to iz0 (minimum iz_rec - spread)
      int offx = Util::CompactOrmsbySpreadCoeff(nx, coeffx, x - ix0, khpass, khcut, &spread_limx);
      int offy = Util::CompactOrmsbySpreadCoeff(ny, coeffy, y - iy0, khpass, khcut, &spread_limy);
      int offz = Util::CompactOrmsbySpreadCoeff(nsz, coeffz, z - iz0, khpass, khcut, &spread_limz, do_dipole);
      //    print1m("coord.x=%f, ix=%f, ix0=%d, nx=%d, offx=%d, coeffx: ", coord.x, x, ix0, nx, offx);
      //    Util::print_vector(coeffx, spread_limx);
      //    print1m("ny=%d, offy=%d, coeffy: ", ny, offy);
      //    Util::print_vector(coeffy, spread_limy);
      //      print1m("z-iz0:%f, nsz=%d, lz=%d, offz=%d, coeffz: ", z-iz0, nsz, spread_limz, offz);
      //      Util::print_vector(coeffz, spread_limz);
      //      exit(1);

      // nsz size did not include ghost, handle boundaries at apply() function

      for(int iy = 0; iy < spread_limy; iy++) {
        for(int ix = 0; ix < spread_limx; ix++) {
          for(int iz = 0; iz < spread_limz; iz++) {
            float coeff = coeffx[ix] * coeffy[iy] * coeffz[iz];
            putTrace(ix + offx, iy + offy, iz + offz, data, itMin, itMax, vol, coeff);
          }
        }
      }
    }
  } else {
    int ixp = grid->getIDx(coord.x, coord.z);
    int iyp = grid->getIDy(coord.y, coord.z);
    int izp = grid->getIDz(coord.z);
    // if trace is not in grid, skip it
    if(ixp < ix0 || ixp > ix1 || iyp < iy0 || iyp > iy1 || izp < iz0 || izp > iz1) return;
    putTrace(ixp, iyp, izp, data, itMin, itMax, vol);
  }
}

void Receiver::taperTrace(float *vol, const float *wtaperXY, int ntaperXY) {
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      for(int iz = 0; iz < nsz; iz++) {
        size_t id = ((size_t)iy * (size_t)nx + ix) * (size_t)nsz + iz;
        if(ny > ntaperXY * 2) {
          if(iy < ntaperXY) vol[id] *= wtaperXY[iy];
          if(iy > ny - ntaperXY - 1) vol[id] *= wtaperXY[ny - iy - 1];
        }
        if(nx > ntaperXY * 2) {
          if(ix < ntaperXY) vol[id] *= wtaperXY[ix];
          if(ix > nx - ntaperXY - 1) vol[id] *= wtaperXY[nx - ix - 1];
        }
      }
    }
  }
}

// z!=0: also do deghost
void Receiver::get_wavenumber(int nkx, float dkx, float *kx) {

  int mkx = nkx / 2 + 1;
  int i = 0;
  if(nkx == 1) {
    kx[0] = 0.0f;
  } else {
    for(i = 0; i < mkx; i++)
      kx[i] = i * dkx;
    for(i = mkx; i < nkx; i++)
      kx[i] = (i - nkx) * dkx;
  }
}

float* Receiver::get_ktaper(int nkx, float *kx, float k1, float k2) {
  float *ktaper = new float[nkx]();
  int mkx = int(nkx / 2 + 1);
  float kmax = fabs(kx[mkx - 1]);
  float kmax1 = kmax * k1 * 2;
  float kmax2 = kmax * k2 * 2;
  for(int ik = 0; ik < nkx; ++ik) {
    float akx = fabs(kx[ik]);
    if(akx < kmax1) {
      ktaper[ik] = 1.0;
    } else if(akx > kmax2) {
      ktaper[ik] = 0.0;
    } else {
      float tmp = (akx - kmax1) / (kmax2 - kmax1) * M_PI / 2.0;
      ktaper[ik] = cos(tmp) * cos(tmp);
      //      ktaper[ik] = (kmax2-akx)/(kmax2-kmax1);
    }
    //    print1m("ik=%d, kx=%f, ktaper=%f, kmax1=%f, kmax2=%f \n", ik, kx[ik], ktaper[ik], kmax1, kmax2);
  }
  return ktaper;
}

void Receiver::spread_kdomain(float vsurf, int trueamp, float z, int deghost) {
  int verbose = global_pars["verbose"].as<int>(0);
  int nky = (ny == 1) ? 1 : libfftv::padfftv(ny);
  int nkx = (nx == 1) ? 1 : libfftv::padfftv(nx);
  assertion(nkx == nx && nky == ny, "nx=%d, ny=%d must be pre-padded with padfftv()!", nx, ny);
  float ratio = global_pars["tpad_ratio"].as<float>(0.5f);
  int ntfft = libfftv::padfftv(nearbyintf(nt * (1 + ratio)));
  int iw_lc = nearbyintf(flcut * dt * ntfft);
  int iw_lp = nearbyintf(flpass * dt * ntfft);
  print1m("in spread_kdomain: vsurf=%f, z_depth=%f, trueamp=%d, deghost=%d \n", vsurf, z, trueamp, deghost);
  print1m("in fk_operation: nt=%d, nx=%d, ny=%d \n", nt, nx, ny);
  print1m("in fk_operation: ntfft=%d (iwlc=%d,iwlp=%d), nkx=%d, nky=%d \n", ntfft, iw_lc, iw_lp, nkx, nky);
  float dxu = grid->getdx(0.0);
  float dyu = grid->getdy(0.0);
  print1m("in fk_operation: dt=%f, dx=%f, dy=%f \n", dt, dxu, dyu);

  float dkxu, dkyu, dwv;
  int idx;

  int nw = ntfft / 2 + 1;
  if(nkx == 1) {
    dkxu = 0;
    dkyu = 2.0f * M_PI / (dyu * nky);
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 1;
  } else if(nky == 1) {
    dkxu = 2.0f * M_PI / (dxu * nkx);
    dkyu = 0;
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 2;
  } else {
    dkxu = 2.0f * M_PI / (dxu * nkx);
    dkyu = 2.0f * M_PI / (dyu * nky);
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 3;
  }
  size_t nxyw = (size_t)nkx * nky * nw;
  size_t nxyt = (size_t)nx * ny * nt;
  float scalar = sqrt(1.0 / nkx / nky / ntfft);

  if(verbose) print1m("in fk_operation: dkx=%f, dky=%f, dwv=%f \n", dkxu, dkyu, dwv);
  if(verbose) print1m("in fk_operation: nxyw=%zu, scalar=%f \n", nxyw, scalar);
  if(transpose_fk) print1m("FK operation requires %fG buffer and %fG output (transpose_fk=true)\n", nxyw / 1e9 * sizeof(Complex),
                           nxyt / 1e9 * sizeof(float));
  else print1m("FK operation requires %fG buffer/output (transpose_fk=false)\n", nxyw / 1e9 * sizeof(Complex));

  int ik;
  float *kx = new float[nkx];
  float *ky = new float[nky];
  float *kw = new float[nw * 2];

  fftwf_plan planf, planb;

  vector<float> vbuf(nxyw * 2);
  Complex *cdata = (Complex*)&vbuf[0];
  float *data = (float*)&vbuf[0];

  switch(idx) {
  case 1:
    planf = fftwf_plan_dft_r2c_2d(nky, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_2d(nky, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  case 2:
    planf = fftwf_plan_dft_r2c_2d(nkx, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_2d(nkx, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  case 3:
    planf = fftwf_plan_dft_r2c_3d(nky, nkx, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_3d(nky, nkx, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  }

  for(ik = 0; ik < nw; ik++) {
    kw[ik] = ik * dwv;
  }

  Complex *trace_w = new Complex[nw]();
  float *trace_t = (float*)trace_w;
  fftwf_plan plant = fftwf_plan_dft_r2c_1d(ntfft, (float*)trace_t, (fftwf_complex*)trace_w, FFTW_MEASURE);

  float *kx_taper = get_ktaper(nkx, kx, khpass, khcut);
  float *ky_taper = get_ktaper(nky, ky, khpass, khcut);

  if(!bufferToDisk) {
    //int iz, ix, iy, it;
    size_t nxz = (size_t)nx * nsz;
    size_t nxyz = (size_t)nx * nsz * ny;
    size_t nkxy = (size_t)nkx * nky;

    size_t id0, id1;

    //spread xy in k
    int nr = traces->getNReceivers();
    float x0 = grid->x0;
    float y0 = grid->y0;

    const std::complex<float> COMPLEX_I(0.0, 1.0);

    for(int ir = 0; ir < nr; ir++) {
      float x = traces->coord[ir].x;
      float y = traces->coord[ir].y;

      for(int it = 0; it < nt; it++)
        trace_t[it] = traces->data[ir][it] * scalar;
      for(int it = nt; it < ntfft; it++)
        trace_t[it] = 0.0f;

      fftwf_execute_dft_r2c(plant, (float*)trace_t, (fftwf_complex*)trace_w);

      if(nky > 1) {
        for(int iky = 0; iky < nky; iky++) {
          float phasey = (y - y0) * ky[iky];
          Complex shifty = cosf(phasey) - COMPLEX_I * sinf(phasey);
          for(int ikx = 0; ikx < nkx; ikx++) {
            float phasex = (x - x0) * kx[ikx];
            Complex shiftx = cosf(phasex) - COMPLEX_I * sinf(phasex);
            Complex ctmp = shiftx * kx_taper[ikx] * shifty * ky_taper[iky];
            size_t id = (size_t)iky * nkx * nw + (size_t)ikx * nw;
            for(int iw = 0; iw < nw; iw++) {
              cdata[id + iw] += trace_w[iw] * ctmp;
            }
          }
        }
      } else {
        for(int ikx = 0; ikx < nkx; ikx++) {
          float phasex = (x - x0) * kx[ikx];
          Complex shiftx = cosf(phasex) - COMPLEX_I * sinf(phasex);
          Complex ctmp = shiftx * kx_taper[ikx];
          size_t id = (size_t)ikx * nw;
          for(int iw = 0; iw < nw; iw++) {
            cdata[id + iw] += trace_w[iw] * ctmp;
          }
        }
      }
    }

    if(global_pars["qc_srcfk"])
      jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd0" : "_fwd0").c_str(), data, nw * 2, nx, ny, dt * 1000);

    float ghost_zerofreq_threshold = global_pars["ghost_zerofreq_threshold"].as<float>(0.1f);
    float fmax = recordLoader->hpFreq;
    float kzdzmax = fmax * FLT_2_PI / vsurf * z;
    if(kzdzmax < FLT_PI_OVER_2) ghost_zerofreq_threshold *= sinf(kzdzmax); // normalize the threshold, review this option when using with FWI
    int sign = (id == RECEIVER) ? -1 : 1;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nkxy; ixy++) {
      int ix = ixy % nkx;
      int iy = ixy / nkx;
      Complex *cdat = cdata + (size_t)iy * nkx * nw + (size_t)ix * nw;
      cdat[0] = 0;
      for(int iw = 1; iw < nw; iw++) {
        float k = kw[iw];
        float tmp = 1.0f - (kx[ix] * kx[ix] + ky[iy] * ky[iy]) / (k * k);
        if(tmp > 0.0) {
          float cosv = sqrtf(tmp);
          float factor = trueamp ? cosv : 1;
          if(deghost) {
            float kzdz = k * cosv * z;
            float sinv = sinf(kzdz);
           factor *= kzdz >= FLT_PI_OVER_2 ? 1 : sinv <= ghost_zerofreq_threshold ? 1 / ghost_zerofreq_threshold : 1 / sinv;
            factor *= 0.5;
            if(iw <= iw_lc) factor = 0;
            else if(iw <= iw_lp) factor *= 0.5f * (1 - cosf(FLT_PI * (iw - iw_lc) / (iw_lp - iw_lc + 1)));
            cdat[iw] *= Complex { 0, factor * sign };
          } else cdat[iw] *= factor;
        } else cdat[iw] = 0;
      }
    }

    fftwf_execute_dft_c2r(planb, (fftwf_complex*)cdata, (float*)data);

    if(global_pars["qc_srcfk"])
      jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd1" : "_fwd1").c_str(), data, nw * 2, nx, ny, dt * 1000);

    if(transpose_fk) {
      spread_data.resize(nxyt);
      Util::transpose(nt, nx * ny, nw * 2, nx * ny, data, &spread_data[0]);
    } else {
      size_t nxy = (size_t)nx * ny;
      for(size_t ixy = 1; ixy < nxy; ixy++)
        memmove(&vbuf[ixy * nt], &vbuf[ixy * nw * 2], sizeof(float) * nt);
      vbuf.resize(nxyt), vbuf.shrink_to_fit(); // release the extra memory
      std::swap(spread_data, vbuf);
      if(global_pars["qc_srcfk"])
        jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd2" : "_fwd2").c_str(), &spread_data[0], nt, nx, ny, dt * 1000);
    }

  } else {
    print1m("This part is not developed. Error in Receiver::trueamplitude \n");
    exit(-1);
  }

  fftwf_destroy_plan(planf);
  fftwf_destroy_plan(planb);
  fftwf_destroy_plan(plant);

  delete[] kx;
  delete[] ky;
  delete[] kw;
  delete[] kx_taper;
  delete[] ky_taper;
  delete[] trace_w;
}

void Receiver::spread_sdomain(float vsurf, int trueamp, float z, int deghost) {
  int verbose = global_pars["verbose"].as<int>(0);
  int nky = (ny == 1) ? 1 : libfftv::padfftv(ny);
  int nkx = (nx == 1) ? 1 : libfftv::padfftv(nx);
  assertion(nkx == nx && nky == ny, "nx=%d, ny=%d must be pre-padded with padfftv()!", nx, ny);
  float ratio = global_pars["tpad_ratio"].as<float>(0.5f);
  int ntfft = libfftv::padfftv(nearbyintf(nt * (1 + ratio)));
  int iw_lc = nearbyintf(flcut * dt * ntfft);
  int iw_lp = nearbyintf(flpass * dt * ntfft);
  print1m("in spread_sdomain: vsurf=%f, z_depth=%f, trueamp=%d, deghost=%d \n", vsurf, z, trueamp, deghost);
  print1m("in fk_operation: nt=%d, nx=%d, ny=%d \n", nt, nx, ny);
  print1m("in fk_operation: ntfft=%d (iwlc=%d,iwlp=%d), nkx=%d, nky=%d \n", ntfft, iw_lc, iw_lp, nkx, nky);
  float dxu = grid->getdx(0.0);
  float dyu = grid->getdy(0.0);
  print1m("in fk_operation: dt=%f, dx=%f, dy=%f \n", dt, dxu, dyu);

  float dkxu, dkyu, dwv;
  int idx;

  int nw = ntfft / 2 + 1;
  if(nkx == 1) {
    dkxu = 0;
    dkyu = 2.0f * M_PI / (dyu * nky);
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 1;
  } else if(nky == 1) {
    dkxu = 2.0f * M_PI / (dxu * nkx);
    dkyu = 0;
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 2;
  } else {
    dkxu = 2.0f * M_PI / (dxu * nkx);
    dkyu = 2.0f * M_PI / (dyu * nky);
    dwv = 2.0f * M_PI / (dt * ntfft * vsurf);
    idx = 3;
  }
  size_t nxyw = (size_t)nkx * nky * nw;
  size_t nxyt = (size_t)nx * ny * nt;
  float scalar = sqrt(1.0 / nkx / nky / ntfft);

  if(verbose) print1m("in fk_operation: dkx=%f, dky=%f, dwv=%f \n", dkxu, dkyu, dwv);
  if(verbose) print1m("in fk_operation: nxyw=%zu, scalar=%f \n", nxyw, scalar);
  if(transpose_fk) print1m("FK operation requires %fGB buffer and %fGB output (transpose_fk=true)\n",
                           nxyw / (1024.0f * 1024 * 1024) * sizeof(Complex), nxyt / (1024.0f * 1024 * 1024) * sizeof(float));
  else print1m("FK operation requires %fGB buffer/output (transpose_fk=false)\n", nxyw / (1024.0f * 1024 * 1024) * sizeof(Complex));

  int ik;
  float *kx = new float[nkx];
  float *ky = new float[nky];
  float *kw = new float[nw * 2];

  fftwf_plan planf, planb;
  vector<float> vbuf(nxyw * 2);
  Complex *cdata = (Complex*)&vbuf[0];
  float *data = (float*)&vbuf[0];

  switch(idx) {
  case 1:
    planf = fftwf_plan_dft_r2c_2d(nky, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_2d(nky, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  case 2:
    planf = fftwf_plan_dft_r2c_2d(nkx, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_2d(nkx, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  case 3:
    planf = fftwf_plan_dft_r2c_3d(nky, nkx, ntfft, (float*)data, (fftwf_complex*)cdata, FFTW_MEASURE);
    planb = fftwf_plan_dft_c2r_3d(nky, nkx, ntfft, (fftwf_complex*)cdata, (float*)data, FFTW_MEASURE);
    get_wavenumber(nkx, dkxu, kx);
    get_wavenumber(nky, dkyu, ky);
    break;
  }

  for(ik = 0; ik < nw; ik++) {
    kw[ik] = ik * dwv;
  }

  if(!bufferToDisk) {
    //int iz, ix, iy, it;
    size_t nxz = (size_t)nx * nsz;
    size_t nxyz = (size_t)nx * nsz * ny;
    size_t nkxy = (size_t)nkx * nky;

    size_t id0, id1;

    //spread xy in k
    int nr = traces->getNReceivers();
    memset(data, 0, sizeof(float) * nxyw * 2);

    for(int ir = 0; ir < nr; ir++) {
      apply_src_coefxy(data, traces->data[ir], ir, nt, nw * 2);
    }

    if(global_pars["qc_srcfk"])
      jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd0" : "_fwd0").c_str(), data, nw * 2, nx, ny, dt * 1000);

    fftwf_execute_dft_r2c(planf, (float*)data, (fftwf_complex*)cdata);

    float ghost_zerofreq_threshold = global_pars["ghost_zerofreq_threshold"].as<float>(0.1f);
    int sign = (id == RECEIVER) ? -1 : 1;
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(size_t ixy = 0; ixy < nkxy; ixy++) {
      int ix = ixy % nkx;
      int iy = ixy / nkx;
      Complex *cdat = cdata + (size_t)iy * nkx * nw + (size_t)ix * nw;
      cdat[0] = 0;
      for(int iw = 1; iw < nw; iw++) {
        float k = kw[iw];
        float tmp = 1.0f - (kx[ix] * kx[ix] + ky[iy] * ky[iy]) / (k * k);
        if(tmp > 0.0) {
          float cosv = sqrtf(tmp);
          float factor = trueamp ? cosv : 1;
          factor *= scalar;
          if(deghost) {
            float kzdz = k * cosv * z;
            float sinv = sinf(kzdz);
            factor *= kzdz >= FLT_PI_OVER_2 ? 1 : sinv <= ghost_zerofreq_threshold ? 1 / ghost_zerofreq_threshold : 1 / sinv;
            factor *= 0.5;
            if(iw <= iw_lc) factor = 0;
            else if(iw <= iw_lp) factor *= 0.5f * (1 - cosf(FLT_PI * (iw - iw_lc) / (iw_lp - iw_lc + 1)));
            cdat[iw] *= Complex { 0, factor * sign };
          } else cdat[iw] *= factor;
        } else cdat[iw] = 0;
      }
    }

    fftwf_execute_dft_c2r(planb, (fftwf_complex*)cdata, (float*)data);

    if(global_pars["qc_srcfk"])
      jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd1" : "_fwd1").c_str(), data, nw * 2, nx, ny, dt * 1000);

    if(transpose_fk) {
      spread_data.resize(nxyt);
      Util::transpose(nt, nx * ny, nw * 2, nx * ny, data, &spread_data[0]);
    } else {
      size_t nxy = (size_t)nx * ny;
      for(size_t ixy = 1; ixy < nxy; ixy++)
        memmove(&vbuf[ixy * nt], &vbuf[ixy * nw * 2], sizeof(float) * nt);
      vbuf.resize(nxyt), vbuf.shrink_to_fit(); // release the extra memory
      std::swap(spread_data, vbuf);
      if(global_pars["qc_srcfk"])
        jseisUtil::save_zxy(getJsFilename("qc_srcfk", id == RECEIVER ? "_bwd2" : "_fwd2").c_str(), &spread_data[0], nt, nx, ny, dt * 1000);
    }

  } else {
    print1m("This part is not developed. Error in Receiver::trueamplitude \n");
    exit(-1);
  }

  fftwf_destroy_plan(planf);
  fftwf_destroy_plan(planb);

  delete[] kx;
  delete[] ky;
  delete[] kw;
}

void Receiver::spread() {
  int ntaperXY = 10;

  int do_trueamp = trueamp & (id + 1);
  int do_deghost = deghost & (id + 1);
  int true_amp_dipole = getBool(global_pars["true_amplitude_dipole"], false); // !do_deghost);
  int do_trueamp_fk = do_trueamp && !true_amp_dipole;
  int do_dipole = do_trueamp && true_amp_dipole;

  did_dipole = false;
  if(!use_src4d) {
    vector<float> zstats = spreadCoeffs(ntaperXY, do_dipole);
    float zmean = zstats[0], zdev = zstats[1];

    float vsurf = global_pars["vsurf"].as<float>();
    post_spread_correction();

    // id is only 0 and 1 (see constructor, 2 is converted to 0)
    print1m("true_amplitude=%d, id=%d, do_trueamp=%d, do_trueamp_fk=%d, do_dipole=%d\n", trueamp, id, do_trueamp, do_trueamp_fk, do_dipole);
    // print1m("traces=%p, dt=%f\n", traces, traces->dt);
    // if(do_dipole) traces->qcData("/tmp/traces0.js");
    if(do_trueamp_fk || do_deghost) {
      assertion(zdev < 0.5f, "FK operation not supported when REC_ELEV deviation(%f) is significant!", zdev);
      string str = global_pars["fk_domain"].as<string>("SDOMAIN");
      std::transform(str.begin(), str.end(), str.begin(), ::toupper);
      if(str == "KDOMAIN") spread_kdomain(vsurf, do_trueamp_fk, zmean, do_deghost);
      else spread_sdomain(vsurf, do_trueamp_fk, zmean, do_deghost);

      traces.reset(); // The raw receiver data is no longer needed
    }

    return;
  }

  float *wtaperXY = new float[ntaperXY];
  for(int i = 0; i < ntaperXY; i++) {
    float temp = M_PI * float(i) / float(ntaperXY);
    wtaperXY[i] = 0.5f * (1.0f - cosf(temp));
  }

// Do a chunk of the time steps at a time
  int BLOCK_SIZE = 100;
  if(!bufferToDisk) {
    // Use large block size when not buffering to disk
    // since we most likely have plenty of space available
    BLOCK_SIZE = std::min(nt, 1000);
  }

  int nPasses = (nt / BLOCK_SIZE) + 1;

  print1m("nt = %d with %d it per pass ==> %d passes.\n", nt, BLOCK_SIZE, nPasses);
  int nr = traces->getNReceivers();

  if(global_pars["print_asum_receiver"])
    for(int ir = 0; ir < nr; ir++) {
      print1m("%d (%f,%f,%f) --> %g\n", ir, traces->coord[ir].x, traces->coord[ir].y, traces->coord[ir].z,
              Util::asumf(traces->data[ir], nt, 1));
    }

  if(global_pars["qc_receiver"]) {
    string qcfile = expEnvVars(global_pars["qc_receiver"].as<string>());
    vector<float> buf((size_t)nr * nt);
    for(int ir = 0; ir < nr; ir++)
      memcpy(&buf[(size_t)ir * nt], traces->data[ir], sizeof(float) * nt);
    jseisUtil::save_zxy(qcfile.c_str(), &buf[0], nt, nr, 1, dt * 1000, 1, 1);
    if(global_pars["qc_receiver_only"].as<int>(0)) exit(-1);  // QC only, end it sooner
  }

  size_t volumeSize = (size_t)nx * ny * nsz;
  for(int pass = 0; pass < nPasses; pass++) {
    int itMin = pass * BLOCK_SIZE;
    int itMax = MIN(itMin + BLOCK_SIZE, nt);
    int ntChunck = itMax - itMin;

    float *chunk = bufferToDisk ? new float[ntChunck * volumeSize]() : &vol4d[pass * BLOCK_SIZE * volumeSize];

    print1m("Pass %d [%d, %d] %d\n", pass, itMin, itMax, ntChunck);

    // loop all the traces
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ir = 0; ir < nr; ir++) {
      spreadReceiver(traces->coord[ir], traces->data[ir], itMin, itMax, chunk, do_dipole);
    }

    // scale the trace
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int it = 0; it < ntChunck; it++) {
      taperTrace(&chunk[it * volumeSize], wtaperXY, ntaperXY);
    }
    // Store the preprocessed data
    if(bufferToDisk) {
      vol4dFile.write((char*)chunk, ntChunck * volumeSize * sizeof(float));
      delete[] chunk;
    }

  }

  if(global_pars["qc_vol4d"] && !bufferToDisk && ny == 1) {
    string qcfile = expEnvVars(global_pars["qc_vol4d"].as<string>());
    jseisUtil::save_zxy(qcfile.c_str(), vol4d, nsz, nx, nt, grid->dz, grid->dx, dt, iz0, 0, 0, ix0, 1);
  }

  delete[] wtaperXY;

  traces.reset();  // The raw receiver data is no longer needed

  /*
   if(DEBUG_RECV) {
   float* buffer = new float[nx*nt];
   Util::transpose(nx, nt, nx, nt, vol4d, buffer);
   saveFdmCube(buffer, nx, 1, nt, 10.0f, 10.0f, dt, "receiver01.fdm");
   delete[] buffer;
   }
   */
}

void Receiver::putTrace(int ix, int iy, int iz, const float *buffs, int itMin, int itMax, float *vol, float scaler) {
  float s = scaler * dt_dxyz;
  size_t nxyz = (size_t)nx * ny * nsz;

  size_t idoffset = ((size_t)iy * nx + ix) * nsz + iz;
  for(int it = itMin; it < itMax; it++) {
    size_t id = (it - itMin) * nxyz + idoffset;
#pragma omp atomic
    vol[id] += buffs[it] * s; //accumulate
  }
}

float* Receiver::getDataForIteration(int it) {
  size_t nxyz = (size_t)nx * ny * nsz;

  if(bufferToDisk) {
    // Read more data than the current iteration, and cache it in memory
    int cacheIndex = (nt - it - 1) % CACHED_IN_MEMORY;
    int nElementsPerSlice = nxyz;

    // Check if we have already read the data we need
    if(cacheIndex == 0 || it == nt - 1) {
      int rest = (((nt - it) / CACHED_IN_MEMORY) + 1) * CACHED_IN_MEMORY;
      int nElementsToCopy = std::min(CACHED_IN_MEMORY, rest);

      int cacheStart = 0;
      int idx = it - nElementsToCopy + 1;
      // Special handling for the last part which is smaller than the cache:
      // Copy in to the cache with an offset.
      if(idx < 0) {
        idx = 0;
        nElementsToCopy = it + 1;
        cacheStart = CACHED_IN_MEMORY - nElementsToCopy;
      }

      vol4dFile.seekg(nElementsPerSlice * idx * sizeof(float));
      vol4dFile.read((char*)&buffer[nElementsPerSlice * cacheStart], nElementsToCopy * nxyz * sizeof(float));
    }

    int cachedIndex = CACHED_IN_MEMORY - cacheIndex - 1;
    return &buffer[size_t(cachedIndex) * nElementsPerSlice];
  } else {
    return vol4d + size_t(it) * nxyz;
  }
}

// no longer needed, for reference only (transpose_fk = true)
void Receiver::apply_src_coefz(float *wf, int ir, float *val, float factor, bool mirror_ghost, float *vdt2) {
  float *__restrict kz = &(kernels_z[ir][0]);
  int lz = lzs[ir];
  int offz = offzs[ir];
  int iz_mirror = grid->getIDz(zsurf);
  size_t nxy = (size_t)nx * ny;

  //  if (ir == nr / 2) print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d\n", offz, lz, iz_mirror, iz0);
  //print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d\n", offz, lz, iz_mirror, iz0);
  for(size_t ixy = 0; ixy < nxy; ixy++) {
    apply_src_coefz_lz(wf + ixy * nz, val[ixy] * factor, kz, lz, offz, iz_mirror, mirror_ghost, vdt2);
  }
}

void Receiver::apply_src_coefz_lz(float *wf, float val, float *__restrict kz, int lz, int offz, int iz_mirror, bool mirror_ghost,
    float *vdt2) {
  //print1m("ix=%d, iy=%d \n", ix, iy);
  for(int iz = 0; iz < lz; iz++)
    wf[offz + iz] += val * kz[iz] * vdt2[offz + iz];

  if(mirror_ghost) {
    size_t off2 = 2 * iz_mirror - offz;
    // requires 2 * iz_mirror - offz - (lz2-1) >=0, i.e., lz2 <= 2 * iz_mirror - offz + 1
    int lz2 = min(lz, 2 * iz_mirror - offz + 1);
    for(int iz = 0; iz < lz2; iz++)
      wf[off2 - iz] -= val * kz[iz] * vdt2[off2 - iz];
  }
}

void Receiver::apply(Wavefield *myWavefield, int it, bool ghost, float *vdt2, float scaler0) {
  if(it < 0 || it > nt - 1) return;
  float scaler = scaler0 * dt_dxyz;

  static int debug_apply = global_pars["debug_apply_force"].as<int>(0);
  float *__restrict waveP0 = myWavefield->w0;  // revised from wb to w0
  size_t nxy = (size_t)nx * ny;
  if(!use_src4d) {
    int do_trueamp = trueamp & (id + 1);
    int do_deghost = deghost & (id + 1);
    int true_amp_dipole = getBool(global_pars["true_amplitude_dipole"], false); // !do_deghost);
    int do_trueamp_fk = do_trueamp && !true_amp_dipole;
    int do_dipole = do_trueamp && true_amp_dipole;
    if(do_trueamp_fk || do_deghost) {
      int iz_mirror = grid->getIDz(zsurf);
      float *__restrict kz = &(kernels_z[0][0]);
      int lz = lzs[0];
      int offz = offzs[0];
      for(size_t ixy = 0; ixy < nxy; ixy++) {
        float val = transpose_fk ? spread_data[it * nxy + ixy] : spread_data[ixy * nt + it];
        apply_src_coefz_lz(waveP0 + ixy * nz, val * scaler, kz, lz, offz, iz_mirror, ghost, vdt2);
      }
    } else {
      int nr = traces->getNReceivers();
#if 0
      static int saved;
      if(nr > 1 && !saved) traces->qcData("/tmp/traces.js"), saved = 1;
#endif
      if(debug_apply > 0) debug_apply = -1, apply_src(waveP0, 0, 1.0f, ghost, vdt2);
      else if(debug_apply == 0) {
        bool omp_src = getBool("omp_src", true) && nr >= nThreads;
        if(omp_src) {
          if(buf_omp.empty()) {
            size_t size_buf = (size_t)nsz * nx * ny;
            print1m("Receiver: allocating buf_omp=%d*%d*%d*%d(%fGB)\n", nThreads, nsz, nx, ny,
                    nThreads * size_buf * sizeof(float) / 1024.0f / 1024 / 1024);
            buf_omp.resize(nThreads, vector<float>(size_buf));
          }
#pragma omp parallel num_threads(nThreads)
          apply_src_omp(it, scaler);
          combine_src_omp(waveP0, vdt2, false);
          if(ghost) combine_src_omp(waveP0, vdt2, true);
        } else for(int ir = 0; ir < nr; ir++)  // no OMP
          apply_src(waveP0, ir, traces->data[ir][it] * scaler, ghost, vdt2);
      }
    }

    return;
  }

// use_src4d (very old option)
  float *dapp = getDataForIteration(it);

#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    int idy = iy0 + iy;
    for(int ix = 0; ix < nx; ix++) {
      int idx = ix0 + ix;
      size_t idv = ((size_t)idy * nx + idx) * nz + iz0;
      size_t ids = ((size_t)iy * nx + ix) * nsz;
      for(int iz = 0; iz < nsz; iz++) {
        waveP0[idv + iz] += dapp[ids + iz];
      }

      if(ghost) {
        int iz_mirror = grid->getIDz(zsurf) - iz0; // mirror around z=0
        idv += 2 * iz_mirror;
        // requires: 2*iz_mirror + iz0 - (nsz2-1) >= 0, i.e., nsz2 <= 2*iz_mirror + iz0 + 1
        int nsz2 = min(nsz, 2 * iz_mirror + iz0 + 1);
        for(int iz = 0; iz < nsz2; iz++) {
          waveP0[idv - iz] -= dapp[ids + iz];
        }
      }
    }
  }
}

void Receiver::apply_old(Wavefield *myWavefield, int it) {
  if(it < 0 || it > nt - 1) return;

  float *waveP0 = myWavefield->wb;
  float *dapp = getDataForIteration(it);
#pragma omp parallel for num_threads(nThreads) schedule(static)
  for(int iy = 0; iy < ny; iy++) {
    int idy = iy0 + iy;
    for(int ix = 0; ix < nx; ix++) {
      int idx = ix0 + ix;
      size_t idv = (size_t)idy * nx + idx;
      idv *= nz;
      size_t ids = (size_t)iy * nx + ix;
      ids *= nsz;
      for(int iz = 0; iz < nsz; iz++) {
        int idz = iz0 + iz;
        waveP0[idv + idz] += dapp[ids + iz];
      }
    }
  }
}
int Receiver::size() {
  return traces->getNReceivers();
}

void Receiver::post_spread_correction() {
  if(traces == nullptr) return;
  if(global_pars["qc_src"]) traces->qcData(getJsFilename("qc_src", id == RECEIVER ? "_bwd" : ""));
  if(!getBool("src_correct_after_spread", false)) return;
  float gaincap = global_pars["post_spread_gaincap"].as<float>(1.5f);

  vector<float> pow_phase = Source::getSpectrumPowPhase(id == RECEIVER ? PROP::BACKWARD : PROP::FORWARD, oper, dim_bits);
  print1m("post_spread_correction: prop=%s, phase=%f\n", id == RECEIVER ? "bwd" : "fwd", pow_phase[1]);
  float w_pow = pow_phase[0], phase = pow_phase[1] * FLT_PI / 180;
  if(id == RECEIVER && (oper == PROP::MOD || oper == PROP::DEMIG)) w_pow = 0, phase = did_dipole ? -FLT_PI / 2 : 0; // MOD or DEMIG receiver, only correct for phase
  complex<float> c_phase = std::exp(complex<float>(0.0f, phase));

  float vsurf = global_pars["vsurf"].as<float>();
  float dt_spread = grid->dz / vsurf;
// print1m("dz=%f, slow0=%f\n", myGrid->dz, slow0); // dt_spread = myGrid->dz * vsurf
  if(global_pars["qc_spread_z"])
    jseisUtil::save_zxy(getJsFilename("qc_spread_z", id == RECEIVER ? "_bwd" : "").c_str(), kernels_z, dt_spread * 1000);

  float pad_ratio = 0;
  DftFilter dft0(kernels_z[0].size(), dt_spread, nt, dt, DftFilter::DFT_FFT, { }, 0, 0, 0, fhcut, pad_ratio);
  int nwcut = dft0.getNwcut();
  int ntfft = dft0.getNtfft();
  float scaler = 1.0f / ntfft;
  scaler *= scaler;
  vector<vector<float>> bufv(nThreads, vector<float>(r2c_size(ntfft)));
  vector<vector<float>> ampv(nThreads, vector<float>(nwcut));
  float dw = 2.0f * FLT_PI / (ntfft * dt);


  if(global_pars["qc_spread_z"]) {
    float *__restrict buf = &bufv[0][0];
    vector<complex<float>> bufw(nwcut);
    vector<float> bufqc(nt);
    dft0.dft_fft(&kernels_z[0][0], (float*)&bufw[0], 1, 0, nThreads, true);

    memset(&buf[0], 0, sizeof(float) * r2c_size(ntfft));
    for(int iw = 0; iw < nwcut; iw++)
      buf[iw * 2] = abs(bufw[iw]);
    fftwf_execute_dft_c2r(dft0.getPlanBwd(), (fftwf_complex*)&buf[0], &buf[0]);
    for(int i = 0; i < nt; i++)
      bufqc[i] = buf[(i - nt / 2 + ntfft) % ntfft];
    jseisUtil::save_zxy(getJsFilename("qc_spread_z", id == RECEIVER ? "_bwd_resample" : "_resample").c_str(), &bufqc[0], nt, 1, 1,
                        dt * 1000);
  }

  int nsrc = kernels_z.size();
  fftwf_plan planf = fftwf_plan_dft_r2c_1d(ntfft, &bufv[0][0], (fftwf_complex*)&bufv[0][0], FFTW_ESTIMATE);
#ifndef NO_MKL
#pragma omp parallel num_threads(nThreads) if(nsrc>1 && nThreads>1)
#endif
  {
    int tid = omp_get_thread_num();
    float *__restrict buf = &bufv[tid][0];
    complex<float> *__restrict bufc = (complex<float>*)buf;
    float *__restrict amp = &ampv[tid][0];
#ifndef NO_MKL
#pragma omp for
#endif
    for(int i = 0; i < nsrc; i++) {
      DftFilter dft(kernels_z[i].size(), dt_spread, nt, dt, DftFilter::DFT_FFT, { }, 0, 0, 0, fhcut, pad_ratio);
      int nwcut = dft.getNwcut();
      int ntfft = dft.getNtfft();
      float dw = 2.0f * FLT_PI / (ntfft * dt);
      vector<complex<float>> bufw(nwcut);
      dft.dft_fft(&kernels_z[i][0], (float*)&bufw[0], 1, 0, nThreads, true);

      memcpy(&buf[0], traces->getTrace(i), sizeof(float) * nt);
      libCommon::fft_padding((float*)&buf[0], nt, ntfft, 0);
      fftwf_execute_dft_r2c(planf, &buf[0], (fftwf_complex*)&buf[0]);
      for(int iw = 0; iw < nwcut; iw++)
        amp[iw] = abs(bufw[iw]) / scaler;

      int iw0 = nwcut / 2;
      if(did_dipole) for(int iw = iw0; iw < nwcut; iw++)
        amp[iw] /= iw;
      int iwhigh = iw0;
      float ampmax = amp[iw0];
      for(int iw = iw0; iw < nwcut; iw++)
        if(amp[iw] >= ampmax) iwhigh = iw, ampmax = amp[iw];
      float cap = ampmax / gaincap;
      for(int iw = iwhigh; iw < nwcut; iw++)
        amp[iw] = max(cap, amp[iw]);
      if(did_dipole) for(int iw = iw0; iw < nwcut; iw++)
        amp[iw] *= iw;

      amp[0] = min(amp[0], amp[1]);
      for(int iw = 0; iw < nwcut; iw++) {
        float wpow = powf(iw == 0 && w_pow <= 0 ? dw : iw * dw, w_pow);
        bufc[iw] *= wpow / amp[iw] * c_phase;
      }
      for(int iw = nwcut; iw < ntfft / 2 + 1; iw++)
        bufc[iw] = 0;
      fftwf_execute_dft_c2r(dft.getPlanBwd(), (fftwf_complex*)&buf[0], &buf[0]);
      float c0 = id == RECEIVER ? 0 : buf[0]; // inconsistent dc shift is not good for receiver side
      float *trace = traces->getTrace(i);
      for(int it = 0; it < nt; it++)
        trace[it] = buf[it] - c0;
    }
  }
  fftwf_destroy_plan(planf);

  if(global_pars["qc_src"]) traces->qcData(getJsFilename("qc_src", id == RECEIVER ? "_bwd_pss" : "_pss"));


