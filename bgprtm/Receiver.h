/*
 * Receiver.h
 *   Receiver data were in Traces but then converted to vol4d by spread() function
 */

#ifndef RECEIVER_H_
#define RECEIVER_H_

#include <fstream>
#include <string>
#include <time.h>

#include <complex>
#include <yaml-cpp/yaml.h>
using YAML::Node;

#include "Util.h"
#ifdef NO_MKL
#include <fftw3.h>
#else
#include <fftw/fftw3.h>
#endif
#include "libFFTV/numbertype.h"
#include "Source.h"
#include "RecordLoader.h"

class vector3;
class Model;
class Traces;
class Grid;
class Wavefield;

class Receiver {
public:
  enum SPREADID { // source/receiver to be consistent with: PROP::FORWARD: 0, PROP::BACKWARD: 1
    SOURCE = 0, RECEIVER = 1, USER_WAVELET = 2
  };

  enum SIDE_BITS { // used for deghost, trueamp_side
    SIDE_NONE, SIDE_SOURCE, SIDE_RECEIVER, SIDE_BOTH
  };
  /** ctor
   *
   */
  Receiver(shared_ptr<Grid> grid, Model *model, float dt, int nt, float maxFreq, float minFreq, int spreadSize, int dim, int id,
      PROP::Operation oper, bool bufferToDisk = false);

  /** dtor
   *
   */
  virtual ~Receiver();

  static SIDE_BITS getSideBits(string sou_rec_side);
  static int getTrueAmp(PROP::Operation oper);
  bool isDipole();
  static bool isDipole(PROP::Direction direction, PROP::Operation oper);
  /** add receiver data to the wavefields
   *
   */
  void apply(Wavefield *myWavefield, int it, bool ghost, float *vdt2, float dt2);
  void apply_old(Wavefield *myWavefield, int it);

  /** load header & data from record file
   *
   */
  int loadData(const char *fileName);
  int loadHdr(const char *fileName);
  int createHdrFromGrid(Node &node);
  void resetTraces(float *dat, int nt, float dt, int nr);

  float extract_value(float *wf, int ir, bool mirror_ghost);
  void apply_src_omp(int it, float scaler);
  void combine_src_omp(float *wf, float *vdt2, bool mirror_ghost);
  void apply_src(float *wf, int ir, float val, bool mirror_ghost, float *vdt2);
  void apply_src_coefz(float *wf, int ir, float *val, float factor, bool mirror_ghost, float *vdt2);
  void apply_src_coefz_lz(float *wf, float val, float *kz, int lz, int offz, int iz_mirror, bool mirror_ghost, float *vdt2);
  void apply_src_coefxy(float *wf, float *trace, int ir, int nt, int nw);

  void spread_kdomain(float vsurf, int trueamp, float z, int deghost);
  void spread_sdomain(float vsurf, int trueamp, float z, int deghost);

  void get_wavenumber(int nkx, float dkx, float *kx);
  float* get_ktaper(int nkx, float *kx, float k1, float k2);
  /** Sets the receiver data.
   *
   * @param traces    The data to set. The data is now owned by
   *                        the receiver object.
   */
  void setData(unique_ptr<Traces> traces);
  int size();

  /** spread the receiver data
   *
   */
  void spread();
  vector<float> spreadCoeffs(int ntaperXY, bool do_dipole); // returns [zmean, zdev]
  // void trueamplitude_retired(float);

  void putTrace(int ix, int iy, int iz, const float *buffs, int itMin, int itMax, float *vol, float scaler = 1.0f);
  void update_zrange(int do_alloc = true);
  void post_spread_correction();

protected:
  friend class ExtractReceiverData;

  void spreadReceiver(const vector3 &coord, const float *data, int itMin, int itMax, float *vol, bool do_dipole);
  void taperTrace(float *data, const float *wtaperXY, int ntaperXY);

  float* getDataForIteration(int it);

  std::string generateFileName();

  void flipCoeff(float *mycoeffz, int lz);

  shared_ptr<Grid> grid { };

  /// Sample rate, dt/(dx*dy*dz)
  float dt, dt_dxyz;
  /// Number of samples
  int nt;

  int nThreads;

  bool bufferToDisk;
  std::fstream vol4dFile;
  std::string bufferFileName;
  float *buffer;
  int sfBoundType;
  bool mirror_ghost;
  bool did_dipole { };
  bool transpose_fk { };
  float zsurf = 0;

  //
  int id; //0: source, 1: receiver
  PROP::Operation oper;
  int dim_bits;
  int ix0;
  int iy0;
  int iz0;
  int ix1;
  int iy1;
  int iz1;
  int nsz;
  int nz_grid; // for forward modeling, where spread offset is relative to the grid
  int nx, ny, nz; // grid->nx, ny, nz
  size_t size4d;   // == nsx*nsy*nsz*nt
  float scaleFactor;
  float *vol4d;
  Model *model;
  int spreadSize, spreadSizeZ;
  bool myInterp, use_src4d;
public:
  vector<vector<float> > kernels_x, kernels_y, kernels_z;
  vector<int> offxs, offys, offzs;
  vector<int> lxs, lys, lzs;
  vector<float> pow_phase;
  unique_ptr<Traces> traces;
  vector<float> spread_data;
  vector<vector<float>> buf_omp;
  int domain; // kdomain or sdomain
  int trueamp;
  int deghost;
  float flcut = 0, flpass = 0;
  float fhpass = 0, fhcut = 0;
  float khpass = 0.5, khcut = 0.5;
  RecordLoader *recordLoader;
};

#endif /* RECEIVER_H_ */

