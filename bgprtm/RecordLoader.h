/*
 * RecordLoader.h
 *
 */

#ifndef RECORDLOADER_H_
#define RECORDLOADER_H_

#include <vector>
#include "libCommon/Utl.hpp"
#include "libCommon/Horizon.hpp"
#include "libTaup/taupProcess.hpp"
#include "ModelPrepare.h"
#include "Source.h"

// forward declarations
class Grid;
class vector3;
class Interpolant;
class Traces;
class PhaseCorrection;

/** load a shot data from a Record file into memory
 *
 */
class RecordLoader {
public:
  /** ctor
   *  @param dt      sample rate (new)
   *  @param nt      # of samples (new)
   *  @param maxFreq maximum frequency
   *  @param minFreq minimum frequency
   */
  RecordLoader(Grid *grid, float dt, int nt, float maxFreq, float minFreq, int dim, PROP::Direction direction, float t0In = 0.0,
      float t0Out = 0.0, bool userwavelet = false);

  /** dtor
   *
   */
  virtual ~RecordLoader();

  /** load Record data into memory
   *  @param fileName      Record file name
   *  @param reverseTraces =true for reverse-time
   *
   *  @return The receiver data (only live traces inside the domain).
   */
  unique_ptr<Traces> readRecord(const char *fileName, bool reverseTraces = true, bool header_only = false, bool filter = true);
  vector<unique_ptr<Traces>> filterTraces(unique_ptr<Traces> tracesIn, float dtIn, int sourceID, bool reverseTraces, bool keepTracesIn =
      false); // return {new_traces, tracesIn}
  static void writeRecord(const char *filename, float *dat, const Traces *traces, int nr, int nt, float dt);

protected:
  /**
   * Build the taper
   */
  void buildTaper(float dt1, int nt1);

  /**
   * Resample the trace
   */
  void resample(float *inTrc, float dt1, int nt1, float *outTrc, float dt2, int nt2, float rotate_deg = 90, float t01 = 0.0,
      float t02 = 0.0);

  /**
   * Check if file is SU (check is only based on file extension).
   */
  static bool isSUFile(const char *fileName);

  /*
   * get the range of models
   */
  void getModelRange(int &ixMin, int &ixMax, int &iyMin, int &iyMax);
  static void getModelRange(Grid *grid, int &ixMin, int &ixMax, int &iyMin, int &iyMax);

  // deghost, interpolation and anti-aliasing
  void pre_processing(vector<float> &x, vector<float> &y, vector<float> &z, vector<float> &data, int ntIn, float dtIn);

public:
  float hpFreq = 0;

protected:
  Grid *grid;

  int nt;
  float dt;

  float maxFreq;
  float minFreq;
  int dim;

  float t0In;
  float t0Out;
  bool userwavelet;
  PROP::Direction prop_dir;

  int nfft, nw; // note nw here is less than the fftw's nw, 0/nyquist is combined

  float sourceX, sourceY, sourceZ;
  float elevation_shift;

  int nThreads;

  float *wtaper;
  Interpolant *intp;
  PhaseCorrection *phaseCorrect;
};

#endif /*RECORDLOADER_H_ */

