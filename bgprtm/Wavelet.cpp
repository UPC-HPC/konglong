#include "Wavelet.h"
#include "GetPar.h"
#include "Grid.h"
#include "RecordLoader.h"
#include "libSWIO/RecordIO.hpp"
#include "Traces.h"
#include "TraceHdrDefine.h"
#include "libCommon/Options.h"
#include "libCommon/WaveletLib.h"
#include "libCommon/CommonMath.h"
#include "Source.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "MpiPrint.h"
using MpiPrint::print1m;
using MpiPrint::printm;

Wavelet::Wavelet(int waveletType, float f1, float f2, float f3, float fmax, int nt0, float dt, float extra_srt, float t_delay, int sinc_avg,
    int phase_type) : waveletType(waveletType), phase_type(phase_type), f1(f1), f2(f2), f3(f3), fmax(fmax), nt(nt0), dt(dt), extra_srt_over_it0(
    extra_srt), t_delay(t_delay), sinc_avg(sinc_avg) {

  init();
}

// t_delay: used only when negative, extra source rising time so later operations can shift wavelet without causing problems
// does not actually apply the shift
Wavelet::Wavelet(int waveletType, float fmax, int nt0, float dt, float extra_srt_over_it0, float t_delay, int sinc_avg, int phase_type) : waveletType(
    waveletType), fmax(fmax), nt(nt0), dt(dt), extra_srt_over_it0(extra_srt_over_it0), t_delay(t_delay), sinc_avg(sinc_avg), phase_type(
    phase_type) {
  init();
  scan();
}

void Wavelet::scan() {
  print1m("Wavelet::maxamp=%g\n", libCommon::maxfabs(mysource, nt));
}

void Wavelet::init() {
  it0 = 0;
  t0 = 0;
  mysource = NULL;

  switch(waveletType) {
  case RICKER:
    ricker();
    break;
  case SPIKY:
    extra_srt_over_it0 *= 2;
    spiky();
    break;
  case DELTA:
    delta();
    break;
  case ORMSBY:
    ormsby();
    break;
  case USERWAVELET:
    userwavelet();
    break;
  default:
    print1m("Wavelet: Unknow waveltType, waveletype=%d !\n", waveletType);
    exit(-1);
    break;
  }

  // In case of the internal wavelet generation, wavelet will be normalized subject to the grid
  // a constant rescale is introduced to output both modeled data and image in a more conventional
  // scale. The constant rescale is disabled for waveletprep call to generate wavelet outside or
  // use userwavelet option in swpro
  if(waveletType != USERWAVELET) { // removed && (sinc_avg) in the condition by wolf
    for(int i = 0; i < nt; i++) {
      mysource[i] *= 1E5; // do not introduce dt inside wavelet generation ...
    }
  }

  if(extra_srt_over_it0 || t_delay < 0) {
    int extra = (int)ceilf(it0 * extra_srt_over_it0 + max(0.0f, -t_delay) / dt);
    assertion(extra >= 0, "extra=%d, extra_srt_over_it0=%f, it0=%d, t_delay=%f, dt=%f", extra, extra_srt_over_it0, it0, t_delay, dt);
    float *mysource2 = new float[nt + extra]();
    for(int i = 0; i < nt; i++)
      mysource2[extra + i] = mysource[i];
    delete[] mysource;
    mysource = mysource2;
    t0 += extra * dt;
    nt += extra, it0 += extra;
  }
}

Wavelet::~Wavelet() {
  if(mysource) {
    delete[] mysource;
    mysource = 0;
  }
}

void Wavelet::ricker() {
  float fpeak = fmax / 3.0f;
  m = libCommon::ricker_design(fpeak, dt);

  it0 = m / 2;
  nt += it0;
  nt = max(nt, m); // make sure nt >= m
  t0 = it0 * dt;

  delete[] mysource, mysource = new float[nt]();
  // memset(mysource, 0.0, sizeof(nt));

  if(phase_type == 0) {
    libCommon::ricker_assign(fpeak, mysource, dt, m);
  } else {
    float *tmpWavelet = (float*)calloc(m, sizeof(float));
    libCommon::ricker_assign(fpeak, tmpWavelet, dt, m);
    libCommon::minimum_phasing(tmpWavelet, m);
    memcpy(&mysource[0], tmpWavelet, m * sizeof(float));
    free(tmpWavelet);
    it0 = 0;
    t0 = 0.0;
  }

  if(sinc_avg) {
    float *wrk = new float[m]();
    libCommon::sincavg(mysource, m, wrk);
    delete[] wrk;
  }
}

void Wavelet::delta() { // for simple QC, do not use it in real application
  it0 = fmax * dt * 100;
  it0 = global_pars["wavelet_it0"].as<int>(it0);
  // it0 = 0;    // star: in waveletprep it is 0;
  nt += it0;
  t0 = it0 * dt;

  delete[] mysource, mysource = new float[nt]();
  //memset(mysource, 0.0, sizeof(nt));
  mysource[it0] = 1;
}

void Wavelet::spiky() {

  float fhigh = global_pars["fhigh"].as<float>(fmax * 0.75f);
  float frdb = global_pars["frdb"].as<float>(70.0f);

  m = libCommon::spiky_design(fmax, fhigh, frdb, dt);
  it0 = m / 2;
  nt += it0;
  nt = max(nt, m); // make sure nt >= m
  t0 = it0 * dt;
  delete[] mysource, mysource = new float[nt]();
  //memset(mysource, 0, sizeof(float)*nt);

  if(phase_type == 0) {
    libCommon::spiky_assign(fmax, fhigh, frdb, mysource, dt, m);
  } else {
    float *tmpWavelet = (float*)calloc(m, sizeof(float));
    libCommon::spiky_assign(fmax, fhigh, frdb, tmpWavelet, dt, m);
    libCommon::minimum_phasing(tmpWavelet, m);
    memcpy(&mysource[0], tmpWavelet, m * sizeof(float));
    free(tmpWavelet);
    it0 = 0;
    t0 = 0.0;
  }

  if(sinc_avg) {
    float *wrk = new float[m]();
    libCommon::sincavg(mysource, m, wrk);
    delete[] wrk;
  }

}

void Wavelet::ormsby() {
  float f11 = (f1 < 0) ? global_pars["f1"].as<float>(0.0f) : f1;
  float f21 = (f2 < 0) ? global_pars["f2"].as<float>(0.0f) : f2;
  float f31 = (f3 < 0) ? global_pars["f3"].as<float>(0.8f * fmax) : f3;
  float f41 = (f4 < 0) ? global_pars["f4"].as<float>(fmax) : f4;
  float frdb = global_pars["frdb"].as<float>(70.0f);

  m = libCommon::ormsby_design(f11, f21, f31, f41, frdb, dt);
  it0 = m / 2;
  nt += it0;
  nt = max(nt, m); // make sure nt >= m
  t0 = it0 * dt;
  delete[] mysource, mysource = new float[nt]();
  // memset(mysource, 0, sizeof(float)*nt);

  if(phase_type == 0) {
    libCommon::ormsby_assign(f11, f21, f31, f41, frdb, mysource, dt, m);
  } else {
    float *tmpWavelet = (float*)calloc(m, sizeof(float));
    libCommon::ormsby_assign(f11, f21, f31, f41, frdb, tmpWavelet, dt, m);
    libCommon::minimum_phasing(tmpWavelet, m);
    memcpy(&mysource[0], tmpWavelet, m * sizeof(float));
    free(tmpWavelet);
    it0 = 0;
    t0 = 0.0;
  }

  if(sinc_avg) {
    float *wrk = new float[m]();
    libCommon::sincavg(mysource, m, wrk);
    delete[] wrk;
  }
}

void Wavelet::userwavelet() {
  int gtype = RECTANGLE;
  int nxu = 1;
  int nyu = 1;
  int nzu = 1;
  int gdx = 1;
  int gdy = 1;
  int gdz = 1;
  float zmin = 0;
  float zmax = global_pars["zMax"].as<float>() - global_pars["zMin"].as<float>(0.0f);

  int local_sid;
  int local_rid;
  float local_sx;
  float local_sy;
  float local_sz;
  float local_rx;
  float local_ry;
  float local_rz;
  float local_dt;
  int local_nSamples;
  int local_nTraces;
  int local_changeEndian1 = 2;
  bool local_changeEndian = true; // always output big Endian as system is little Endian
  int loc = 109;   // delay time position
  int valType = 1;     // float type

  string fileName = expEnvVars(global_pars["waveletInputFile"].as<string>());
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(fileName);
  if(global_pars["SegyFormat"]) reader->setTraceHdrEntry(TraceHdrDefine::readSegyFormat(global_pars["SegyFormat"]));

  vector<libCommon::Trace*> vecTraces;
  int nframes = reader->getNFrames();
  bool success = false;
  if(nframes > 1) {
    assertion(bool(global_pars["_posLogical"]) && global_pars["_posLogical"].IsSequence(),
              "Param '_posLogical' must be programmed to array of ints for nframes(%d)>1", nframes);
    vector<int> posLogical = global_pars["_posLogical"].as<vector<int>>();
    printm("USERWAVELET (nframes=%d): reading frame at logical position: [%s]\n", nframes, COptions::ints2str(posLogical).c_str());
    success = reader->readFrame(posLogical, vecTraces);
  } else success = reader->readFrame(0, vecTraces);

  libCommon::Trace *trace0 = vecTraces[0];
  t0 = trace0->getWaveletT0();
  int ntIn = reader->getNSample();
  float dtIn = reader->getDSample();

  delete reader;

  it0 = int(ceil(t0 / dt) + 1); // this normally is not a integer number. It's the tricky part on resample+shift

  nt = nt + it0;

  cout << "userwavelet" << endl;
  cout << "zmin = " << zmin << endl;
  cout << "zmax = " << zmax << endl;
  cout << "t0    = " << t0 << endl;
  cout << "it0   = " << it0 << endl;
  cout << "nt    = " << nt << endl;

  float tmpMinFreq = 0.0;
  int tmpDim = 1; // no meaning dummy dimension used to accord to the format.
  Grid grid(gtype, nxu, nyu, nzu, gdx, gdy, gdz, zmin, zmax, 1);
  RecordLoader recordLoader(&grid, dt, nt, fmax, tmpMinFreq, tmpDim, PROP::FORWARD, -t0, -it0 * dt, true);

  unique_ptr<Traces> traces = make_unique<Traces>(ntIn, trace0->getData(), 1, dtIn);
  traces->addReceiver(vector3 { }); // add one trace with dummy receiver coordinates
  traces = move(recordLoader.filterTraces(move(traces), dtIn, trace0->getShotID(), false)[0]);

  libCommon::Utl::deletePtrVect(vecTraces);
  delete[] mysource, mysource = new float[nt]();

  t0 = it0 * dt;

  for(int i = 0; i < nt; i++) {
    mysource[i] = traces->data[0][i];// * 1E8 * dt;
  }
}

