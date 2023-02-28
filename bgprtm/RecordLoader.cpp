#include "RecordLoader.h"
#include "ModelLoader.h"
#include "AssertAlways.h"
#include "GetPar.h"
#include "Grid.h"
#include "libFFTV/fft1dfunc.h"
#include "libFFTV/fftapp.h"
#include "fdm.hpp"
#include "Vector3.h"
#include "Interpolant.h"
#include "Geometry.h"
#include "Source.h"
#include "PhaseCorrection.h"
#include "libCommon/WaveletLib.h"
#include "libCommon/CommonMath.h"
#include "libCommon/padfft.h"
#include "libCommon/fftSize.hpp"

#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;
//
#include <string.h>
#include <math.h>
#include <omp.h>
#include <cfloat>
#include "Traces.h"
#include "libSWIO/RecordIO.hpp"

#include "MpiPrint.h"
using MpiPrint::print1m;

using namespace std;
//
RecordLoader::RecordLoader(Grid *grid, float dt, int nt, float maxFreq, float minFreq, int dim, PROP::Direction direction, float t0In,
    float t0Out, bool userwavelet) :
    grid(grid), nt(nt), dt(dt), maxFreq(maxFreq), minFreq(minFreq), dim(dim), prop_dir(direction), t0In(t0In), t0Out(t0Out), userwavelet(
        userwavelet) {
  nfft = nw = 0;
  sourceX = sourceY = sourceZ = 0;

  wtaper = NULL;

  nThreads = omp_get_max_threads();
  nThreads = global_pars["nThreads"].as<int>(nThreads);  // If this isn't specified, use all threads
  if(nThreads < 1) nThreads = omp_get_max_threads();
  intp = NULL;
  phaseCorrect = NULL;

  elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);

  print1m("Receiver: the dt is  %f\n", dt);
}

RecordLoader::~RecordLoader() {
  //

  if(wtaper) delete[] wtaper;

  if(intp) delete intp;

  if(phaseCorrect) delete phaseCorrect;
}

// also include the w_pow due to 1D-3D, Laplacian or not
void RecordLoader::buildTaper(float dt1, int nt1) {

  nfft = libCommon::padfft(nt1);
  nw = nfft / 2; // fftv's special

  float dw1 = (2.0f * M_PI) / (dt1 * nfft);

  float f1 = global_pars["recLowCut"].as<float>(0.0f);
  float f2, f3, f4;
  if(userwavelet) {
    f2 = 0.0f;
    f3 = maxFreq;
    f4 = maxFreq;
  } else {
    f2 = global_pars["recLowPass"].as<float>(3.0f);
    f3 = global_pars["recHighpass"].as<float>(0.8f * maxFreq);
    f4 = global_pars["recHighCut"].as<float>(1.0f * maxFreq);
  }
  hpFreq = f3;

  int iw1 = (int)(f1 * 2.0f * M_PI / dw1);

  int iw4 = (int)(f4 * 2.0f * M_PI / dw1);
  iw4 = MIN(iw4, nw);

  int iw2 = (int)(f2 * 2.0f * M_PI / dw1);
  assertion(iw1 <= nw && iw2 <= nw, "maxFreq=%f, nw1/2=%d, iw1=%d, iw2=%d(%f*2pi/%f), dt1=%f\n", maxFreq, nw, iw1, iw2, f2, dw1, dt1);
  iw2 = MAX(iw2, iw1);

  int iw3 = (int)(f3 * 2.0f * M_PI / dw1);
  iw3 = MAX(iw3, iw2);
  iw3 = MIN(iw3, iw4);

  print1m("\n Build Taper for Receivers: \n");
  print1m(" Low cut   : %.2f Hz \n", (float)iw1 * dw1 / (2.0f * M_PI));
  print1m(" Low pass  : %.2f Hz \n", (float)iw2 * dw1 / (2.0f * M_PI));
  print1m(" High pass : %.2f Hz \n", (float)iw3 * dw1 / (2.0f * M_PI));
  print1m(" High cut  : %.2f Hz \n", (float)iw4 * dw1 / (2.0f * M_PI));
  // print1m("maxFreq=%f, nw1/2+1=%d, iw1=%d, iw2=%d, iw3=%d, iw4=%d\n", maxFreq, nw1 / 2 + 1, iw1, iw2, iw3, iw4), fflush(stdout);

  //
  wtaper = new float[nw + 1];
  float dk = 2 * M_PI / (nfft * dt1);

  float w_pow = 0.0;
  if(!userwavelet && !getBool("src_correct_after_spread", false)) {
    vector<float> pow_phase = Source::getSpectrumPowPhase(prop_dir, prop_dir == PROP::FORWARD ? PROP::RTMM : PROP::RTM, dim);
    w_pow = pow_phase[0];
  }

  for(int iw = 0; iw < iw1; iw++)
    wtaper[iw] = 0.0f;

  for(int iw = iw1; iw <= iw2; iw++) {
    float alpha = (float)(iw - iw1 + 1) / (float)(iw2 - iw1 + 1) * M_PI;
    wtaper[iw] = 0.5f * (1.0f - cosf(alpha)) * (w_pow == 0 ? 1 : iw == 0 ? 0 : powf(iw * dk, w_pow));
  }

  for(int iw = iw2 + 1; iw < iw3; iw++)
    wtaper[iw] = (w_pow == 0 ? 1 : iw == 0 ? 0 : powf(iw * dk, w_pow));

  for(int iw = iw3; iw <= iw4; iw++) {
    float alpha = (float)(iw4 - iw + 1) / (float)(iw4 - iw3 + 1) * M_PI;
    wtaper[iw] = 0.5f * (1.0f - cosf(alpha)) * (w_pow == 0 ? 1 : iw == 0 ? 0 : powf(iw * dk, w_pow));
  }

  for(int iw = iw4 + 1; iw < nw + 1; iw++)
    wtaper[iw] = 0.0f;
}

// rotate deg + resample (old rotate90 = -i)
void RecordLoader::resample(float *inTrc, float dt1, int nt1, float *outTrc, float dt2, int nt2, float rotate_deg, float t01, float t02) {
  vector<float> buff(nw * 4, 0);
  complex<float> *cbuf = (complex<float>*)&buff[0];
  memcpy(&buff[0], inTrc, nt1 * sizeof(float));

  if(!userwavelet) {
    float scalar = 1.0f / nfft;

    libfftv::fft_r2c_fdf(&buff[0], nfft);

    buff[0] *= wtaper[0] * scalar;  //0
    buff[1] *= wtaper[nw] * scalar; //NQuist

    float phase = rotate_deg / 180 * M_PI;
    complex<float> cphase = scalar * std::exp(complex<float> { 0, -phase }); // old rotate90=1 is -i
    for(int iw = 1; iw < nw; iw++)
      cbuf[iw] *= wtaper[iw] * cphase;

    libfftv::fft_c2r_bdf(&buff[0], nfft);
  }

  intp->interpolate1D(&buff[0], nt1, t01, dt1, outTrc, nt2, t02, dt2);
}
bool RecordLoader::isSUFile(const char *fileName) {
  return (strlen(fileName) > 3 && strncmp(&fileName[strlen(fileName) - 3], ".su", 3) == 0);
}

/*
 * output SEGY-1, headers is native: __BYTE_ORDER, data: __BIG_ENDIAN
 * dat will be changed in general, from __LITTLE_ENDIAN to __BIG_ENDIAN
 */
void RecordLoader::writeRecord(const char *filename, float *dat, const Traces *inTraces, int nr, int nt, float dt) {
  vector<libCommon::Grid1D> grids(2);
  grids[0] = libCommon::Grid1D(nt, 0, dt);
  grids[1] = libCommon::Grid1D(nr, 0, 12.5);
  libSeismicFileIO::RecordWriter writer(string(filename), &grids);
  print1m("Writing file: %s\n", filename);

  float sx = global_pars["_sourceX"].as<float>(0);
  float sy = global_pars["_sourceY"].as<float>(0);
  float sz = global_pars["_sourceZ"].as<float>(0);
  int sid = global_pars["_sourceID"].as<int>(1);
  int ridBegin = global_pars["receiverID"].as<int>(1);

  vector<libCommon::Trace*> traces(nr, 0);
  for(int ir = 0; ir < nr; ir++) {
    traces[ir] = new libCommon::Trace(grids[0], new float[nt]());
    traces[ir]->setShotID(sid);
    traces[ir]->setRecvID(ir + ridBegin);
    memcpy((void*)traces[ir]->getData(), (void*)(dat + ir * nt), nt * sizeof(float));
    traces[ir]->setShotLoc(libCommon::Point(sx, sy, sz));
    traces[ir]->setRecvLoc(libCommon::Point(inTraces->coord[ir].x, inTraces->coord[ir].y, inTraces->coord[ir].z));
  }
  writer.writeNextFrame(traces);
  libCommon::Utl::deletePtrVect<libCommon::Trace>(traces);
}

void RecordLoader::getModelRange(int &ixMin, int &ixMax, int &iyMin, int &iyMax) {
  getModelRange(grid, ixMin, ixMax, iyMin, iyMax);
}

void RecordLoader::getModelRange(Grid *grid, int &ixMin, int &ixMax, int &iyMin, int &iyMax) {

  string geomFile = expEnvVars(global_pars["local"]["geometryFile"].as<string>(GEOM_LOCAL_DEFAULT));
  Geometry *geom = new Geometry();
  geom->read(geomFile);
  GeomHeader *gHeader = geom->getHeader();

  int ixl = grid->getIDx(gHeader->x0);
  int iyl = grid->getIDy(gHeader->y0);

  int ixr = grid->getIDx(gHeader->x0 + (gHeader->nx - 1) * gHeader->dx);
  int iyr = grid->getIDy(gHeader->y0 + (gHeader->ny - 1) * gHeader->dy);

  delete geom;

  print1m("Model range: ixMin=%4d, iyMin=%4d ixMax=%4d, iyMax=%4d\n", ixl, iyl, ixr, iyr);

  ixMin = MAX(ixMin, ixl);
  iyMin = MAX(iyMin, iyl);

  ixMax = MIN(ixMax, ixr);
  iyMax = MIN(iyMax, iyr);

  print1m("Final range: ixMin=%4d, iyMin=%4d ixMax=%4d, iyMax=%4d\n", ixMin, iyMin, ixMax, iyMax);

}

// read the traces in the record file, process when filter is true
unique_ptr<Traces> RecordLoader::readRecord(const char *fileName, bool reverseTraces, bool hdr_only, bool filter) {
  // open file
  libSeismicFileIO::RecordReader reader(fileName);
  reader.setCoordinateScale(global_pars["data_coord_scale"].as<float>(1.0));

  //
  float dtIn = reader.getDSample();
  int ntIn = reader.getNSample();

  int doSequentialShotNum = global_pars["doSequentialShotNum"].as<int>(1);
  vector<libCommon::Trace*> shotGather;
  reader.readFrame(doSequentialShotNum - 1, shotGather);  // default from 0
  int nTraces = shotGather.size();
  int sourceID = shotGather[0]->getShotID();

  // start handling the traces
  float coordScalar = global_pars["segyCoordScalar"].as<float>(1.0f); // in case for segy file
  unique_ptr<Traces> traces = make_unique<Traces>(hdr_only ? 0 : ntIn, nullptr, 0, dt);
  for(int itrace = 0; itrace < nTraces; itrace++) {
    libCommon::Trace &trace = *shotGather[itrace];
    libCommon::Point shotLoc = trace.getShotLoc();
    libCommon::Point recvLoc = trace.getRecvLoc();
    float selev = shotLoc.z - elevation_shift;  //source depth - elevation
    float relev = recvLoc.z - elevation_shift;  //receiver depth - elevation
    float sx = shotLoc.x * coordScalar;
    float sy = shotLoc.y * coordScalar;
    float rx = recvLoc.x * coordScalar;
    float ry = recvLoc.y * coordScalar;

    if(itrace == 0) traces->coord_src = vector3(sx, sy, selev);
    traces->addReceiver(vector3(rx, ry, relev));
    if(!hdr_only) memcpy(traces->data[itrace], trace.getData(), sizeof(float) * ntIn);
  }

  libCommon::Utl::deletePtrVect(shotGather);  // release memory

  if(filter) return move(filterTraces(move(traces), dtIn, sourceID, reverseTraces)[0]);
  return traces;
}

// return {new_traces, tracesIn}
vector<unique_ptr<Traces>> RecordLoader::filterTraces(unique_ptr<Traces> tracesIn, float dtIn, int sourceID, bool reverseTraces,
    bool keepTracesIn) {
  bool hdr_only = tracesIn->nt == 0;
  //
  float maxOffset = global_pars["maxOffset"].as<float>(-1.0f);

  int ixMin = -1;
  int ixMax = grid->nx - 1;
  int iyMin = -1;
  int iyMax = grid->ny - 1;

  //skip the traces outside of model range
  if(getBool(global_pars["skipTraceOutsideOfModel"], false)) {
    getModelRange(ixMin, ixMax, iyMin, iyMax);
  }

  int ntIn = tracesIn->nt;

  // build the taper and table for resampling
  if(!hdr_only) {
    if(!wtaper) buildTaper(dtIn, ntIn);
    if(!intp) {
      intp = new Interpolant(LANCZOS, userwavelet ? 10 : 5, nThreads, true, userwavelet);
    }
  }

  int nTraces = tracesIn->getNReceivers();

  // start handling the traces
  int rampLength = 0;
  vector<float> inTraces;
  if(!hdr_only) {
    inTraces.resize(nTraces * ntIn, 1.e+30f);

    // ramp the very end of the traces being fed back into the wavefield
    if(!userwavelet) {
      rampLength = nt / 80.0f + 15.0f / maxFreq / dt + 0.5f;
      if(rampLength > nt / 10) rampLength = nt / 10;
      if(global_pars["receiverRampTime"]) rampLength = (int)(global_pars["receiverRampTime"].as<float>() / dt);
      rampLength = global_pars["receiverRampLength"].as<int>(rampLength);
      print1m("Trace ramp length: %g out of %g trace length.\n", rampLength * dt, nt * dt);
    }
    //
    print1m("Allocating %.1f GB for receiver data (nt=%d x nr=%d x 4) to be back propagated.   Not buffering this data to disk! \n",
            nt * (long)nTraces * 4 * 1.e-9f, nt, nTraces);
  }

  long offsetZeroCount = 0;
  long offsetDampCount = 0;

  float *wrk = hdr_only ? NULL : new float[nt];

  float rotate_deg = 0;
  if(!userwavelet && !getBool("src_correct_after_spread", false)) {
    vector<float> pow_phase = Source::getSpectrumPowPhase(prop_dir, prop_dir == PROP::FORWARD ? PROP::RTMM : PROP::RTM, dim);
    print1m("RecordLoader %s: pow=%f, phase=%f\n", prop_dir == PROP::FORWARD ? "fwd(RTMM)" : "bwd", pow_phase[0], pow_phase[1]);
    rotate_deg = pow_phase[1];
  }

  int normalize_per_trace = global_pars["normalize_per_trace"].as<int>(0);
  int time_offset_gain = (bool)global_pars["vel_time_offset_gain"];
  int gain_t = global_pars["gain_t"].as<int>(time_offset_gain ? 0 : 1); // default time_offset_gain=0, gain_t=1 (for FFT et al only)
  float vel_time_offset_gain = global_pars["vel_time_offset_gain"].as<float>(1500);
  int phaseCorrect1st = global_pars["phaseCorrect1st"].as<int>(0); // Receiver: do PhaseCorrection before or after resampling
  if(!phaseCorrect) phaseCorrect = new PhaseCorrection(phaseCorrect1st ? ntIn : nt, phaseCorrect1st ? dtIn : dt, dt, 0, maxFreq,
                                                       PhaseCorrection::FWD);

  if(userwavelet) {
    rotate_deg = 0;
    normalize_per_trace = global_pars["normalize_per_trace"].as<int>(0);
    gain_t = global_pars["gain_t"].as<int>(0);
  }

  int scaleFarOffset = global_pars["scaleFarOffset"].as<int>(1);
  int nbadTraces = 0;
  vector<float> x;
  vector<float> y;
  vector<float> z;
  int nLiveTraces = 0;
  unique_ptr<Traces> traces = make_unique<Traces>(hdr_only ? 0 : nt, nullptr, 0, dt); // nt=0: no data
  // print1m("traces=%p, dt=%f\n", traces, traces->dt);
  auto coord_src = tracesIn->coord_src;
  float selev = coord_src.z;
  float sx = coord_src.x;
  float sy = coord_src.y;
  for(int itrace = 0; itrace < nTraces; itrace++) {
    auto coord = tracesIn->coord[itrace];
    float relev = coord.z;  //receiver depth - elevation
    float rx = coord.x;
    float ry = coord.y;

    //override by user input
    relev = global_pars["receiverZ0"].as<float>(relev);
    int spreadSize = global_pars["receiverSpreadSize"].as<int>(); // should already be set at this stage

    if(!hdr_only) {
      // read trace
      float *trc = &inTraces[(size_t)itrace * ntIn];
      memcpy(trc, tracesIn->data[itrace], ntIn * sizeof(float));
      if(normalize_per_trace) {
        float absmax = 0;
        for(int it = 0; it < ntIn; it++)
          absmax = max(absmax, fabsf(trc[it]));
        if(absmax > 10 * FLT_MIN) for(int it = 0; it < ntIn; it++)
          trc[it] /= absmax;
      }

      float offset = 0;
      if(time_offset_gain || scaleFarOffset) offset = sqrtf((sx - rx) * (sx - rx) + (sy - ry) * (sy - ry));
      if(time_offset_gain) {
        for(int it = 0; it < ntIn; it++)
          trc[it] *= hypotf(it * dtIn, 2 * offset / vel_time_offset_gain);
      }
      //scale the far offset
      if(scaleFarOffset) {
        if(maxOffset > 0.0f && offset > 0.5f * maxOffset) {
          float scale = 1.0;

          if(offset >= maxOffset) {
            scale = 0.0f;
            offsetZeroCount++;
          } else if(offset > maxOffset * 0.5f) {
            scale = 0.08f + 0.92f * cosf((offset - 0.5 * maxOffset) / (0.5 * maxOffset) * M_PI * 0.5);
            offsetDampCount++;
          }

          for(int it = 0; it < ntIn; it++)
            trc[it] *= scale;
        }
      }

      //taper the end of trace
      //   if(!userwavelet) for(int it = 0; it < ntIn; it++) {
      //     float scale = 1.0f;
      //     if(it > ntIn * 0.8f) {
      //       scale = 0.08f + 0.92f * cosf((it - 0.8f * ntIn) / (0.8f * ntIn) * M_PI * 0.5);
      //     }
      //     trc[it] *= scale;
      //   }
    }

    // determine grid coordinates for this trace
    float ixr = grid->getIDxf(rx, relev);
    float iyr = grid->getIDyf(ry, relev);
    int izr = floorf(grid->getIDzf(relev)); // need to be consistent with Receiver::update_zrange()
    if(!userwavelet) {
      if(izr - spreadSize + 1 < 0 || izr + spreadSize >= grid->nz || ixr < ixMin || ixr > ixMax
          || (grid->ny > 1 && (iyr < iyMin || iyr > iyMax))) {
        nbadTraces++;
        continue;
      }
    }
    x.push_back(rx);
    y.push_back(ry);
    z.push_back(relev);
    traces->addReceiver(vector3(x[nLiveTraces], y[nLiveTraces], z[nLiveTraces]));
    nLiveTraces++;

    if(itrace < 10 || itrace > nTraces - 8) {
      print1m("  Receiver %d  values read are x=%g, y=%g, z=%g.\n", itrace + 1, rx, ry, relev);
    }
  }

  print1m(" input: maxFreq = %f, ntIn=%d, tIn=%g, output: nt=%d t=%g \n", maxFreq, ntIn, (ntIn - 1) * dtIn, nt, (nt - 1) * dt);

  // do deghost, interpolation and anti-aliasing here
  if(!hdr_only) {
    if(global_pars["recv_preprocess"]) this->pre_processing(x, y, z, inTraces, ntIn, dtIn);
  }
  nTraces = x.size();   // interpolation will change the number of traces, also update to make it nLiveTraces in other cases

  if(!hdr_only) {
    if(global_pars["qc_receiver_group"]) {  // first group of raw data (after offset damp)
      string qcfile = expEnvVars(global_pars["qc_receiver_group"].as<string>());
      jseisUtil::save_zxy(qcfile.c_str(), &inTraces[0], ntIn, nTraces, 1, dtIn * 1000, 1, 1);
      if(global_pars["qc_receiver_only"].as<int>(0)) exit(-1);  // QC only, end it sooner
    }

    int sinc_avg = (global_pars["src_sinc_avg"].as<int>(0));

    int nt2 = nt + 32;
    float invRampLength = 1.0f / (rampLength + 1.0f);
    // print1m("Resample: dtIn=%f, ntIn=%d, dt=%f, nt=%d, rotate90=%d\n", dtIn, ntIn, dt, nt, rotate90), fflush(stdout);
#pragma omp parallel for num_threads(nThreads)
    for(int itrace = 0; itrace < nTraces; itrace++) {
      float *inTrc = &inTraces[itrace * ntIn];
      vector<float> outTrc(nt2, 0);
      if(gain_t) for(int it = 0; it < ntIn; it++)
        inTrc[it] *= it * dtIn;
      if((phaseCorrect1st) && (!userwavelet)) phaseCorrect->applyForward(inTrc, 1);  // source wavelet apply at Source.cpp instead

      ////////////////  Resample and optional smooth ///////////////////////////
      if(itrace == 0 && global_pars["qc_rec0_resample"]) jseisUtil::save_zxy(
          expEnvVars(global_pars["qc_rec0_resample"].as<string>()).c_str(), inTrc, ntIn, 1, 1, dtIn * 1000, 1, 1, 0, 0, 0, 1, 1, 1, 1,
          jsIO::DataFormat::FLOAT, false, -t0In);
      resample(inTrc, dtIn, ntIn, &outTrc[0], dt, nt, rotate_deg, t0In, t0Out);
      // print1m("ir=%d, maxvalI=%f,maxvalO=%f\n", itrace, libCommon::maxf(inTrc, ntIn), libCommon::maxf(&outTrc[0], nt2)), fflush(stdout);
      if(sinc_avg) libCommon::sincavg(&outTrc[0], nt, wrk);
      if(itrace == 0 && global_pars["qc_rec1_resample"]) jseisUtil::save_zxy(
          expEnvVars(global_pars["qc_rec1_resample"].as<string>()).c_str(), &outTrc[0], nt, 1, 1, dt * 1000, 1, 1, 0, 0, 0, 1, 1, 1, 1,
          jsIO::DataFormat::FLOAT, false, -t0Out);

      if((!phaseCorrect1st) && (!userwavelet)) phaseCorrect->applyForward(&outTrc[0], 1);  // source wavelet apply at Source.cpp instead
      if(gain_t == 1) {  // for gain_t == 2, data is permanently gained
        outTrc[0] = 0;
        for(int it = 1; it < nt; it++)
          outTrc[it] /= it * dt;
      }

      float maxval = 0;
      for(int it = 0; it < nt; it++) {
        float w = 1;
        if(!userwavelet) {
          w = 1.0f / (nt * dt * 100.0f);
          if(it < rampLength) w *= (it + 1.0f) * invRampLength;
        }
        float value = (reverseTraces ? outTrc[nt - 1 - it] : outTrc[it]) * w;
        maxval = max(maxval, fabsf(value));
        traces->data[itrace][it] = value;
      }
      assertion(maxval < 1.e+25f, "ShotID=%d, val=%g\n", sourceID, maxval);
      // print1m("ir=%d, maxval=%f\n", itrace, maxval), fflush(stdout);
    }
  }
  delete[] wrk;
  if(!keepTracesIn) tracesIn.reset();

  if(nbadTraces != 0) {
    printf("Number of receivers outside of the compute region: %d out of %d. \n", nbadTraces, nbadTraces + nTraces);
  }

  if(maxOffset > 0.0f) {
    printf("Because of offset limit, discarded %ld out of %d traces, damped %ld traces. \n", offsetZeroCount, nTraces, offsetDampCount);
  }

  if(nTraces == 0) {
    printf("WARNING: total number of valid receivers = 0!\n");
    traces.reset();
  }

  vector<unique_ptr<Traces>> vec_traces;
  vec_traces.push_back(move(traces));
  vec_traces.push_back(move(tracesIn));
  return vec_traces;
}

void RecordLoader::pre_processing(vector<float> &x, vector<float> &y, vector<float> &z, vector<float> &data, int ntIn, float dtIn) {
  YAML::Node node = global_pars["recv_preprocess"];
  float xWinLen = node["x_win"].as<float>();
  float yWinLen = node["y_win"].as<float>();
  float dx = node["dx"].as<float>(0.0);
  float dy = node["dy"].as<float>(0.0);
  float fmax = node["fmax"].as<float>();
  float vel = node["velocity"].as<float>();
  float angmax = node["angle_max"].as<float>();
  float p_overSampling_rate = node["p_overSampling_rate"].as<float>();
  float rho = node["rho"].as<float>();
  float lambda = node["lambda"].as<float>();
  int boundary = node["boundary"].as<int>();
  float pad_factor = node["pad_factor"].as<float>();
  float zr = node["zr"].as<float>();
  int apply_deghost = node["apply_deghost"].as<int>(1);
  int apply_antialiasing = node["apply_antialiasing"].as<int>(1);
  int apply_interpolation = node["apply_interpolation"].as<int>(1);
  nThreads = omp_get_num_threads();

  int nTraces = x.size();
  libCommon::Range2D range;
  for(int i = 0; i < nTraces; i++) {
    range += libCommon::Point2d(x[i], y[i]);
  }

  dx = MAX(1.0, (dx < 1 ? grid->dx : dx));
  libCommon::Grid1D xGrid((int)((grid->nx - 1) * grid->dx / dx + 1.5), grid->x0, dx);
  float x0 = this->grid->x0 + xGrid.idx_floor(range.xRange.begin) * dx;
  int nx = xGrid.idx_ceil(range.xRange.end) - xGrid.idx_floor(range.xRange.begin) + 1;

  dy = MAX(1.0, (dy < 1 ? grid->dy : dy));
  libCommon::Grid1D yGrid((int)((grid->ny - 1) * grid->dy / dy + 1.5), grid->y0, dy);
  float y0 = this->grid->y0 + yGrid.idx_floor(range.yRange.begin) * dy;
  int ny = yGrid.idx_ceil(range.yRange.end) - yGrid.idx_floor(range.yRange.begin) + 1;

  int nTraces_out = nTraces;
  vector<float> x_out, y_out;
  if(apply_interpolation == 1) {
    for(int iy = 0; iy < ny; iy++) {
      for(int ix = 0; ix < nx; ix++) {
        x_out.push_back(x0 + ix * dx);
        y_out.push_back(y0 + iy * dy);
      }
    }
    nTraces_out = nx * ny;
  } else {
    x_out = x;
    y_out = y;
  }

  vector<float> data_out((size_t)nTraces_out * (size_t)ntIn, 0);
  float t0 = 0;  // ATTENTION, here suppose the t begin with t0 = 0
  libTaup::taup_process(apply_deghost, apply_antialiasing, nTraces, t0, ntIn, dtIn, &x[0], &y[0], &data[0], xWinLen, yWinLen, x0, dx, nx,
                        y0, dy, ny, 0, fmax, libCommon::fftsize(nt * (1 + pad_factor)), vel, angmax, p_overSampling_rate, rho, lambda,
                        boundary, pad_factor, zr, nThreads, nTraces_out, &x_out[0], &y_out[0], &data_out[0]);

  if(apply_interpolation == 1) {
    // create horizon to interpolate z coordinate
    libCommon::Horizon hor;
    hor.setGrid(libCommon::Grid2D(libCommon::Grid1D(nx, x0, dx), libCommon::Grid1D(ny, y0, dy)));
    for(int i = 0; i < nTraces; i++) {
      hor.add(libCommon::Point(x[i], y[i], z[i]));
    }
    hor.fillHole();

    // create the new traces
    x = x_out;
    y = y_out;
    z.resize(nx * ny);
    for(int i = 0; i < nTraces_out; i++)
      z[i] = hor.getDepth(libCommon::Point2d(x[i], y[i]));
  }

  data = data_out;
}

