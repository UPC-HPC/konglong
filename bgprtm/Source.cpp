#include "Util.h"
#include "Interpolant.h"
#include "Source.h"
#include "Receiver.h"
#include "Propagator.h"
#include "PhaseCorrection.h"
#include "ImagingCondition.h"
#include "Wavefield.h"
#include "libFFTV/fftapp.h"
#include "libCommon/padfft.h"
#include "GetPar.h"
#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;

#include "MpiPrint.h"
using MpiPrint::print1m;

Source::Source(Grid *grid0, int nth) :
    sourceType(POINT), grid(grid0), nThreads(nth) {
  create();
}

//Source::Source(SourceType type0, Grid *grid0, int nth) :
//    sourceType(type0), grid(grid0), nThreads(nth) {
//  create();
//}

void Source::create() {
  zsurf = - global_pars["zMin"].as<float>(0.0f);
  vol4d = NULL;
  vfile = NULL;
  point = NULL;
  vol4d90 = NULL;

  sfBoundType = getSurfaceType(PROP::FORWARD);

}

Source::~Source() {
  if(vol4d) {
    delete[] vol4d;
    vol4d = NULL;
  }
  if(point) {
    delete[] point;
    point = NULL;
  }
  if(vol4d90) {
    delete[] vol4d90;
    vol4d90 = NULL;
  }
}

int Source::getWaveletType() {
  //get the wavelet type from job
  string waveletType = "ORMSBY";  //default is ORMSBY wavelet
  int wtype = ORMSBY;
  if(global_pars["waveletType"]) {
    waveletType = global_pars["waveletType"].as<string>();
    transform(waveletType.begin(), waveletType.end(), waveletType.begin(), ::toupper);

    if(waveletType.compare("RICKER") == 0) {
      wtype = RICKER;
    } else if(waveletType.compare("SPIKY") == 0) {
      wtype = SPIKY;
    } else if(waveletType.compare("ORMSBY") == 0) {
      wtype = ORMSBY;
    } else if(waveletType.compare("DELTA") == 0) {
      wtype = DELTA;
    } else if(waveletType.compare("USERWAVELET") == 0) {
      wtype = USERWAVELET;
    } else {
      print1m("Error: Unknown wavelet type! waveletType=%s \n", waveletType.c_str());
      exit(-1);
    }
  }

  print1m("waveletType=%d \n", wtype);
  return wtype;
}

int Source::getWaveletPhaseType() {
  //get the wavelet phase type from job
  string phaseTypeString = "ZERO";  //default is zero phase wavelet
  int phaseType = ZERO;
  if(global_pars["phaseType"]) {
    phaseTypeString = global_pars["phaseType"].as<string>();
    transform(phaseTypeString.begin(), phaseTypeString.end(), phaseTypeString.begin(), ::toupper);

    if(phaseTypeString.compare("ZERO") == 0) {
      phaseType = ZERO;
    } else if(phaseTypeString.compare("MINIMUM") == 0) {
      phaseType = MINIMUM;
    } else {
      print1m("Error: Unknown phase type! phaseType=%s \n", phaseTypeString.c_str());
      exit(-1);
    }
  }

  print1m("phaseType=%d \n", phaseType);
  return phaseType;
}

int Source::getSurfaceType(PROP::Direction direction) {
  int bndType = ABSORB;
  string sourceSurfaceType = global_pars["sourceSurfaceType"].as<string>("ABSORB");
  if(direction == PROP::BACKWARD) sourceSurfaceType = global_pars["receiverSurfaceType"].as<string>(sourceSurfaceType);
  transform(sourceSurfaceType.begin(), sourceSurfaceType.end(), sourceSurfaceType.begin(), ::toupper);

  if(sourceSurfaceType.compare("FREESURFACE") == 0) {
    bndType = FREESURFACE;
  } else if(sourceSurfaceType.compare("ABSORB") == 0) {
    bndType = ABSORB;
  } else if(sourceSurfaceType.compare("GHOST") == 0) {
    bndType = GHOST;
  } else if(sourceSurfaceType.compare("SOURCE_GHOST_ONLY") == 0) {
    bndType = SOURCE_GHOST;
  } else if(sourceSurfaceType.compare("RECEIVER_GHOST_ONLY") == 0) {
    bndType = RECEIVER_GHOST;
  } else {
    print1m("Error: Unknown sourceSurfaceType! sfBoundType=%s \n", sourceSurfaceType.c_str());
    exit(-1);
  }
  print1m("sourceSurfaceType = %s \n", sourceSurfaceType.c_str());
  return bndType;
}

Wavelet* Source::setupWavelet(int wtype, int phaseType, float slow0, int nt0, float dt0, float maxfreq, PROP::Operation oper, int dim_bits,
    float fhigh, float flow, float min_t_delay) {
  if(fhigh == 0) fhigh = 0.9 * maxfreq;   // just in case the user does not set the fhigh for Omsby
  nt = nt0;
  dt = dt0;
  slow = slow0;

  vector<float> pow_phase = getSpectrumPowPhase(PROP::FORWARD, oper, dim_bits);
  float w_pow = pow_phase[0], phase_deg = pow_phase[1];
  float extra_srt = (phase_deg != 0) * 1.0f; // for phase rotating, extra source-rising-time needed (ratio to it0, default 1.0)
  int sinc_avg = global_pars["src_sinc_avg"].as<int>(0); // default value changed from 1 to 0 by wolf
  Wavelet *mylet = new Wavelet(wtype, maxfreq, nt, dt, extra_srt, min_t_delay, sinc_avg, phaseType);
  // simba 2 add new option not using Wavelet.h
  nt = mylet->nt;
  it0 = mylet->it0;
  t0 = mylet->t0;

  if(global_pars["src_qc_file0"]) jseisUtil::save_zxy(getFilename("src_qc_file0").c_str(), mylet->mysource, nt, 1, 1, dt * 1000, 1, 1, 0, 0,
                                                      0, 1, 1, 1, 1, jsIO::DataFormat::FLOAT, false, t0);
  if(!getBool("src_correct_after_spread", false)) {
    assertion(dt == global_pars["_dt_prop"].as<float>(), "dt=%f does not match global_pars['_dt_prop']=%f", dt,
              global_pars["_dt_prop"].as<float>());
    PhaseCorrection phaseCorrect(nt, dt, dt, t0, maxfreq, PhaseCorrection::FWD, w_pow, phase_deg);
    phaseCorrect.applyForward(mylet->mysource, 1);

    if(global_pars["src_qc_file1"]) jseisUtil::save_zxy(getFilename("src_qc_file1").c_str(), mylet->mysource, nt, 1, 1, dt * 1000, 1, 1, 0,
                                                        0, 0, 1, 1, 1, 1, jsIO::DataFormat::FLOAT, false, t0);
  }

  if(mylet->waveletType == USERWAVELET) {
    dt_dxyz = 1.0;
  } else {
    dt_dxyz = 1.0 / grid->dz; // FIXME: right now only use grid's const dz
    if(!(dim_bits & OneD)) dt_dxyz /= grid->dx;
    if(dim_bits & ThreeD) dt_dxyz /= grid->dy;
  }
  // print1m("Source::dt_dxyz=%g(dt=%f,dz=%f,dx=%f)\n", dt_dxyz, dt, grid->dz, grid->dx); // no longer used, do not print to avoid confusion

  return mylet;
}
vector<float> Source::getSpectrumPowPhase(PROP::Direction direction, PROP::Operation oper, int dim_bits, int verbose) {
  int do_dipole = Receiver::isDipole(direction, oper);
  bool post_spread = getBool("src_correct_after_spread", false);
  string dipole = post_spread ? "dipole_post_spread" : "dipole";

  float w_pow = 0, phase_deg = 0;
  stringstream ss;
  if(direction == PROP::FORWARD) {
    ss << "################ PROP::FORWARD ####################\n";

    if((dim_bits & Sim3D) && (dim_bits & TwoD)) { // 2D3
      phase_deg = 45, w_pow = 0.5f;
      ss << "    2D3:  w_pow:  0.5,    phase:   45\n";
    } else if((dim_bits & Sim3D) && (dim_bits & OneD)) { // 1D3
      phase_deg = 90, w_pow = 1;
      ss << "    1D3:  w_pow:  1.0,    phase:   90\n";
    }
    if(oper == PROP::RTMM) {
      phase_deg -= 180;
      if(dim_bits & TwoD) {
        phase_deg += 45, w_pow += 0.5;
        ss << "   RTMM:  w_pow += 0.5,    phase -= 135\n";
      } else if(dim_bits & ThreeD) {
        phase_deg += 90, w_pow += 1;
        ss << "   RTMM:  w_pow += 1.0,    phase -= 90\n";
      }
    }

    if(global_pars["global"]["reflectivity"]) { // if demigtation, rotate 90 degreee
      phase_deg += -90;
      ss << "    Demigration:  phase -= 90\n";
    }

    float extra_phase = global_pars["source_extra_phase_deg"].as<float>(0.0f);
    float extra_pow = global_pars["w_pow_fwd"].as<float>(0.0f); // additional user supplied for flexibility
    phase_deg += extra_phase;
    w_pow += extra_pow;
    ss << "    +'w_pow_fwd': " << extra_pow << ", +'source_extra_phase_deg': " << extra_phase << std::endl;
    if(do_dipole) {
      w_pow -= post_spread ? 0 : 1;
      phase_deg += 90;
      ss << "    " << dipole << ": w_pow += " << (post_spread ? 0 : -1) << ", phase += " << 90 << std::endl;
    }
  } else {
    ss << "################ PROP::BACKWARD ###################\n";

    string LowcutFilter = expEnvVars(global_pars["LowcutFilter"].as<string>("Inversion"));
    std::transform(LowcutFilter.begin(), LowcutFilter.end(), LowcutFilter.begin(), ::tolower);
    int doLaplacian = (LowcutFilter == "laplacian");

    w_pow = doLaplacian ? -1 : 1;
    phase_deg = doLaplacian ? 0.0f : 180.0f;
    ss << "  laplacian=" << doLaplacian << " ==> w_pow:  " << w_pow << ",    phase: " << phase_deg << std::endl;

    if((dim_bits & TwoD) && !(dim_bits & Sim3D)) { // pure 2D RTM
      w_pow += 1.0f;
      ss << " 2D (pure): +w_pow: 1.0" << std::endl;
    } else if((dim_bits & OneD) && !(dim_bits & Sim3D)) { // pure 1D RTM, TO BE VERIFIED
      w_pow += 0.5f;
      ss << " 1D (pure, TO-BE-VERIFIED): +w_pow: 0.5" << std::endl;
    }

    float extra_pow = global_pars["w_pow_bwd"].as<float>(0.0f); // additional user supplied for flexibility
    w_pow += extra_pow;
    ss << "    +'w_pow_bwd': " << extra_pow << std::endl;
    if(do_dipole) {
      w_pow -= post_spread ? 0 : 1;
      phase_deg -= 90;
      ss << "    " << dipole << ": w_pow += " << (post_spread ? 0 : -1) << ", phase += " << -90 << std::endl;
    }
    int rotate90 = (1 + global_pars["rotate90"].as<int>(0)) % 4;
    ss << "    RTM additional 90 degrees: 1 + 'rotate90'(" << global_pars["rotate90"].as<int>(0) << ") = " << rotate90 << std::endl;
    phase_deg += 90.0f * rotate90;
    if(phase_deg >= 360) phase_deg -= 360;
  }

  if(oper == PROP::RTM && ((dim_bits & TwoD) && !(dim_bits & Sim3D))) { // pure 2D, balance the pow for RTM
    float extra_pow = (direction == PROP::FORWARD) ? 0.5f : -0.5f;
    w_pow += extra_pow;
    ss << " RTM2D (pure): +w_pow: " << extra_pow << std::endl;
  }

  if(verbose) {
    ss << "    Final w_pow: " << w_pow << ",    phase: " << phase_deg << std::endl;
    std::cout << ss.str();
  }

  return vector<float> { w_pow, phase_deg };
}
