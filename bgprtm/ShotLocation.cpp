#include "ShotLocation.h"
#include "GetPar.h"

#include <math.h>
#include <fstream>
#include <sstream>
using std::ifstream;
using std::istringstream;

ShotLocation::ShotLocation(float sx, float sy, int sid, int rid, float rx, float ry, long iframe, int nrs) :
  sid_(sid), rid_(rid), rx_(rx), ry_(ry), iframe_(iframe), nrs_(nrs) {
  sx_.resize(1, sx);
  sy_.resize(1, sy);
  src_delays.resize(1, 0);
}

ShotLocation::ShotLocation(vector<float> sx, vector<float> sy, vector<float> s_delays, int sid, int rid, float rx,
                           float ry, long iframe, int nrs) :
  sx_(sx), sy_(sy), src_delays(s_delays), sid_(sid), rid_(rid), rx_(rx), ry_(ry), iframe_(iframe), nrs_(nrs) {
}

// NOTE: in the below function, only populate from fname once, release memory when isLast=true
ShotLocation ShotLocation::fromDelayFile(float sx, float shotXinc, float sy, float shotYinc, int sid, int rid, float rx,
    float ry, long iframe, int nrs, int isLast) {
  static map<int, vector<float>> delay_map;

  if(isLast > 1) {
    delay_map.clear();
    return ShotLocation(0, 0, 0, 0, 0, 0, 0, 0);
  }

  if(delay_map.size() == 0) {
    string fname = expEnvVars(global_pars["simshotDelayFile"].as<string>());
    ifstream fs(fname);
    assertion(fs.is_open(), "Could not open %s to read!", fname.c_str());
    for(string line; std::getline(fs, line);) {
      if(line.size() == 0 || line[0] == '#') continue;
      istringstream ss(line);
      int key_sid;
      float val;
      vector<float> vec;
      if(!(ss >> key_sid)) continue;
      while(ss && (ss >> val)) {
        vec.push_back(val);
      }
      if(vec.size() > 0) delay_map[key_sid] = vec;
    }
  }

  vector<float> srcX(1, sx), srcY(1, sy), delays(1, 0);
  if(delay_map.count(sid)) {
    vector<float> vec = delay_map[sid];

    if(global_pars["simshotContinuous"].as<int>(0)) {
      assertion(vec.size() % 2 == 0, "sid/delay are not paired for sid=%d (simshotContinuous=1)", sid);
      for(size_t i = 0; i < vec.size(); i += 2) {
        int sid2 = (int) nearbyintf(vec[i]);
        srcX.push_back(sx + (sid2 - sid) * shotXinc);
        srcY.push_back(sy + (sid2 - sid) * shotYinc);
        delays.push_back(vec[i + 1]);
      }
    } else {
      assertion(vec.size() % 3 == 0, "srcXmod/srcYmod/delay are not tripled for sid=%d (simshotContinuous=0)", sid);
      for(size_t i = 0; i < vec.size(); i += 3) {
        int sid2 = (int) nearbyintf(vec[i]);
        srcX.push_back(sx + vec[i]);
        srcY.push_back(sy + vec[i + 1]);
        delays.push_back(vec[i + 2]);
      }
    }
  }

  if(isLast) delay_map.clear();  // release the memory

  return ShotLocation(srcX, srcY, delays, sid, rid, rx, ry, iframe, nrs);
}

void ShotLocation::print(string space, ostream &os) {
  os << space << "sid " << sid_ << " rid_ " << rid_ << " rx " << rx_ << " ry " << ry_ << " iframe " << iframe_ << " nrs_ " << nrs_;
  for(size_t i = 0; i < sx_.size(); i++) {
    os << " irs " << i << " sx " << sx_[i] << " sy " << sy_[i];
    if(src_delays.size() > i)
      os << " src_delay " << src_delays[i];
  }
  os << endl;
}

