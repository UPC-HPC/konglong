#include "Traces.h"
#include "GetPar.h"

#include <cassert>
#include "libCommon/Assertion.h"
#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;

Traces::Traces(int traceLength, float *extern_mem, int trc_limit, float dt) :
    nt(traceLength), extern_mem(extern_mem), trc_limit(trc_limit), dt(dt) {
  assert(nt >= 0);
  xyzMin.initMin();
  xyzMax.initMax();
}

Traces::~Traces() {
  if(extern_mem) return;

  int ntraces = data.size();
  for(int i = 0; i < ntraces; i++) {
    if(data[i]) delete[] data[i];
    data[i] = 0;
  }
  data.clear();
}

void Traces::addReceiver(vector3 location) {
  size_t idx = coord.size(); // current index
  if(extern_mem) assertion((int )idx < trc_limit, "Extern trc_limit(%d) reached!", trc_limit);

  coord.push_back(location);
  xyzMin.updateMin(location);
  xyzMax.updateMax(location);

  if(nt > 0) {
    float *trace = extern_mem ? extern_mem + idx * nt : new float[nt]();
    data.push_back(trace);
  }
}

int Traces::getNReceivers() const {
  return coord.size();
}

void Traces::validateReceivers(vector<int> map, vector<int> invalid) {
  int ntraces = coord.size(), nlives = map.size(), n_invalid = invalid.size();
  assertion(ntraces == nlives + n_invalid, "Total traces(%d) does not match nlives(%d)+n_invalid(%d)", ntraces, nlives, n_invalid);
  if(ntraces == nlives) return; // nothing need to be done

  if(nt > 0 && !extern_mem) for(int i = 0; i < n_invalid; i++)
    delete[] data[i], data[i] = nullptr;

  for(int i = 0; i < nlives; i++) {
    int j = map[i];
    if(j != i) {
      coord[i] = coord[j];
      if(nt > 0) data[j] = data[i];
    }
  }
  coord.resize(nlives);
  if(nt > 0 && !extern_mem) data.resize(nlives);

  xyzMin.initMin(), xyzMax.initMax();
  for(int i = 0; i < nlives; i++)
    xyzMin.updateMin(coord[i]), xyzMax.updateMax(coord[i]);

}

void Traces::qcData(string filename) {
  assertion(extern_mem || !data.empty(), "Data is empty!");

  int nr = getNReceivers();
  vector<float> dat2;
  if(!extern_mem) {
    dat2.resize((size_t)nr * nt);
    for(int ir = 0; ir < nr; ir++)
      memcpy(&dat2[(size_t)ir * nt], data[ir], sizeof(float) * nt);
  }

  jseisUtil::save_zxy(expEnvVars(filename).c_str(), extern_mem ? extern_mem : &dat2[0], nt, nr, 1, dt > 0 ? dt * 1000 : 4.0f);
}

float* Traces::getTrace(int ir) {
  return extern_mem ? extern_mem + (size_t)ir * nt : data[ir];
}
