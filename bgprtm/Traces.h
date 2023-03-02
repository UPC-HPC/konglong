#ifndef TRACES_H_
#define TRACES_H_

#include <vector>
#include "Vector3.h"
using std::vector;
#include <string>
using std::string;

class Traces {
public:
  /** ctor
   *
   */
  Traces(int traceLength = 0, float *extern_mem = nullptr, int trc_limit = 0, float dt = 0);

  /** dtor
   *
   */
  virtual ~Traces();

  void addReceiver(vector3 location);
  void validateReceivers(vector<int> map, vector<int> invalid); // keep the mapped receivers only
  void qcData(string filename);
  float *getTrace(int ir);

  int getNReceivers() const;

  vector3 coord_src;
  std::vector<vector3> coord;
  std::vector<float *> data;
  vector3 xyzMin, xyzMax;

  int nt;
  float dt = 0; // normally not used
  float *extern_mem;
  int trc_limit;
};

#endif

