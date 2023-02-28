#ifndef MODEL_H
#define MODEL_H

#include "fdm.hpp"
#include "ModelVolumeID.h"
#include <vector>
#include <string>
using std::string;
using std::vector;

class vector3;
//
enum ModelType { ISO = 1, VTI, TTI, ORT, TOR};

class Model {
public:

  //
  Model();

  //
  virtual ~Model();

  float getModelValue(int indexModel, vector3 &x) {return fdms[indexModel]->getvalue(x);}
  float getModelValue(int indexModel, float x, float y, float z) {return fdms[indexModel]->getvalue(x, y, z);}
  float getvel(vector3 x) {return fdms[VEL]->getvalue(x);}
  float getrho(vector3 x) {return fdms[RHO]->getvalue(x);}
  float getreflectivity(vector3 x) {return fdms[REFLECTIVITY]->getvalue(x);}
  float geteps(vector3 x) {return fdms[EPS]->getvalue(x);}
  float getdel(vector3 x) {return fdms[DEL]->getvalue(x);}
  float getdpx(vector3 x) {return fdms[PJX]->getvalue(x);}
  float getdpy(vector3 x) {return fdms[PJY]->getvalue(x);}

  void free(ModelVolID id);
  const Fdm &getFdm(ModelVolID id) const;

  ModelType  modeltype;
  int useRho = 0;
  int useReflectivity = 0;

  std::vector<Fdm *> fdms;

  int useQ = 0;
};

#endif

