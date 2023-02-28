#include <unistd.h>
#include "fdm.hpp"
#include "Model.h"

Model::Model() {
  fdms.resize(SIZE_ModelVolID, NULL);
  for(size_t id = 0; id < fdms.size(); id++) {
    fdms[id] = NULL;
  }
  modeltype = ISO;
  useRho = 0;
}

Model::~Model() {
  for(size_t id = 0; id < fdms.size(); id++) {
    if(fdms[id] != NULL) {
      delete fdms[id];
      fdms[id] = NULL;
    }
  }
}

void Model::free(ModelVolID id) {
  if(fdms[id] != NULL) {
    delete fdms[id];
    fdms[id] = NULL;
  }
}

const Fdm &Model::getFdm(ModelVolID id) const {
  return *fdms[id];
}

