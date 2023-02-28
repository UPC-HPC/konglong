#ifndef MODEL_VOLUME_ID_H
#define MODEL_VOLUME_ID_H

enum ModelVolID {
  VEL = 0, RHO, REFLECTIVITY, Q, EPS, DEL, PJX, PJY, VEL2, EPS2, DEL2, PJZ, SIZE_ModelVolID //always keep SIZE_ModelVolID at the last, RHO must be before TTI params
};
// VEL2 for dual flood by wolf on Nov 4, 2022
#endif

