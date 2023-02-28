/*
 * DomainRange.h
 *
 */

#ifndef DOMAINRANGERTM_H_
#define DOMAINRANGERTM_H_

#include <vector>
using std::vector;
class vector3;

/** calculate the computing range for RTM
 *
 */
class DomainRange {
public:
  /** ctor
   *
   */
  DomainRange();

  /** dtor
   *
   */
  virtual ~DomainRange();

  /** get computing range of for current source location
   *
   */
 static void getComputingRange(vector<float> &sourceX, vector<float> &sourceY,
                                float &xMinValid, float &xMaxValid,
                                float &yMinValid, float &yMaxValid,
                                vector3 &minRecv, vector3 &maxRecv, int dim, bool peekRecord = true);

protected:

};

#endif /* DOMAINRANGERTM_H_ */

