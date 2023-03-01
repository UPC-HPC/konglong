/*
 * RecordUtilities.h
 *
 */

#ifndef RECORDUTILITIES_H_
#define RECORDUTILITIES_H_

#include "Vector3.h"
#include <string>
using std::string;

/** get source locations from Record file
 *
 */
class RecordUtilities {
public:
  /** ctor
   *
   */
  RecordUtilities();

  /** dtor
   *
   */
  virtual ~RecordUtilities();

  /** get a source location from a Record file
   *  @param fileName       Record file name
   *  @param sourceX        returned source x coordinate
   *  @param sourceY        returned source y coordinate
   *  @param sourceZ        returned source z coordinate
   */
  static void getRecordSourceLoc(const char *fileName,
                                 float &sourceX, float &sourceY, float &sourceZ, int &sourceID);
  static void getRecordSourceLoc(string &fileName,
                                 float &sourceX, float &sourceY, float &sourceZ, int &sourceID);

  /** get source array locations from a Record file
   *  @param fileName       Record file name
   *  @param sourceX        returned source x coordinate
   *  @param sourceY        returned source y coordinate
   *  @param sourceZ        returned source z coordinate
   *  @param nArraySources  # of source arrays
   *  @param sourceArrayX   pointer to source x coordinate arrays
   *  @param sourceArrayY   pointer to source y coordinate arrays
   *  @param sourceArrayZ   pointer to source z coordinate arrays
   */
  static void getRecordSourceArrayLoc(const char *fileName,
                                      float &sourceX, float &sourceY, float &sourceZ,
                                      int nArraySources,
                                      float *sourceArrayX, float *sourceArrayY, float *sourceArrayZ);
  static void getRecordSourceArrayLoc(string &fileName,
                                      float &sourceX, float &sourceY, float &sourceZ,
                                      int nArraySources,
                                      float *sourceArrayX, float *sourceArrayY, float *sourceArrayZ);

  /** take a look at the Record receiver data
   *
   */
  static void peekRecordFile(const char *fileName, int doSequentialShotNum,
                             int &nr,
                             vector3 &minRecv,
                             vector3 &maxRecv,
                             vector3 &centroidRecv);
  static void peekRecordFile(string &fileName, int doSequentialShotNum,
                             int &nr,
                             vector3 &minRecv,
                             vector3 &maxRecv,
                             vector3 &centroidRecv);

  static void getRecordDt(const char *fileName, float &dt);
protected:

};

#endif /* RecordUTILITIES_H_ */

