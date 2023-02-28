/*
 * RecordShot.h
 *
 *  Created on: Jun 19, 2017
 *      Author: tiger
 */

#ifndef RECORDSHOT_H_
#define RECORDSHOT_H_

#include <stddef.h>
#include "libCommon/Trace.hpp"
#include "libCommon/Utl.hpp"

class GlobalTranspose;

class RecordShot {

public:

  RecordShot(int transposeOption = 0, bool isSwapShotRecv = false);
  virtual ~RecordShot();

  bool save2disk(const char *shotname);

  void openFile(const char *fileName);

  bool sentTo(int nodeIdx);
  bool recvFrom(int nodeIdx);

  int getShotID()const {return traces.size() > 0 ? traces[0]->getShotID() : 0 ;}
  libCommon::Point getShotLoc() {return traces.size() > 0 ? traces[0]->getShotLoc() : libCommon::Point(0, 0, 0);}

  void print(string space = "", ostream &os = cout)const;

public:
  vector<libCommon::Trace *> traces;

  int tranposeOption;
  bool isSwapShotRecv;
  GlobalTranspose *gTrans;

};



#endif /* SEGYSHOT_H_ */

