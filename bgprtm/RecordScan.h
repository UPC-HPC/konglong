/*
 * SEGYSCAN.h
 *
 */

#ifndef RecordScan_H_
#define RecordScan_H_

#include "ShotLocation.h"
#include "libSWIO/RecordIO.hpp"
//
#include <vector>
using namespace libSeismicFileIO;
class RecordScan {
public:

  // ctor
  RecordScan(const char *fname) : myTraceHdrEntryPtr(0) {this->scan(fname);}
  RecordScan(const char *fname, TraceHdrEntry hdrEntry) {myTraceHdrEntryPtr = new TraceHdrEntry(hdrEntry); this->scan(fname);}

  virtual ~RecordScan() {if(myTraceHdrEntryPtr)delete myTraceHdrEntryPtr;}

  void scan(const char *fname);

  void printLog() const;

  const vector<ShotLocation> &getShotLocations() const {
    return locations_;
  }

  void scanData(const char *fname);

  void sortData();

  int getNumberofShot();

  float getDt() {return dt;}

  bool getSwapStatus() {return isSwapShotRecv;}

  void print(string space = "", ostream &os = cout);
protected:
  //
  vector<ShotLocation> locations_;
  float elevation_shift;
  float dt;

  bool isSwapShotRecv;
  int transOption;

private:
  libSeismicFileIO::TraceHdrEntry *myTraceHdrEntryPtr; /* SEGYSCAN_H_ */
};
#endif
