/*
 * SEGYExtractShot.h
 *
 */

#ifndef SEGYEXTRACTSHOT_H_
#define SEGYEXTRACTSHOT_H_

//
#include <vector>
#include "ShotLocation.h"
#include "libSWIO/RecordIO.hpp"

class RecordShot;

class ExtractShot {
public:

  ExtractShot(const char *fname, const char *path, const int maxTracesPerShot, const vector<ShotLocation> &locations,
              libSeismicFileIO::TraceHdrEntry *tracehdrEntryPtr = NULL);

  virtual ~ExtractShot();

  void open(const char *fname, libSeismicFileIO::TraceHdrEntry *tracehdrEntryPtr = NULL);

  bool extract(const int shortid);

  bool save2disk(const int ishot);

  RecordShot *getShot();

  void print(string space = "", ostream &os = cout);

public:
  //
  const char *path_;
  const int maxTracesPerShot_;
  vector<ShotLocation> locations_;
  
  float *traces;
  
  libSeismicFileIO::RecordReader *reader;
  
  int transOption;
  
  //size_t bufSize;
  RecordShot *recordShot;
};

#endif /* SEGYEXTRACTSHOT_H_ */

