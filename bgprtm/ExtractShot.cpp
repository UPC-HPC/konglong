/*
 * ExtractShot.cpp
 *
 */

#include "ExtractShot.h"
//
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "GlobalTranspose.h"
#include "RecordShot.h"
#include "GetPar.h"
#include "libCommon/Vec2.hpp"

using namespace std;

#define SAVE_TO_DISK 0
//
ExtractShot::ExtractShot(const char *fname, const char *path, const int maxTracesPerShot, const vector<ShotLocation> &locations,
                         libSeismicFileIO::TraceHdrEntry *tracehdrEntryPtr)
  : path_(path), maxTracesPerShot_(maxTracesPerShot), locations_(locations) {
  reader       = 0;
  recordShot     = 0;

  transOption = 0;
  if(global_pars["geometry"]) transOption = WORLD2LOCAL;

  open(fname, tracehdrEntryPtr);
}
ExtractShot::~ExtractShot() {
  if(recordShot) delete recordShot;
  recordShot = 0;
  if(reader)
    delete reader;
  reader     = 0;

}


void ExtractShot::open(const char *fname, libSeismicFileIO::TraceHdrEntry *tracehdrEntryPtr) {
  if(reader)
    delete reader;
  reader = new libSeismicFileIO::RecordReader(string(fname));
  if(tracehdrEntryPtr != NULL) reader->setTraceHdrEntry(*tracehdrEntryPtr);
  reader->setCoordinateScale(global_pars["data_coord_scale"].as<float>(1.0));
    string acqOption = expEnvVars(global_pars["acquisition"].as<string>("streamer"));
  if(acqOption == "obn" || acqOption == "obc" || acqOption == "obs")
    reader->swapShotRecv();
}
bool ExtractShot::extract(const int ishot) {
  if(recordShot) delete recordShot;
  recordShot = new RecordShot(transOption);

  //jump to the offset
  long iframe = locations_[ishot].iframe_;

  libCommon::Grid1I frameGrid = reader->getFrameLogical();

  reader->readFrame(iframe, recordShot->traces);

  int shotid = locations_[ishot].sid_;
  char shotname[2024];
  sprintf(shotname, "%s/shot%d.js", path_, shotid);
  if(SAVE_TO_DISK) recordShot->save2disk(shotname);

  return true;
}

bool ExtractShot::save2disk(const int ishot) {
  int shotid = locations_[ishot].sid_;
  long iframe = locations_[ishot].iframe_;
  char shotname[256];
  sprintf(shotname, "%s/shot%d.js", path_, shotid);
  //libCommon::Grid1I frameGrid = reader->getFrameLogical();
  //reader->readFrame(frameGrid.idx_snap(iframe), recordShot->traces);
  reader->readFrame(iframe, recordShot->traces);  // even for the javaseis, iframe is the index of frame, sid is the frame logical.

  recordShot->save2disk(shotname);
  return true;
}
RecordShot *ExtractShot::getShot() {
  RecordShot *retRecord = recordShot;
  recordShot = 0;
  return retRecord;
}

void ExtractShot::print(string space, ostream &os) {
  os << space << " ---------------------- Extract Shot Info  ---------------------" << endl;
  os << space << "path: " << path_ << endl;
  os << space << "maxTracesPerShot: " << maxTracesPerShot_ << endl;
  os << space << "NLocations: " << locations_.size() << endl;
  for(size_t i = 0; i < locations_.size(); i++)
    locations_[i].print(space + "   " + std::to_string(i) + "  ", os);
  os << endl;
}
       
