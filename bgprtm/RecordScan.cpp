/*
 * RecordScan.cpp
 *
 */

#include "GetPar.h"
#include "RecordScan.h"
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <limits>
#include "GlobalTranspose.h"
#include "libSWIO/RecordIO.hpp"

#ifndef ABS
#define ABS(a) ((a) > 0 ? (a) : (-(a)))
#endif

using namespace std;

void RecordScan::scan(const char *fname) {
  char tblname[2024];
  sprintf(tblname, "%s.tbl", fname);
  dt = 0;

  isSwapShotRecv = false;

  string acqOption = expEnvVars(global_pars["acquisition"].as<string>("streamer"));
  if(acqOption == "obn" || acqOption == "obc" || acqOption == "obs") isSwapShotRecv = true;

  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fname));
  if(myTraceHdrEntryPtr) reader->setTraceHdrEntry(*myTraceHdrEntryPtr);
  if(isSwapShotRecv)  reader->swapShotRecv();
  reader->setCoordinateScale(global_pars["data_coord_scale"].as<float>(1.0));

  dt = reader->getDSample();

  vector<libSeismicFileIO::FrameInfo> frameInfos = reader->getFrameInfos();
  // for(auto &p : frameInfos)
  //   p.print();
  for(size_t iframe = 0; iframe < frameInfos.size(); iframe++) {
    float sx = frameInfos[iframe].sx;
    float sy = frameInfos[iframe].sy;
    float rx = frameInfos[iframe].rx;
    float ry = frameInfos[iframe].ry;
    int shotid = frameInfos[iframe].sid;
    int chanid = frameInfos[iframe].rid;
    int frameID = frameInfos[iframe].frameid;
    int nrs = frameInfos[iframe].ntraces;

    if(global_pars["geometry"]) {
      GlobalTranspose gtrans;
      gtrans.worldToLocal(sx, sy);
      gtrans.worldToLocal(rx, ry);
    }

    if(global_pars["simshotDelayFile"]) {
      locations_.push_back(ShotLocation::fromDelayFile(sx, 0, sy, 0, shotid, chanid, 0, 0, frameID, nrs, true));
    } else {
      locations_.push_back(ShotLocation(sx, sy, shotid, chanid, rx, ry, frameID, nrs));
    }
  }
  delete reader;

  elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);
}

//
struct sxsyCmp {
  bool operator()(const ShotLocation &a, const ShotLocation &b) {
    if(a.sx_ < b.sx_) return true;

    //
    if(a.sx_ == b.sx_) {
      if(a.sy_ < b.sy_) return true;
    }

    //
    return false;
  }
};

void RecordScan::sortData() {
  //
  std::sort(locations_.begin(), locations_.end(), sxsyCmp());
}

int RecordScan::getNumberofShot() {
  return (int) locations_.size();
}

void RecordScan::printLog() const {
  //
  FILE *fp = fopen("recordscan.log", "w");
  if(fp == NULL) {
    printf("ERROR: Can't open the log file \n");
    exit(-1);
  }
  fprintf(fp, "index    shotx    shoty  shotId  ntrace \n");
  for(int i = 0; i < (int) locations_.size(); i++) {
    //
    const ShotLocation &sl = locations_[i];
    fprintf(fp, " %d, %f, %f, %d, %d \n", i, sl.sx_[0], sl.sy_[0], sl.sid_, sl.nrs_);
  }
  fclose(fp);

  printf(" Scan information saved into recordscan.log \n");
}

void RecordScan::print(string space, ostream &os) {
  cout << "----------------Record Scan Info-----------------" << endl;
  cout << "there are " << locations_.size() << " locations." << endl;
  cout << "elevation_shift: " << elevation_shift << endl;
  cout << "dt: " << dt << endl << endl;
}


