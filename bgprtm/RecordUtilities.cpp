/*
 * RecordUtilities.cpp
 *
 */

#include "RecordUtilities.h"
#include "AssertAlways.h"
#include "GetPar.h"
#include "libSWIO/RecordIO.hpp"
#include "libCommon/Utl.hpp"
#include <cfloat>

RecordUtilities::RecordUtilities() {
}

RecordUtilities::~RecordUtilities() {
}

void RecordUtilities::getRecordDt(const char *fileName, float &dt) {
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fileName));
  dt = reader->getDSample();
  delete reader;
}

void RecordUtilities::getRecordSourceLoc(const char *fileName, float &sourceX, float &sourceY,
    float &sourceZ, int &sourceID) {
  vector<libCommon::Trace *> traces;
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fileName));
  reader->readNextFrame(traces);
  libCommon::Point shotLoc = traces[0]->getShotLoc();
  float elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);
  sourceX = shotLoc.x;
  sourceY = shotLoc.y;
  sourceZ = shotLoc.z - elevation_shift;
  sourceID = traces[0]->getShotID();

  libCommon::Utl::deletePtrVect(traces);
  delete reader;
}

//
void RecordUtilities::getRecordSourceArrayLoc(const char *fileName, float &sourceX, float &sourceY,
    float &sourceZ, int nArraySources, float *sourceArrayX, float *sourceArrayY, float *sourceArrayZ) {

  vector<libCommon::Trace *> traces;
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fileName));
  int nFrames = reader->getNFrames();
  float elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);

  float sourceXsum = 0.0f;
  float sourceYsum = 0.0f;
  float sourceZsum = 0.0f;

  int iSrc = 0;
  int iFrame = 0;
  while(iFrame < nFrames && iSrc < nArraySources) {
    reader->readFrame(iFrame, traces);
    for(auto trace : traces) {
      libCommon::Point shotLoc = trace->getShotLoc();
      sourceArrayX[iSrc] = shotLoc.x + sourceX;
      sourceArrayY[iSrc] = shotLoc.y + sourceY;
      sourceArrayZ[iSrc] = shotLoc.z - elevation_shift + sourceZ;
      sourceXsum += sourceArrayX[iSrc];
      sourceYsum += sourceArrayY[iSrc];
      sourceZsum += sourceArrayZ[iSrc];
      iSrc++;
      if(iSrc >= nArraySources)
        break;
    }
  }
  // use averages as working source location
  sourceX = sourceXsum / nArraySources;
  sourceY = sourceYsum / nArraySources;
  sourceZ = sourceZsum / nArraySources;

  // release memory
  libCommon::Utl::deletePtrVect(traces);

  delete reader;
  printf("Average X,Y,Z of source array components: %f, %f, %f \n", sourceX, sourceY, sourceZ);
}

// process the traces in the record file
void RecordUtilities::peekRecordFile(const char *fileName, int doSequentialShotNum, int &nr,
                                     vector3 &minRecv, vector3 &maxRecv, vector3 &centroidRecv) {

  vector<libCommon::Trace *> traces;
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fileName));
  int nFrames = reader->getNFrames();
  float elevation_shift = global_pars["elevation_shift"].as<float>(0.0f) + global_pars["zMin"].as<float>(0.0f);

  if(nFrames < doSequentialShotNum)
    libCommon::Utl::fatal(string("Peek Record File: doSequentialShotNum is larger than the shot number"));
  reader->readFrame(doSequentialShotNum - 1, traces);

  // # of traces in this shot
  nr = traces.size();

  // 3. get min/max x-y ranges
  centroidRecv.x = 0;
  centroidRecv.y = 0;
  centroidRecv.z = 0;
  for(int itrace = 0; itrace < nr; itrace++) {
    libCommon::Point recvLoc = traces[itrace]->getRecvLoc();
    recvLoc.z = recvLoc.z - elevation_shift;
    if(itrace == 0 || minRecv.x > recvLoc.x) minRecv.x = recvLoc.x;
    if(itrace == 0 || maxRecv.x < recvLoc.x) maxRecv.x = recvLoc.x;
    if(itrace == 0 || minRecv.y > recvLoc.y) minRecv.y = recvLoc.y;
    if(itrace == 0 || maxRecv.y < recvLoc.y) maxRecv.y = recvLoc.y;
    if(itrace == 0 || minRecv.z > recvLoc.z) minRecv.z = recvLoc.z;
    if(itrace == 0 || maxRecv.z < recvLoc.z) maxRecv.z = recvLoc.z;

    centroidRecv.x += recvLoc.x;
    centroidRecv.y += recvLoc.y;
    centroidRecv.z += recvLoc.z;

    delete traces[itrace];
  }

  centroidRecv /= static_cast<float>(nr);
  delete reader;
}

void RecordUtilities::getRecordSourceLoc(string &fileName, float &sourceX, float &sourceY,
    float &sourceZ, int &sourceID) {
  getRecordSourceLoc(fileName.c_str(), sourceX, sourceY, sourceZ, sourceID);
}

void RecordUtilities::getRecordSourceArrayLoc(string &fileName, float &sourceX, float &sourceY,
    float &sourceZ, int nArraySources, float *sourceArrayX, float *sourceArrayY, float *sourceArrayZ) {
  getRecordSourceArrayLoc(fileName.c_str(), sourceX, sourceY, sourceZ, nArraySources, sourceArrayX,
                          sourceArrayY, sourceArrayZ);
}

void RecordUtilities::peekRecordFile(string &fileName, int doSequentialShotNum, int &nr,
                                     vector3 &minRecv, vector3 &maxRecv, vector3 &centroidRecv) {
  peekRecordFile(fileName.c_str(), doSequentialShotNum, nr, minRecv, maxRecv, centroidRecv);
}

