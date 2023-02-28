/*
 * RecordShot.cpp
 *
 *  Created on: Jun 19, 2017
 *      Author: tiger
 */


#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "libSWIO/RecordIO.hpp"
#include "RecordShot.h"
#include "GlobalTranspose.h"

RecordShot::RecordShot(int tranposeOption, bool isSwapShotRecv)
  : tranposeOption(tranposeOption), isSwapShotRecv(isSwapShotRecv) {
  gTrans = 0;
  if(tranposeOption != NOTRANSPOSE) gTrans = new GlobalTranspose();
}

RecordShot::~RecordShot() {
  libCommon::Utl::deletePtrVect(traces);

  if(gTrans) delete gTrans;
}

void RecordShot::openFile(const char *fileName) {
  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(string(fileName));
  //reader->readNextFrame(traces);
  if(isSwapShotRecv)
    reader->swapShotRecv();
  reader->readNextVolume(traces);

  delete reader;
}



bool RecordShot::save2disk(const char *shotname) {
  if(traces.size() == 0)
    return false;
  vector<libCommon::Grid1D> axis(3);
  axis[0] = traces[0]->getGrid();
  axis[1] = libCommon::Grid1D(traces.size(), 0, 12.5);
  axis[2] = libCommon::Grid1D(1, 0, 12.5);  // at lease 3 axis

  libSeismicFileIO::RecordWriter *writer = new libSeismicFileIO::RecordWriter(string(shotname), &axis);
  // writer->writeNextVolume(traces);
  writer->writeNextFrame(traces);
  delete writer;

  return true;
}

bool RecordShot::sentTo(int nodeIdx) {
  if(traces.size() == 0) {
    libCommon::Utl::fatal("shot is empty!!!");
    return false;
  }
  size_t size = 0;
  for(size_t i = 0; i < traces.size(); i++)
    size += traces[i]->size();
  char *buf = new char[size];
  char *curP = buf;
  for(size_t i = 0; i < traces.size(); i++) {
    if(tranposeOption == WORLD2LOCAL)
      gTrans->worldToLocal(*traces[i]);
    else if(tranposeOption == LOCAL2WORLD)
      gTrans->localToWorld(*traces[i]);

    curP = traces[i]->writeToBuf(curP);
  }
  MPI_Send((char *) &size, sizeof(size_t), MPI_CHAR, nodeIdx, 993,
           MPI_COMM_WORLD);
  libCommon::Utl::send(MPI_COMM_WORLD, buf, size, nodeIdx, 993);

  delete[] buf;
  return true;
}

bool RecordShot::recvFrom(int nodeIdx) {
  // clear memory first
  libCommon::Utl::deletePtrVect(traces);
  size_t size = 0;
  MPI_Recv((char *)&size, sizeof(size_t), MPI_CHAR, nodeIdx, 993, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  char *buf = new char[size];
  libCommon::Utl::recv(MPI_COMM_WORLD, buf, size, nodeIdx, 993);
  char *curP = buf;
  char *endP = buf + size;
  while(curP < endP) {
    libCommon::Trace *trace = new libCommon::Trace;
    curP = trace->readFromBuf(curP);
    traces.push_back(trace);
  }
  delete [] buf;
  return true;
}

void RecordShot::print(string space, ostream &os)const {
  os << space << "------------------ Record Information --------------------------" << endl;
  os << space << "Trace number: " << traces.size() << endl;
  libCommon::Range1I shotRange, recvRange;
  for(auto &p : traces) {
    shotRange += p->getShotID();
    recvRange += p->getRecvID();
  }
  os << space << "Shot ID Range: " << shotRange.begin << " to " << shotRange.end << endl;
  os << space << "Recv ID Range: " << recvRange.begin << " to " << recvRange.end << endl;
  os << endl;
}

