#ifndef MODEL_FILE_FORMAT_H
#define MODEL_FILE_FORMAT_H


#include "libCommon/Volume.hpp"
#include "libSWIO/RecordIO.hpp"
#include "GlobalTranspose.h"
#include <algorithm>

class ModeFileFormat {
public:
  ModeFileFormat(){};
    virtual ~ModeFileFormat(){};

    bool segyToFdm(string sgyFile, string fdmFile, int coorIdx, libSeismicFileIO::TraceHdrEntry* hdrEntry=NULL, GlobalTranspose* gTrans=NULL, libCommon::Grid3D* grid=NULL);
};

#endif

