
#ifndef TRACE_HDR_DEFINE_H_
#define TRACE_HDR_DEFINE_H_

#include <yaml-cpp/yaml.h>
#include "libSWIO/RecordIO.hpp"

class TraceHdrDefine {
public:
  static libSeismicFileIO::TraceHdrEntry readSegyFormat(YAML::Node node) {
    libSeismicFileIO::TraceHdrEntry t;
    t.shotID.pos = node["loc_shotID"].as<int>(-1) - 1;  // location begin from 0, and user input usually begin from 1
    t.recvID.pos = node["loc_recvID"].as<int>(-1) - 1;
    t.shotX.pos =  node["loc_shotX"].as<int>(-1) - 1;
    t.shotY.pos = node["loc_shotY"].as<int>(-1) - 1;
    t.shotZ.pos = node["loc_shotZ"].as<int>(-1) - 1;
    t.recvX.pos = node["loc_recvX"].as<int>(-1) - 1;
    t.recvY.pos = node["loc_recvY"].as<int>(-1) - 1;
    t.recvZ.pos = node["loc_recvZ"].as<int>(-1) - 1;
    t.shotID.type =  t.recvID.type = node["type_ID"].as<int>(libSeismicFileIO::DataType::INT);
    t.shotX.type = t.shotY.type = t.recvX.type = t.recvY.type = node["type_XY"].as<int>(libSeismicFileIO::DataType::DOUBLE);
    t.shotZ.type = t.recvZ.type = node["type_Z"].as<int>(libSeismicFileIO::DataType::FLOAT);

    t.xline.pos = node["loc_crossline"].as<int>(193) - 1;
    t.yline.pos = node["loc_inline"].as<int>(189) - 1;
    t.xline.type = t.yline.type = libSeismicFileIO::DataType::INT;
    return t;
  }
};

#endif

