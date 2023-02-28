/*
 * ModelRegrid.h
 *
 */

#ifndef MODELREGRID_H_
#define MODELREGRID_H_

#include <string.h>

class MigParams;
class NodeParams;
class IOParams;

class ModelRegrid {
public:
  /** constructor
   *
   */
  ModelRegrid();

  ~ModelRegrid();

  void prepModel(MigParams *migParams, NodeParams *nodeParams);

  void broadcastGrid(MigParams *migParams, NodeParams *nodeParams);

  void broadcastModels(MigParams *migParams, NodeParams *nodeParams);

  void broadcastModel(NodeParams *nodeParams, std::string &fileName1, std::string &fileName2);

  void setPath(const char *path);

  std::string path;
};


#endif /* MODELREGRID_H_ */

