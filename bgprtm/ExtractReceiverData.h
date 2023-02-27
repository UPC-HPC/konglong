/*
 * ExtractReceiverData.h
 *
 */

#ifndef EXTRACTRECEIVERDATA_H_
#define EXTRACTRECEIVERDATA_H_

#include <vector>
#include <memory>
using std::vector;
using std::unique_ptr;

#include "Receiver.h"

//forward declare
class Grid;

class ExtractReceiverData {
public:
  /*
   * constructor
   */
  ExtractReceiverData(shared_ptr<Grid> grid, int nt, float dt, float t0, int bndType, int nThreads);

  /*
   * destructor
   */
  virtual ~ExtractReceiverData();

  //open recevier file
  // void openFile(char* fileName, bool changeEndian);

  void setNt(int nt, float t0) {
	  this->nt = nt;
    this->t0 = t0;
  }
  void setResampleNt(float resampleDt, int resampleNt) {
    this->dtOut = resampleDt;
    this->ntOut = resampleNt;
    this->tmax = (resampleNt - 1) * resampleDt;
  }

  void specifyReceivers();

  void extract(int it, float *w0);
  float extract_value(float *wf, int ir);

  void saveToRecData(float *data = NULL);
  void resampleRecData(float *dat);
  void saveRecFile(const char *fileName);

private:
  void specifyReceiverGrid();
  void saveRecordFile(const char *fileName, float *data, int ntraceOut, int ntOut, float dtOut);

public:
  int nt, ntOut;
  int nx, ny, nz; // modgrid size
  int nr;
  float dt, dtOut;
  float t0;
  shared_ptr<Receiver> receivers { };
  vector<float> oData;

protected:
  shared_ptr<Grid> modGrid;
  unique_ptr<Grid> outGrid, tmpGrid;
  int bndType;
  int nThreads;
  int da_removal; // direct arrival
  int izr, iyr;
  int iz0, nrz;
  float zsurf = 0;
  float izrf, iyrf;
  bool isZslice;
  float tmax;
  float khpass = 0.5, khcut = 0.5;

  size_t recSizeRaw;
  size_t nxy;

  vector<float> recDataRaw, oDataPass1, mycoeffz; // recDataPass1: pass1 for direct arrival

};

#endif /* EXTRACTRECEIVERDATA_H_ */

