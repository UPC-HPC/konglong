/*
 * Params.h
 *
 *  Created on: May 15, 2016
 *      Author: tiger
 */

#ifndef PARAMS_H_
#define PARAMS_H_

#include <stdio.h>
#include <vector>
using std::vector;

class MigParams {
public:
  //model, migration and image type
  int modType;
  int imgType;

  int gridType;

  //source, receiver
  float sdepth;
  float rdepth;

  // shot information
  int isbeg;
  int isend;
  int isinc;
  int nshot;
    int output_batch_n;  // middel output result for qc and further restart, 0 means no midle ouput


  // Input traces
  int trlen;      // Trace length
  int srate;      // Trace sample rate
  int maxTracesPerShot;// Max number of traces per shot in the current dataset
  int curshot;            // Current shot number
  int ntraces;      // How many traces for the current shot
  vector<float> shotDelays;
  vector<float> shotx;      // Shot position
  vector<float> shoty;
  float shotz;      // only for INVOBC && FLTREC.
  float *traces;    // Trace buffer, ntraces * trlen

  float recX0;
  float recY0;
  int   sid;
  int   rid;

  int shotcount;

  float timeout;    // Timeout in seconds for a shot.

  // GPU parameters
  int firstgpu;     // First GPU to use
  int ngpus;      // How many GPUs.

  // Wavefieds eventually stored in memory instead of disk.
  int usemem;

  float dt;
};


class ShotInfo {
public:
  float maxIn;      // Amplitudes
  float maxOut;

  float elapsed;    // Timers
  float shotWait;
  float srcPrep;
  float rcvPrep;
  float modPrep;
  float srcModeling;
  float rcvModeling;
  float imaging;

};

class IOParams {
public:
  // Input dataset
  void *inputId;
  long long ntraces;    // Number of instances
  int *inputOrdinals;   // Index of the attribute
  int *inputTypes;    // Attribute types
  int *inputLengths;    // Attribute length
  int nshots;     // Total number of shots in the dataset
  float *shotXForShot;    // Shot positions
  float *shotYForShot;
  float shotxmin;   // Min / Max coordinates of shots
  float shotxmax;
  float shotymin;
  float shotymax;
  int *shotNumberForShot; // Shot numbers
  long long *firstTraceForShot; // Trace bounds for each shot
  long long *lastTraceForShot;
  float vsurf;
  int numpx;
  int numpy;
  float *px;
  float *py;

  // Shot processing status
  int curindex;     // Index of the current shot ready to be sent.
  int *shotStatus;    // Array with status of every shot.
  int *slaveForShot;    // Which slave is processing this shot
  int shotsAvailable;   // Number of shots to be processed
  int allShotsDone;   // Flag when all shots have been processed.
  ShotInfo *shotInfos;

  // Output dataset
  void *outputId;   // output Store ID.
};

class NodeParams {
public:
  int mpiRank;      // MPI_Comm_Rank
  int mpiSize;      // MPI_Comm_Size

  int nodeRank;     // Rank inside the node
  int nodeTotal;    // Number of MPI processes on the same node
  int active;     // Active flag

  int superslave; // Superslave flag (1 = has lowest MPI rank on the host, or 0)
  int nextSuperslave;   // rank of the next superslave, or -1 if no other.
  int prevSuperslave; // rank of the previous superslave, or -1 if no other.

  int prevRank;     // Previous MPI rank, or -1.
  int nextRank;     // Next MPI rank, or -1.

  char scratchdir[FILENAME_MAX];

  // Master information about all the slaves
  char *hostnames;// All hostnames (MAXHOSTNAME characters per MPI process)
  int *superslaves;   // If a node is superslave or not
  int *hosts;     // Machine number for each MPI process
  int *shotsPerHost;  // Number of shots that can be processed on that host.
  int *shotsInMem;    // If the wavefields are stored in memory.
  int *hostGPUs;    // Number of GPUs on each host
  int *activeSlaves;    // Which slaves are active.
  int numDeadSlaves;    // Number of dead slaves

};

struct PartialImageInfo {

  PartialImageInfo(int shotid, float maxValue)
    : shotid(shotid), maxValue(maxValue)
  {}
  int shotid;
  float maxValue;
};

//
struct maxCmp {
  bool operator()(const PartialImageInfo &a, const PartialImageInfo &b) {
    if(a.maxValue > b.maxValue)
      return true;
    else
      return false;
  }
};

#endif /* PARAMS_H_ */

