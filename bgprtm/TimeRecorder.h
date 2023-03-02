/*
 * TimeRecorder.h
 *
 *  Created on: Sep 15, 2018
 *      Author: tiger
 */

#ifndef LIBWAVEPROP_TIMERECORDER_H_
#define LIBWAVEPROP_TIMERECORDER_H_

#include <sys/time.h>

enum eventRecord {
  FORWARD_TIME = 0,
  BACKWARD_TIME,
  RECEIVER_TIME,
  IMAGING_TIME,
  DERIVATIVE_TIME,
  PML_TIME,
  TOTAL_TIME,
  COUNT
};

class TimeRecorder {
public:
  TimeRecorder();

  virtual ~TimeRecorder();

  void start(int event);

  void end(int envent);

  void print();

  //protected:
  int     nTimers;
  struct timeval t1[COUNT];
  struct timeval t2[COUNT];
  float elapsed[COUNT];
};



#endif /* LIBWAVEPROP_TIMERECORDER_H_ */

