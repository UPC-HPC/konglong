/*
 * TimeRecorder.cpp
 *
 *  Created on: Sep 15, 2018
 *      Author: tiger
 */
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>
#include "TimeRecorder.h"

TimeRecorder::TimeRecorder() {
  nTimers = (int)eventRecord::COUNT;
  // elapsed = new float[nTimers]();
}

TimeRecorder::~TimeRecorder() {
  // delete[] elapsed;
}

void TimeRecorder::start(int event) {
  if(event < nTimers) gettimeofday(&t1[event], NULL);
}

void TimeRecorder::end(int event) {
  if(event < nTimers) {
    gettimeofday(&t2[event], NULL);
    elapsed[event] += (t2[event].tv_sec - t1[event].tv_sec + (t2[event].tv_usec - t1[event].tv_usec) * 1E-6);
  }
}

void TimeRecorder::print() {
  char hostname[_POSIX_HOST_NAME_MAX];
  gethostname(hostname, _POSIX_HOST_NAME_MAX);
  printf("\n");
  printf("---------------- Time statistics ------------------\n");
  printf("[%s] Total Elapsed Time (s): %6.2f \n", hostname, elapsed[TOTAL_TIME]);
  printf("          Forward  propagation: %6.2f \n", elapsed[FORWARD_TIME]);
  printf("          Receiver prepare    : %6.2f \n", elapsed[RECEIVER_TIME]);
  printf("          Backward propagation: %6.2f \n", elapsed[BACKWARD_TIME]);
  printf("          Imaging  Condition  : %6.2f \n", elapsed[IMAGING_TIME]);
  printf("          Derivative          : %6.2f \n", elapsed[DERIVATIVE_TIME]);
  printf("          PML                 : %6.2f \n", elapsed[PML_TIME]);
}

