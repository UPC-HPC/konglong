/*
 * MpiPrint.cpp
 *
 *  Created on: May 11, 2022
 *      Author: owl
 */

#include "MpiPrint.h"
#include "GetPar.h"

namespace MpiPrint {

int printed1 = 0;
int rank = 0;
int inited = 0;
int verbose = 0;

void init() {
  if(!inited) {
    int mpi_is_init;
    MPI_Initialized(&mpi_is_init);
    if(mpi_is_init) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    inited = 1;
  }
  const char *mpi_verbose = getenv("SEISAPI_MPI_VERBOSE");
  verbose = global_pars["mpi_verbose"].as<int>(mpi_verbose ? atoi(mpi_verbose) : 2);
}

int print1(const char *fmt, ...) {
  init();
  if(!verbose && printed1) return 0;

  va_list ar;
  va_start(ar, fmt);
  int ret = vprintf(fmt, ar);
  va_end(ar);
  return ret;
}

int print1m(const char *fmt, ...) {
  init();
  if(verbose < 2 && (rank != 0 || printed1)) return 0;

  va_list ar;
  va_start(ar, fmt);
  int ret = vprintf(fmt, ar);
  va_end(ar);
  return ret;
}

int print2m(const char *fmt, ...) {
  init();
  if(verbose < 2 && (rank != 0 || printed1 >= 2)) return 0;

  va_list ar;
  va_start(ar, fmt);
  int ret = vprintf(fmt, ar);
  va_end(ar);

  printed1++;
  return ret;
}

int printm(const char *fmt, ...) {
  init();
  if(verbose < 3 && rank != 0) return 0;

  va_list ar;
  va_start(ar, fmt);
  int ret = vprintf(fmt, ar);
  va_end(ar);
  return ret;
}

} // namespace MpiPrint

