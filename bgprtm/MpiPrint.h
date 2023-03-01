/*
 * MpiPrint.h
 *
 *  Created on: May 11, 2022
 *      Author: owl
 */

#ifndef SWPRO_LIBWAVEPROP_MPIPRINT_H_
#define SWPRO_LIBWAVEPROP_MPIPRINT_H_

#include <mpi.h>

namespace MpiPrint {

extern int printed1;
extern int rank;
extern int verbose;

void init();
int print1(const char *fmt, ...); // print only for first shot, controlled by global variable MpiPrint::printed1, unless override by verbose > 0
int printm(const char *fmt, ...); // print on master, unless override by verbose > 1
int print1m(const char *fmt, ...); // print once and only on master, unless override by verbose > 2
int print2m(const char *fmt, ...); // print twice and only on master, unless override by verbose > 2

} // namespace MpiPrint
#endif /* SWPRO_LIBWAVEPROP_MPIPRINT_H_ */

