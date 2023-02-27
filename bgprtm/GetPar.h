#ifndef _SWPRO_GETPAR_H_
#define _SWPRO_GETPAR_H_

#include "libCommon/GetPar.h"

extern Node global_pars;

void load_params(const char *ini_file);
void setpar(int argc, char **argv, bool verbose = true);
void merge_cmdline(Node &opt, int argc, char **argv, bool verbose = true);
void parse_mpiexe();

// avoid surprise when user mixed bool/int types
bool getBool(string key_for_global_pars);
bool getBool(string key_for_global_pars, bool defaults);
string getFilename(string key_for_global_pars);
string getJsFilename(string key_for_global_pars, string postfix = "");

int getDimension();
int init_num_threads(bool verbose = false);

#endif


