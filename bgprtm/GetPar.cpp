#include "GetPar.h"
#include "merge_yaml.hpp"
#include "MpiPrint.h"
using MpiPrint::print1m;
#include "version.h"
#include <unistd.h>
#include <limits.h>    // std::cout, std::fixed
#include <iomanip>
#include <omp.h>

//Node global_pars = NULL;

static void version() {
  cout << get_selfpath() << ": " << GIT_TAG_VERSION << endl;
  cout << "Git sha: " << GIT_SHA_SHORT << " (" << GIT_SHA << ")" << endl;
  cout << "Date commit: " << GIT_DATE << endl;
  cout << "Date compiled: " << VER_DATE << endl << endl;
  string ver = GIT_SHA_SHORT;
  if(strstr(GIT_TAG_VERSION, "dirty")) ver += "_d";
  setenv("ver", ver.c_str(), true);
  cout << "setenv('ver','" << ver << "') (e.g. $ver in file outputs)" << endl;
}

void load_params(const char *ini_file) {
  global_pars = loadYaml(ini_file);
}

void setpar(int argc, char **argv, bool verbose) {
  if(verbose) version();

  if(argc < 2 && verbose) {
    fprintf(stderr, "Format: %s yaml_file\n", argv[0]);
    exit(-1);
  }
  merge_cmdline(global_pars, argc, argv, verbose);

  if(global_pars.IsNull() || !global_pars.IsDefined()) {
    if(verbose) fprintf(stderr, "Failed to load YAML file or key/value paris\n");
    exit(-1);
  }

  parse_mpiexe();
}

void parse_mpiexe() {
  char exe[PATH_MAX + 1];
  ssize_t count = readlink("/proc/self/exe", exe, PATH_MAX);
  exe[count] = 0;
  char *pt = strrchr(exe, '/') + 1;
  if(strncmp(pt, "mpi", 3) == 0) {
    memmove(pt, pt + 3, strlen(pt + 3) + 1); // avoid strcat() to handle overlapping memory
    if(global_pars["exe"]) fprintf(stderr, "WARNING: original 'exe' key will be overwritten by %s\n", exe);
    global_pars["exe"] = exe;
  }
}


void merge_cmdline(Node &opt, int argc, char **argv, bool verbose) {
  for(int i = 1; i < argc; i++) {
    char *pt = strchr(argv[i], '=');
    Node opt2;
    if(pt == NULL) {
      if(!(strstr(argv[i], ".yaml") || strstr(argv[i], ".yml"))) {
        printf("%s need to be either .yaml/.yml file or 'key=value' pair!\n", argv[i]);
        exit(-1);
      }
      opt2 = YAML::LoadFile(argv[i]);
      assertion(!opt2.IsNull(), "Failed to load yaml file %s", argv[i]);
    } else {
      size_t c = pt - argv[i];
      char key[80];
      // printf("argv=%s, pt=%p, argv=%p, c=%ld\n", argv[i], pt, argv[i], c);
      assertion(c < 80, "key '%s' is too long!", argv[i]);
      strncpy(key, argv[i], c), key[c] = '\0';
      string val = pt + 1;
      opt2 = yaml_parse_tree(key, val);
      assertion(!opt2.IsNull(), "Failed to load (key, value) (%s,%s)", key, val.c_str());
    }
    if(opt.IsNull()) opt = opt2;
    else jb::yaml::merge_node(opt, opt2);
  }

  if(opt["env"]) {
    for(auto it = opt["env"].begin(); it != opt["env"].end(); ++it) {
      string key = (it->first).as<string>();
      string value = (it->second).as<string>();
      if(verbose) cout << "setenv('" << key << "','" << value << "')" << endl;
      setenv(key.c_str(), value.c_str(), true);
    }
    if(verbose) cout << endl;
  }
}

bool getBool(string key) { // throws YAML::InvalidNode
  assertion(bool(global_pars[key]), "getBool(%s): The node is null, probably the key does not exist!", key.c_str());

  try {
    int value = global_pars[key].as<int>();
    return value;
  } catch(const YAML::BadConversion &e) {
    try {
      bool value = global_pars[key].as<bool>();
      return value;
    } catch(const YAML::BadConversion &e) {
      assertion(false, "Failed to convert YAML node to bool!");
    }
  }
  return false;
}

bool getBool(string key, bool defaults) {
  return getBool(global_pars[key], defaults);
}

string getFilename(string key) {
  return expEnvVars(global_pars[key].as<string>(""));
}

string getJsFilename(string key, string postfix) {
  string filename = getFilename(key);
  if(postfix.empty()) return filename;
  if(path_extension_lc(filename).compare("js") != 0) return filename + postfix + ".js";

  auto idx = filename.rfind('.'); // can always find it
  return filename.substr(0, idx) + postfix + ".js";
}

int getDimension() {
  string dim_str = global_pars["dimension"].as<string>("3D");
  return atoi(dim_str.c_str());
}


int init_num_threads(bool verbose) {
  int max_threads = omp_get_max_threads();
  int nThreads = global_pars["nThreads"].as<int>(max_threads);  // If this isn't specified, use all threads
  if(nThreads < 1) nThreads = max_threads;
  if(nThreads >= 6) {
    nThreads -= 2;
    if(verbose) print1m("OMP num_threads is: %d (auto-reduced by 2)\n", nThreads);
  } else if(verbose) print1m("OMP num_threads is: %d\n", nThreads);
  return nThreads;
}


