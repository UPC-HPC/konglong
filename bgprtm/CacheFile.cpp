#include <assert.h>
#include <errno.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "CacheFile.h"
#include "WaveFieldCompress.h"
#include "GetPar.h"
#include "libCommon/io_util.h"
using libCommon::pread_all;
using libCommon::pwrite_all;

#include "MpiPrint.h"
using MpiPrint::print1m;

int CacheFile::nfile = 0;

void CacheFile::setparas(float xo, float yo, float zo, float tlive, float tfull, float dx0, float dy0, float dz0, float dt0) {
  x0 = xo;
  y0 = yo;
  z0 = zo;
  t0 = tlive;
  t1 = tfull;
  dx = dx0;
  dy = dy0;
  dz = dz0;
  dt = dt0;
}
CacheFile::CacheFile(int nx, int ny, int nz, int nt, int seq, const char *prefix, int gid, int write) : seq(seq), content(SAVE_ORIG), nx(
    nx), ny(ny), nz(nz), nt(nt), fd(-1), write(write), keep(0) { // cache file is in z,x,t,y order
  dt = 0, dx = 0, dy = 0, dz = 0;
  t0 = 0, x0 = 0, y0 = 0, z0 = 0;
  nbytes_vol = sizeof(ushort) * WaveFieldCompress::nshort_pack(nz) * nx * ny * nt;
  //header_elements = 128; // num of floats, so size *= sizeof(float)
  header_elements = 0;
  std::stringstream strm;
  strm << prefix << "_p" << getpid() << "_g" << gid << "_" << nfile << ".cache";
  filename = strm.str();
  nfile++;
  float cache_limit = global_pars["cache_mem_limit"].as<float>(0.0);
  float mem_needed = nbytes_vol / (1024.0 * 1024.0 * 1024.0);
  in_mem = (cache_limit >= mem_needed);
  static int printed;
  if(!printed)
    print1m("CacheFile: 'cache_mem_limit'=%fGB, mem_needed=%fGB, in_mem=%d, MemFree: %s\n", cache_limit, mem_needed, in_mem,
            libCommon::Utl::free_memory().c_str()), printed = 1;
  this->open(O_TRUNC | O_CREAT | O_RDWR);
}

// only use this function for reading in different program. In same program rely on the CacheFile object
CacheFile::CacheFile(int nx, int ny, int nz, int nt, int seq, std::string fname) : seq(seq), content(SAVE_ORIG), nx(nx), ny(ny), nz(nz), nt(
    nt), fd(-1), write(0), keep(0), filename(fname) { // cache file is in z,x,t,y order
  dt = 0, dx = 0, dy = 0, dz = 0;
  t0 = 0, x0 = 0, y0 = 0, z0 = 0;
  nbytes_vol = sizeof(ushort) * WaveFieldCompress::nshort_pack(nz) * nx * ny * nt;
  // re-open existing cache file on disk, nothing need to be done here ...
  //header_elements = 128; // num of floats, so size *= sizeof(float)
  header_elements = 0;
  this->open(O_RDONLY);
}
CacheFile::~CacheFile() { // if need to keep the cache file, set this->keep to 1 before delete the object
  if(!keep) this->remove();
  this->close();
}

void CacheFile::open(int flags) {
  if(in_mem) {
    mem.resize(nbytes_vol);
    return;
  }
  const char *mkdir_umask = getenv("MKDIR_UMASK");
  string mask = mkdir_umask ? mkdir_umask : "000";
  mode_t mode = 0777 & ~strtol(mask.c_str(), NULL, 8);

  if(fd >= 0) return;
  fd = ::open(filename.c_str(), flags, mode);

  assertion(fd >= 0, "Cannot create file %s !", filename.c_str());
  print1m("The cachefile (fd=%d) is %s\n", fd, filename.c_str());
}

void CacheFile::close() {
  if(!mem.empty()) vector<char>().swap(mem); // release the memory explicitly
  if(fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

void CacheFile::remove() {
  this->close();
  if(!in_mem) unlink(filename.c_str());
}
ssize_t CacheFile::pread(void *buf, size_t nbytes, __off_t offset) {
  if(!in_mem) return pread_all(fd, buf, nbytes, offset);

  memcpy(buf, &mem[offset], nbytes);
  return nbytes;
}

ssize_t CacheFile::pwrite(const void *buf, size_t nbytes, __off_t offset) {
  if(!in_mem) return pwrite_all(fd, buf, nbytes, offset);

  memcpy(&mem[offset], buf, nbytes);
  return nbytes;
}
                 
