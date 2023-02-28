#ifndef CACHEFILE_H
#define CACHEFILE_H

#include <string>
using std::string;
#include <vector>
using std::vector;

#define SEQ_ZXTY 1
#define SEQ_ZXYT 2

#define SAVE_ORIG 0
#define SAVE_LAPLACE 1
#define SAVE_OMEGA_SQR 2
class CacheFile {
public:
  int seq;
  int content;
  int nx, ny, nz, nt;
  size_t nbytes_vol;
  int fd;
  bool in_mem;
  int write;
  int keep;
  string filename;
  vector<char> mem;

  float dx, dy, dz, dt;
  float x0 = 0, y0 = 0, z0 = 0, t0 = 0, t1 = 0; // when doing IC, ramp from t0 to t1
  static int nfile;
  int header_elements;

public:
  CacheFile(int nx, int ny, int nz, int nt, int seq, const char *prefix, int gid = 0, int write = 1);
  CacheFile(int nx, int ny, int nz, int nt, int seq, std::string fname);
  ~CacheFile();
  void setparas(float xo, float yo, float zo, float tlive, float tfull, float dx0, float dy0, float dz0, float dt0);
  void open(int flags = 0);
  ssize_t pread(void *buf, size_t nbytes, __off_t offset);
  ssize_t pwrite(const void *buf, size_t nbytes, __off_t offset);
  void close();
  void remove();
};

#endif //CACHEFILE_H

