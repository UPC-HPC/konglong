#ifndef WAVEFIELD_H
#define WAVEFIELD_H

class Wavefield {
public:
  float *w0;
  float *w1;
  float *wb;
  float *wx;
  float *wy;
  float *wz;
  float *wr;
  float *d_w0;
  float *d_w1;
  float *d_wb;
  float *d_wx;
  float *d_wy;
  float *d_wz;
  float *d_wr;

  Wavefield(int nx, int ny, int nz);
  ~Wavefield();
  void allocate_mem();
  void allocate_cpu_mem();
  void allocate_gpu_mem();
  void clean_cpu_mem();
  void deallocate_mem();
  void deallocate_cpu_mem();
  void deallocate_gpu_mem();
  void set_data();
  void copy_host2dev();
  void copy_dev2host();
  int compare_host_dev();
private:
  bool cpuInitialized;
  bool gpuInitialized;
  int nx_, ny_, nz_;
};

#endif

