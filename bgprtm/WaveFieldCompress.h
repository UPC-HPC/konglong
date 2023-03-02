#ifndef WAVEFIELDCOMPRESS_H_
#define WAVEFIELDCOMPRESS_H_

class WaveFieldCompress {
public:
  enum {COMPRESS_NONE = 0, COMPRESS_16 = 1, COMPRESS_16P = 2, COMPRESS_8P = 3};
  static int compression;
  /*
   * ctor
   */
  WaveFieldCompress(int nThreads);

  /*
   * dtor
   */
  virtual ~WaveFieldCompress();

  static void init_compression();
  static int getCompression();
  static size_t nshort_pack(size_t nfloats); // return num of ushort
  static size_t nshort_volume(int nz, int nx, int ny);

  void compress(float *in, ushort *out, int nx, int ny, int nz);
  void uncompress(ushort *in, float *out, int nx, int ny, int nz);

private:
  void compress16(float *in, ushort *out, int nx, int ny, int nz);
  void uncompress16(ushort *in, float *out, int nx, int ny, int nz);
  int compress16p(float *in, ushort *out, size_t n);
  int uncompress16p(ushort *in, float *out, size_t n);
  int compress8p(float *in, ushort *out, size_t n);
  int uncompress8p(ushort *in, float *out, size_t n);
  int autobit16(float range1, float range2);
  int autobit8(float range1, float range2);

  /*
   * Init the table
   */
  void init();

  /*
   *
   */
  void shift(int *in, ushort *out, size_t n);

private:
  float *table;
  float *wtaper, *wztaper;
  int npoint;
  int nThreads;
  int ntaper, nztaper;
  float eps = 1e-25;
};
#endif /* WAVEFIELDCOMPRESS_H_ */
