#ifndef LIBCONSTANT_HPP
#define LIBCONSTANT_HPP

#include <cstddef>
class LibConstant {
public:
  static const int MAX_STR_LEN = 512;
  static const int MAX_PAIRS = 128;
  static const int MAX_N_FILES = 128;
  static const int FDM_HEADER_LEN = 512;
  static const int MAX_SELECT_SHOTS = 1024;
  static const int MASTER = 0;
  static const size_t KB = 1024;
  static const size_t MB = KB * KB;
  static const size_t GB = KB * MB;
  static const size_t TB = KB * GB;
  static const size_t BLOCK_SIZE = 100 * MB;
  static constexpr double pi = 3.1415926535897932384626;
  static constexpr double RAD_TO_DEG = 57.29577951308232;
  static constexpr double DEG_TO_RAD = 0.017453292519943295;

  static constexpr int    NULL_INT    = -99999999;
  static constexpr float  NULL_FLOAT  = -1E20;
  static constexpr double NULL_DOUBLE = -1E20;
  static constexpr float  MASK_OUT = -10000;    // mask outside region value
  static constexpr float  MASK_IN  = 0;    // mask inside region value

};

#endif

