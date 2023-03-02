#ifndef WAVEFILEDHOLDER_HPP
#define WAVEFILEDHOLDER_HPP

#include <string>
#include <iostream>
#include <vector>
#include "AsyncIO.h"
#include "WaveFieldCompress.h"

class WaveFieldHolder {
public:
  WaveFieldHolder(CacheFile* waveFiledCacheFile, WaveFieldCompress* wfComp, int nBuf, int dbl_size);
  ~WaveFieldHolder();

    float** getWaveField(int it);
    void clear();

    size_t getCubeSize()const{return myCubeSize;}
    int get_dbl_size()const{return myDblSize;}
    void printCompBuf();
private:
    bool readNextSlice();


private:
    AsyncIO* myAsyncIO; // file IO
    WaveFieldCompress* myWfComp; // decompressed data

    int mynx, myny, mynz, mynt;  // buffer grid size
    int myNBuf;
    size_t myCompBufSize;
    size_t myCubeSize;
    vector<ushort> myCompBuf;   // compressed buff
    int myDblSize;
    vector <  float** > myTimeSliceBuf;
    int myNSliceRead;
};


#endif

