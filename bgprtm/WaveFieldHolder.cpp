#include "WaveFieldHolder.hpp"
#include <iostream>
using namespace std;

WaveFieldHolder::WaveFieldHolder(CacheFile* waveFiledCacheFile, WaveFieldCompress* wfComp, int nBuf, int dbl_size){
    myNBuf = nBuf;
    myDblSize = dbl_size;
    mynx = waveFiledCacheFile->nx;
    myny = waveFiledCacheFile->ny;
    mynz = waveFiledCacheFile->nz;
    mynt = waveFiledCacheFile->nt;

    myCompBufSize = WaveFieldCompress::nshort_volume(mynz, mynx, myny);
    myCompBuf.resize(myCompBufSize);
    myCubeSize = (size_t)mynx * (size_t)myny * (size_t)mynz;

    myNSliceRead = 0;

    myAsyncIO = new AsyncIO(waveFiledCacheFile, myCompBufSize * sizeof(ushort));
    myWfComp = wfComp;
}


WaveFieldHolder::~WaveFieldHolder(){
    this->clear();
}

void WaveFieldHolder::clear(){
    for(auto p : myTimeSliceBuf){
        for(int i=0; i<myDblSize; i++)
            delete [] p[i];
        delete [] p;
    }
    myTimeSliceBuf.clear();
    if(myAsyncIO != nullptr){
        delete myAsyncIO;
        myAsyncIO = nullptr;
    }
}
float** WaveFieldHolder::getWaveField(int it){
    if(it<0 || it>= mynt) return 0;
    while(myNSliceRead<it+1){
        this->readNextSlice();
    }
    int n = myTimeSliceBuf.size();
    if(it<myNSliceRead-n)return 0;  // less than the first one
    return myTimeSliceBuf[it - (myNSliceRead-   n)];
}

bool WaveFieldHolder::readNextSlice(){

    if(myNSliceRead>=mynt)return false;

    // read into compressed buffer
    size_t nbytes = myCompBufSize * sizeof(ushort);
    myAsyncIO->pread(&myCompBuf[0], myNSliceRead, &nbytes);

    // get time slice buffer
    float **timeSliceBuf = 0;
    if(myTimeSliceBuf.size() < (size_t)myNBuf){
        timeSliceBuf = new float*[myDblSize];
        for(int i = 0; i < myDblSize; i++)
            timeSliceBuf[i] = new float[myCubeSize];
        myTimeSliceBuf.push_back(timeSliceBuf);
    }else{
        timeSliceBuf = myTimeSliceBuf.front();
        myTimeSliceBuf.erase( myTimeSliceBuf.begin() );  // equal to pop_front()
        myTimeSliceBuf.push_back(timeSliceBuf);
    }

    // decompress the data
    myAsyncIO->join();
    myWfComp->uncompress(&myCompBuf[0], timeSliceBuf[0], mynx, myny, mynz);

    myNSliceRead++;
    return true;
}

void WaveFieldHolder::printCompBuf(){
    double sum = 0;
  for(size_t i=0; i<myCompBufSize; i++)
        sum += myCompBuf[i];

    cout<<endl<<"---------------- CompBuf Info ----------------------- "<<endl;
    cout<<"sum:  "<<sum<<endl;
    cout<<myCompBufSize/2<<":  "<<myCompBuf[myCompBufSize/2]<<endl;
}

