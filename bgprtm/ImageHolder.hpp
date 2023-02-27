#ifndef IMAGEHOLDER_HPP
#define IMAGEHOLDER_HPP

#include <string>
#include <iostream>
#include <vector>
#include "AsyncIO.h"
#include "WaveFieldCompress.h"

class ImageHolder {
public:
  ImageHolder(size_t volSize, int nAmp, int nImg, int nFWI, int nRaw){
        createVolumes(myAMPCubes, volSize, nAmp);
        createVolumes(myImgCubes, volSize, nImg);
        createVolumes(myFWICubes, volSize, nFWI);
        createVolumes(myRawCubes, volSize, nRaw);
    }
  ~ImageHolder(){};

    float* getAMPCube(int idx){return myAMPCubes.empty() ? nullptr : &myAMPCubes[idx][0];}
    float* getImgCube(int idx){return myImgCubes.empty() ? nullptr : &myImgCubes[idx][0];}
    float* getFWICube(int idx){return myFWICubes.empty() ? nullptr : &myFWICubes[idx][0];}
    float* getRawCube(int idx){return myRawCubes.empty() ? nullptr : &myRawCubes[idx][0];}

    int getGatherN(){return myImgCubes.size()>0? myImgCubes.size() : myFWICubes.size();}

    float getMaxAmp(){
        return std::max(vector2DMaxAbs(myAMPCubes), std::max(vector2DMaxAbs(myImgCubes), std::max(vector2DMaxAbs(myFWICubes), vector2DMaxAbs(myRawCubes))));
    }

private:
    void createVolumes(vector< vector<float> >& cubes, size_t volSize, int n){
        if(n==0)return;
        cubes.resize(n);
        for(auto& p : cubes)
            p.resize(volSize, 0);
    }

    float vector2DMaxAbs(vector< vector<float> >& vec2D){
        float maxVal = 0;
        for(auto &p : vec2D)
            maxVal = std::max(maxVal, vector1DMaxAbs(p));
        return maxVal;
    }

    float vector1DMaxAbs(vector<float>& vec1D){
        float maxVal = 0;
        for(auto &p : vec1D)
            maxVal = std::max(maxVal, p);
        return maxVal;
    }


private:

    vector< vector<float> > myAMPCubes;
    vector< vector<float> > myImgCubes;
    vector< vector<float> > myFWICubes;
    vector< vector<float> > myRawCubes;

};


#endif

