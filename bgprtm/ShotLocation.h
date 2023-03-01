/*
 * ShotLocation.h
 *
 *  Created on: Jun 18, 2017
 *      Author: tiger
 */

#ifndef SHOTLOCATION_H_
#define SHOTLOCATION_H_

#include <vector>
#include <map>
#include <iostream>
#include <string>
using std::vector;
using std::map;
using namespace std;

struct ShotLocation {

  ShotLocation(float sx, float sy, int sid, int rid, float rx, float ry, long iframe, int nrs);
  ShotLocation(vector<float> sx, vector<float> sy, vector<float> s_delays, int sid, int rid, float rx, float ry,
               long iframe, int nrs);

  //
  vector<float> sx_, sy_;
  vector<float> src_delays;
  int sid_, rid_;
  float rx_, ry_;
  long iframe_;        // frame logical idx
  int nrs_;           // # of traces

  static ShotLocation fromDelayFile(float sx, float shotXinc, float sy, float shotYinc, int sid, int rid,
                                    float rx, float ry, long iframe, int nrs, int isLast);
  void print(string space = "", ostream &os = cout);
};

#endif /* SHOTLOCATION_H_ */

