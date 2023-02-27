/*
 * DomainRange.cpp
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include "DomainRange.h"
#include "Propagator.h"
#include "GetPar.h"
#include "RecordUtilities.h"
#include "Geometry.h"

#include "MpiPrint.h"
using MpiPrint::print1m;

//
DomainRange::DomainRange() {

}

DomainRange::~DomainRange() {

}
void DomainRange::getComputingRange(vector<float> &sourceX, vector<float> &sourceY, float &xMinValid, float &xMaxValid, float &yMinValid,
                                       float &yMaxValid, vector3 &minRecv, vector3 &maxRecv, int dim, bool peekRecord) {
  if(global_pars["apertureX"]||global_pars["aperture"]) {
    fprintf(stdout, "Version updated! Please use padX and padY instead of apertureX and apertureY!\n");
    fflush(stdout);
    exit(-1);
  }
  const float padX = global_pars["padX"].as<float>(1000);
  const float padY = global_pars["padY"].as<float>(1000);
  assertion(padX >= 0.0f && padY >= 0.0f, "Invalid padX or padY: must be a positive value.");

  // get min/max x-y ranges of the file
  int nr = 0, nxr = 0, nyr = 0;
  float minSrcX = *min_element(sourceX.begin(), sourceX.end());
  float maxSrcX = *max_element(sourceX.begin(), sourceX.end());
  float minSrcY = *min_element(sourceY.begin(), sourceY.end());
  float maxSrcY = *max_element(sourceY.begin(), sourceY.end());
  float minX, maxX, minY, maxY;

  vector3 centroidRecv;
  if(!peekRecord)  {
    minX = minRecv.x;
    maxX = maxRecv.x;
    minY = minRecv.y;
    maxY = maxRecv.y;
  }
  else if(global_pars["ReceiverTemplateFile"] || global_pars["RTMInputFile"])  {
      string fileName = expEnvVars(global_pars["RTMInputFile"].as<string>());
      int doSequentialShotNum = global_pars["doSequentialShotNum"].as<int>(1);
      RecordUtilities::peekRecordFile(fileName, doSequentialShotNum, nr, minRecv, maxRecv, centroidRecv);
      minX = minRecv.x;
      maxX = maxRecv.x;
      minY = minRecv.y;
      maxY = maxRecv.y;
   }
  else if(global_pars["nreceiversInline"] || global_pars["nxreceivers"])
  {
    if(global_pars["nreceiversInline"])
    {
      nxr = nyr = global_pars["nreceiversInline"].as<int>();
    }
    else if(global_pars["nxreceivers"])
    {
      nxr = global_pars["nxreceivers"].as<int>();
      nyr = global_pars["nyreceivers"].as<int>(1);
    }
    float frx = global_pars["receiverX0"].as<float>();
    float fry = global_pars["receiverY0"].as<float>(0);
    float dxr = global_pars["receiverXinc"].as<float>();
    float dyr = global_pars["receiverYinc"].as<float>(0);
    if(dxr >= 0)
    {
      minX = frx;
      maxX = frx + (nxr - 1) * dxr;
    }
    else
    {
      minX = frx + (nxr - 1) * dxr;
      maxX = frx;
    }
    if(dyr >= 0)
    {
      minY = fry;
      maxY = fry + (nyr - 1) * dyr;
    }
    else
  {
      minY = fry + (nyr - 1) * dyr;
      maxY = fry;
    }

  }
  else {
    minX = minSrcX;
    maxX = maxSrcX;
    minY = minSrcY;
    maxY = maxSrcY;
  }

  // compute CMP bounding box based on input receiver locations for RTM
//  print1m("Minimum X receiver value: %f, Maximum X receiver value: %f, Minimum Y receiver value: %f, Maximum Y receiver value: %f \n", minX,
//         maxX, minY, maxY);

  printf("xMinSource = %f, xMaxSource = %f, yMinSource = %f, yMaxSource = %f\n", minSrcX, maxSrcX, minSrcY, maxSrcY);
  printf("xMinReceiver = %f, xMaxReceiver = %f, yMinReceiver = %f, yMaxReceiver = %f\n", minX, maxX, minY, maxY);

  xMinValid = min(minSrcX , minX) - padX;
  xMaxValid = max(maxSrcX , maxX) + padX;
  yMinValid = min(minSrcY , minY) - padY;
  yMaxValid = max(maxSrcY , maxY) + padY;


  if((dim & TwoD) || (dim & OneD)) yMinValid = yMaxValid = sourceY[0];
  if(dim & OneD) xMinValid = xMaxValid = sourceX[0];
}


