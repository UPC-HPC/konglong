/*
 * ExtractReceiver.cpp
 *
 */

#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <xmmintrin.h>

#include "Source.h"
#include "ExtractReceiverData.h"
#include "libSWIO/RecordIO.hpp"
#include "ImagingCondition.h"
#include "PhaseCorrection.h"
#include "Grid.h"
#include "GetPar.h"
#include "fdm.hpp"
#include "Util.h"
#include "RecordLoader.h"
#include "RecordUtilities.h"
#include <jseisIO/jseisUtil.h>
using jsIO::jseisUtil;

#include "MpiPrint.h"
using MpiPrint::print1m;
using MpiPrint::print2m;
ExtractReceiverData::ExtractReceiverData(shared_ptr<Grid> grid, int nt, float dt, float t0, int bndType, int nThreads) :
    modGrid(grid), nt(nt), dt(dt), t0(t0), bndType(bndType), nThreads(nThreads) {
  da_removal = global_pars["da_removal"].as<int>(0);
  nr = 0;
  iyr = izr = 0;
  iyrf = izrf = 0.0f;
  zsurf = -global_pars["zMin"].as<float>(0.0f);
  isZslice = 1;
  iz0 = 0;
  recSizeRaw = 0;
  nrz = 0;
  nx = modGrid->nx, ny = modGrid->ny, nz = modGrid->nz;
  nxy = nx * ny;

  assertion(global_pars["tmax"].IsDefined(), "tmax keyword is required for forward modeling!");
  tmax = global_pars["tmax"].as<float>();

  // compute resampleDt and resampleNt for output data
  dtOut = dt;
  if(global_pars["outputDt"]) {
    dtOut = global_pars["outputDt"].as<float>();
  } else if(global_pars["ReceiverTemplateFile"]) {
    string fileName = expEnvVars(global_pars["ReceiverTemplateFile"].as<string>());
    RecordUtilities::getRecordDt(fileName.c_str(), dtOut);
  }

  ntOut = (int)nearbyintf(tmax / dtOut) + 1;
  ntOut = global_pars["outputNt"].as<int>(ntOut);

  khpass = global_pars["rcv_record_khpass"].as<float>(0.40f);
  khcut = global_pars["rcv_record_khcut"].as<float>(0.48f);
  print1m("Receiver recording: khpass=%f, khcut=%f\n", khpass, khcut);
}
ExtractReceiverData::~ExtractReceiverData() {
}

void ExtractReceiverData::specifyReceivers() {
  if(!receivers) specifyReceiverGrid();  // if no templates, use receiver grid
  else {
    nr = receivers->size();
    recSizeRaw = (size_t)nr * nt;
    recDataRaw.resize(recSizeRaw);
    oData.resize((size_t)nr * ntOut);
    if(da_removal) oDataPass1.resize((size_t)nr * ntOut);
  }

}
void ExtractReceiverData::resampleRecData(float *dat) {
  print2m("ExtractReceiverData::resampleRecData(): did_dipole=%d\n", receivers->did_dipole);

  if(global_pars["rec_qc_file0"]) jseisUtil::save_zxy(expEnvVars(global_pars["rec_qc_file0"].as<string>()).c_str(), &recDataRaw[0], nt, nr,
                                                      1, dt * 1000, 1, 1, 0, 0, 0, 1, 1, 1, 1, jsIO::DataFormat::FLOAT, false, -t0);

#if 0 // may not be a good idea, fwd modeling have significant horizontal components
  receivers->resetTraces(&recDataRaw[0], nt, dt, nr);
  receivers->post_spread_correction();
#endif

  if(global_pars["rec_qc_file1a"]) {  // save an example w/o dispersion correction
    PhaseCorrection phaseCorrect(nt, dt, 0, -t0, global_pars["maxFreq"].as<float>(), PhaseCorrection::BWD, 0, 0, 1, ntOut, dtOut); // do resample and shift only!
#pragma omp parallel for
    for(int ir = 0; ir < nr; ir++)
      phaseCorrect.applyBackward(&recDataRaw[(size_t)ir * nt], 1, &dat[(size_t)ir * ntOut]);
    jseisUtil::save_zxy(expEnvVars(global_pars["rec_qc_file1a"].as<string>()).c_str(), &dat[0], ntOut, nr, 1, dtOut * 1000, 1, 1, 0, 0, 0,
                        1, 1, 1, 1, jsIO::DataFormat::FLOAT, false, 0);
  }

  // t0 here is negative or 0, phaseCorrect's t0 is positive or 0
  bool keep_t0 = getBool("modeling_keep_t0", false);
  float w_pow = 0, phase_deg = 0, dt_spread = 1;
  if(receivers->did_dipole) {  //  && !getBool("src_correct_after_spread", false)
    w_pow = -1, phase_deg = 90, dt_spread = 0.5f * receivers->grid->dz / global_pars["vsurf"].as<float>();
  }
  PhaseCorrection phaseCorrect(nt, dt, global_pars["_dt_prop"].as<float>(), -t0, global_pars["maxFreq"].as<float>(), PhaseCorrection::BWD,
                               w_pow, phase_deg, 1, ntOut, dtOut, keep_t0 ? -t0 : 0); // do both phase correction and resample
#pragma omp parallel for
  for(int ir = 0; ir < nr; ir++)
    phaseCorrect.applyBackward(&recDataRaw[(size_t)ir * nt], 1, &dat[(size_t)ir * ntOut], 1 / dt_spread);

   if(global_pars["rec_qc_file1"]) jseisUtil::save_zxy(expEnvVars(global_pars["rec_qc_file1"].as<string>()).c_str(), &dat[0], ntOut, nr, 1,
                                                      dtOut * 1000, 1, 1, 0, 0, 0, 1, 1, 1, 1, jsIO::DataFormat::FLOAT, false, 0);
}

// called by specifyReceivers
void ExtractReceiverData::specifyReceiverGrid() {
  //assertion(receivers, "New code need to create receivers with Receiver::createHdrFromGrid()!");

  float frx = global_pars["receiverX0"].as<float>();
  float fry = global_pars["receiverY0"].as<float>(0);
  float frz = global_pars["receiverZ0"].as<float>(0.0f);
  float dxr = global_pars["receiverXinc"].as<float>();
  float dyr = global_pars["receiverYinc"].as<float>(0);
  float dzr = global_pars["receiverZinc"].as<float>(0);

  int nxr = global_pars["nxreceivers"].as<int>(1);
  int nyr = global_pars["nyreceivers"].as<int>(1);
  int nzr = global_pars["nzreceivers"].as<int>(1);
  if(nxr == 1 && dxr == 0) dxr = 10.0f;
  if(nyr == 1 && dyr == 0) dyr = 10.0f;
  if(nzr == 1 && dzr == 0) dzr = 10.0f;
  //isZslice = (nyr > 1) || modGrid->mytype != RECTANGLE || (bndType & RECEIVER_GHOST); // prefer Y-slice if nyr = 1 and nzr = 1; however for irreg grid, prefer Z-slice
  isZslice = (nzr == 1);
  assertion(nyr == 1 || nzr == 1, "Receiver repetition is not allowed along both y and z direction.");
  assertion(nzr == 1 || modGrid->mytype == RECTANGLE, "Receiver repetition along z requires RECTANGLE grid type.");

  recSizeRaw = nxy * nt;

  izrf = modGrid->getIDzf(frz);
  izr = (int)nearbyintf(izrf);
  float zloc = modGrid->getmyZloc(frz);

  int spread = global_pars["receiverSpreadSize"].as<int>(30);
  int izpmin = MIN(modGrid->getIDz(frz), modGrid->getIDz(-frz));
  int izpmax = MAX(modGrid->getIDz(frz), modGrid->getIDz(-frz));
  int iz1 = MAX(0, izpmin - spread + 1);
  int spread_real = izpmin - iz1 + 1;
  int iz2 = MIN(nz - 1, izpmax + spread_real);

  iz0 = iz1;
  nrz = iz2 - iz1 + 1;
  print1m("receiver: iz1=%d, iz2=%d, nrz=%d \n", iz1, iz2, nrz);

  mycoeffz.resize(nrz);
  Util::CompactOrmsbySpreadCoeff(nrz, &mycoeffz[0], zloc - iz0, khpass, khcut);

  //  if (bndType & RECEIVER_GHOST ) {
  if((bndType == RECEIVER_GHOST) || (bndType == GHOST)) {

    vector<float> mycoeffz1(nrz);

    int izloc = modGrid->getIDz(zsurf) - iz0;

    for(int iz = 0; iz < nrz; iz++) {
      if(2 * izloc - iz >= 0 && 2 * izloc - iz < nrz) {
        mycoeffz1[iz] = mycoeffz[2 * izloc - iz];
      } else {
        mycoeffz1[iz] = 0.0;
      }
      //      print1m("receiver ghost: iz=%d, %f, %f \n", iz, mycoeffz[iz], mycoeffz1[iz]);
    }
    //    OrmsbySpreadCoeff(iz0, nsz, mycoeffz1, zloc, khpass, khcut);
    for(int iz = 0; iz < nrz; iz++) {
      mycoeffz[iz] = mycoeffz[iz] - mycoeffz1[iz];
    }
 }

  //  print1m("bndType= 0x%X \n", bndType);
  print1m("bndType= %d \n", bndType);

  outGrid.reset(
      new Grid(RECTANGLE, isZslice ? nxr : nzr, isZslice ? nyr : nxr, ntOut, isZslice ? dxr : dzr, isZslice ? dyr : dxr, dtOut * 1000, 0,
               tmax * 1000, nThreads));
  outGrid->setupRectangle();
  outGrid->setOrigin(isZslice ? frx : frz, isZslice ? fry : frx);
  recDataRaw.resize(max(recSizeRaw, outGrid->mysize), 0);
  if(da_removal) oDataPass1.resize(outGrid->mysize, 0);

  tmpGrid.reset(
      new Grid(RECTANGLE, isZslice ? modGrid->nx : modGrid->nz, isZslice ? modGrid->ny : modGrid->nx, nt,
               isZslice ? modGrid->dx : modGrid->dz, isZslice ? modGrid->dy : modGrid->dx, dt * 1000, t0 * 1000, tmax * 1000, nThreads));
  tmpGrid->setupRectangle();
  tmpGrid->setOrigin(isZslice ? modGrid->x0 : modGrid->z0, isZslice ? modGrid->y0 : modGrid->x0);

  print1m(isZslice ? "tmpGrid:\n fx,fy,nx,ny,nt,dx,dy,dt=" : "tmpGrid:\n fz,fx,nz,nx,nt,dz,dx,dt=");
  print1m("%f %f %d %d %d %f %f %f \n", tmpGrid->x0, tmpGrid->y0, tmpGrid->nx, tmpGrid->ny, tmpGrid->nz, tmpGrid->dx, tmpGrid->dy,
          tmpGrid->dz);

  print1m(isZslice ? "outGrid:\n fx,fy,nx,ny,nt,dx,dy,dt=" : "outGrid:\n fz,fx,nz,nx,nt,dz,dx,dt=");
  print1m("%f %f %d %d %d %f %f %f \n", outGrid->x0, outGrid->y0, outGrid->nx, outGrid->ny, outGrid->nz, outGrid->dx, outGrid->dy,
          outGrid->dz);
}
oid ExtractReceiverData::extract(int it, float *w0) {
  if(it < 0 || it >= nt) return;  // mem outbound

  if(receivers == NULL) {  // grid receivers
    //if(true) {
    if(isZslice) {
      //#pragma omp parallel for num_threads(nThreads) schedule(static)
      for(size_t ixy = 0; ixy < nxy; ixy++) {
        //int izz = iz + izr1; //??
        float val = 0;
        for(int iz = iz0; iz < iz0 + nrz; iz++) {
          int izz = iz - iz0;
          val += w0[ixy * nz + iz] * mycoeffz[izz];
          //          print1m("iz=%d, coeffz=%f,%d \n",iz,mycoeffz[izz],nrz);
        }
        recDataRaw[it * nxy + ixy] = val;
      }
    } else { // Yslice
#pragma omp parallel for num_threads(nThreads) schedule(static)
      for(size_t i = 0; i < nxy; i++) {
        recDataRaw[it * nxy + i] = w0[iyr * nxy + i];
      }
    }
  } else {  // template receivers
    // assertion(recDataRaw.size() == (size_t )nr * nt, "Memory err: size=%ld != nr(%d)*nt(%d)", recDataRaw);
#pragma omp parallel for num_threads(nThreads) schedule(static)
    for(int ir = 0; ir < nr; ir++) {
      recDataRaw[(size_t)ir * nt + it] = extract_value(w0, ir);
    }
  }
}
float ExtractReceiverData::extract_value(float *wf, int ir) {
#if 1 // moved the method to Receiver class
  return receivers->extract_value(wf, ir, (bndType == RECEIVER_GHOST) || (bndType == GHOST));
#else
  float val = 0;
  float *kx = &(receivers->kernels_x[ir][0]);
  float *ky = &(receivers->kernels_y[ir][0]);
  float *kz = &(receivers->kernels_z[ir][0]);
  int lx = receivers->lxs[ir], ly = receivers->lys[ir], lz = receivers->lzs[ir];
  int offx = receivers->offxs[ir], offy = receivers->offys[ir], offz = receivers->offzs[ir];
  int iz_mirror = modGrid->getIDz(zsurf) - iz0;
  // if (ir == nr / 2) print1m("offz=%d, lz=%d, iz_mirror=%d, iz0=%d\n", offz, lz, iz_mirror, iz0);
  for(int iy = 0; iy < ly; iy++) {
    float cy = ky[iy];
    for(int ix = 0; ix < lx; ix++) {
      float cx = cy * kx[ix];
      size_t off = ((iy + offy) * nx + ix + offx) * nz + offz;
      for(int iz = 0; iz < lz; iz++)
        val += cx * kz[iz] * wf[off + iz];

      if((bndType == RECEIVER_GHOST) || (bndType == GHOST)) {
        size_t off2 = ((iy + offy) * nx + ix + offx) * nz + 2 * iz_mirror - offz;
        // requires 2 * iz_mirror - offz - (lz2-1) >=0, i.e., lz2 <= 2 * iz_mirror - offz + 1
        int lz2 = min(lz, 2 * iz_mirror - offz + 1);
        for(int iz = 0; iz < lz2; iz++)
          val -= cx * kz[iz] * wf[off2 - iz];
      }
    }
  }

  return val;
#endif
}
void ExtractReceiverData::saveRecFile(const char *recFile) {

  if(receivers) {
    saveToRecData(&oData[0]);
    if(da_removal) {
#pragma omp parallel for
      for(int ir = 0; ir < nr; ir++) {
        size_t offset = (size_t)ir * ntOut;
        for(int it = 0; it < ntOut; it++)
          oData[offset + it] = oDataPass1[offset + it] - oData[offset + it];
      }
    }

    if(recFile) {
      //recordLoader::writeRecords(recFile, &oData[0], receivers->traces, nr, ntOut, dtOut);
      saveRecordFile(recFile, &oData[0], nr, ntOut, dtOut);
    }
    return;
  }

  float fx = outGrid->x0;
  float fy = outGrid->y0;
  float ft = outGrid->z0;
  assert(outGrid->z0 == 0);
  int nx = outGrid->nx;
  int ny = outGrid->ny;
  //int nt = outGrid->nz; // is ntOut
  float dx = outGrid->dx;
  float dy = outGrid->dy;
  float dt = outGrid->dz;

  saveToRecData(&recDataRaw[0]);
 // old: recData was not cleared to 0, so it's 2*da+signal, while recDataPass1=da+signal. 2*recDataPass1 - recData
  // new: recData is now da
  if(da_removal) {
    size_t nxy = (size_t)nx * ny;
#pragma omp parallel for
    for(size_t ixy = 0; ixy < nxy; ixy++) {
      size_t offset = ixy * ntOut;
      for(int it = 0; it < ntOut; it++)
        recDataRaw[offset + it] = oDataPass1[offset + it] - recDataRaw[offset + it];
    }
  }

  print1m("Saving modeling data to %s ...\n", recFile);
  int sid = global_pars["_sourceID"].as<int>(1);
  int ridBegin = global_pars["receiverID"].as<int>(1);
  libCommon::Point srcLoc(global_pars["_sourceX"].as<float>(0), global_pars["_sourceY"].as<float>(0), global_pars["_sourceZ"].as<float>(0));
  libCommon::Point recvLoc0(global_pars["receiverX0"].as<float>(0), global_pars["receiverY0"].as<float>(0),
                            global_pars["receiverZ0"].as<float>(0));

  libCommon::Grid1D tGrid(ntOut, ft, dt);
  vector<libCommon::Trace*> traces;
  for(int iy = 0; iy < ny; iy++) {
    for(int ix = 0; ix < nx; ix++) {
      libCommon::Trace *trace = new libCommon::Trace(tGrid, new float[ntOut]());
      trace->setShotID(sid);
      trace->setShotLoc(srcLoc);
      trace->setRecvID(iy * nx + ix + ridBegin);
      trace->setRecvLoc(recvLoc0 + libCommon::Point(ix * dx, iy * dy, 0));
      memcpy((void*)trace->getData(), (void*)&recDataRaw[(iy * nx + ix) * ntOut], ntOut * sizeof(float));
      traces.push_back(trace);
    }
  }
vector<libCommon::Grid1D> axisReal;
  axisReal.push_back(libCommon::Grid1D(ntOut, ft, dt * 0.001));  // weishan here dt is milliseconds ???
  axisReal.push_back(libCommon::Grid1D(nx, fx, dx));
  axisReal.push_back(libCommon::Grid1D(ny, fy, dy));
  vector<libCommon::Grid1I> axisLogical;
  axisLogical.push_back(libCommon::Grid1I(ntOut, 0, 1));  // weishan here dt is milliseconds ???
  axisLogical.push_back(libCommon::Grid1I(nx, (int)(fx / dx + 1.5), 1));
  axisLogical.push_back(libCommon::Grid1I(ny, (int)(fy / dy + 1.5), 1));

  vector<string> axisHdrs;
  axisHdrs.push_back("TIME");
  axisHdrs.push_back("XAXIS");
  axisHdrs.push_back("YAXIS");

  libSeismicFileIO::RecordWriter *writer = new libSeismicFileIO::RecordWriter(string(recFile), &axisReal, &axisLogical, &axisHdrs);

  // write the traces
  for(int iy = 0; iy < ny; iy++) {
    vector<libCommon::Trace*> frame;
    for(int ix = 0; ix < nx; ix++) {
      frame.push_back(traces[iy * nx + ix]);
    }
    writer->writeNextFrame(frame);
  }

  delete writer;

  libCommon::Utl::deletePtrVect(traces);
}
void ExtractReceiverData::saveToRecData(float *dataOut) {
  if(dataOut == NULL) dataOut = &oDataPass1[0];

  if(receivers) {
    resampleRecData(dataOut);
    return;
  }

  vector<float> work(tmpGrid->mysize);
  Util::transpose(nxy, nt, nxy, nt, &recDataRaw[0], &work[0]);
  if(global_pars["rec_qc_file0"]) tmpGrid->savefdm(expEnvVars(global_pars["rec_qc_file0"].as<string>()).c_str(), &work[0]);
  {
    // t0 here is negative or 0, phaseCorrect's t0 is positive or 0
    PhaseCorrection phaseCorrect(nt, dt, global_pars["_dt_prop"].as<float>(), -t0, global_pars["maxFreq"].as<float>(),
                                 PhaseCorrection::BWD);
#pragma omp parallel for
    for(size_t ixy = 0; ixy < nxy; ixy++)
      phaseCorrect.applyBackward(&work[ixy * nt], 1);
  }

  if(global_pars["rec_qc_file"]) tmpGrid->savefdm(expEnvVars(global_pars["rec_qc_file"].as<string>()).c_str(), &work[0]);

  // print1m("data size: nx=%d, ny=%d, nz=%d, %ld\n", tmpGrid->nx, tmpGrid->ny, tmpGrid->nz, data.size());
  outGrid->FillVolume(tmpGrid.get(), &work[0], &dataOut[0]);

}
void ExtractReceiverData::saveRecordFile(const char *fileName, float *data, int ntraceOut, int ntOut, float dtOut) {

  string tempFile = expEnvVars(global_pars["ReceiverTemplateFile"].as<string>(""));
  if(tempFile.length() == 0) {
    RecordLoader::writeRecord(fileName, data, receivers->traces.get(), ntraceOut, ntOut, dtOut);
    return;
  }

  libCommon::Grid1D tGridOut(ntOut, 0, dtOut);
  tGridOut.print();
  libSeismicFileIO::RecordWriter *writer = new libSeismicFileIO::RecordWriter(string(fileName), tempFile, &tGridOut);

  libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(tempFile);

  int ntIn = reader->getNSample();
  float dtIn = reader->getDSample();
  int ntraceIn = reader->getNTrace();
  if(ntraceIn != ntraceOut) libCommon::Utl::fatal(
      string("output ntrace doesn't match with the template file! nTraceIn=") + std::to_string(ntraceIn) + string(", nTranOut=")
          + std::to_string(ntraceOut));

  libCommon::Trace traceIn;
  libCommon::Trace traceOut(tGridOut, new float[ntOut]());
  int count = 0;
  while(reader->readNextTrace(traceIn)) {
    libCommon::TraceHead head = traceIn.getHead();
    head.tGrid = tGridOut;
    traceOut.setHead(head);
    memcpy((void*)(traceOut.getData()), (void*)(data + count * ntOut), ntOut * sizeof(float));
    count++;
    //traceOut.checkData();
    writer->writeNextTrace(traceOut);
  }
 // release memeory
  delete writer;
  delete reader;
}

