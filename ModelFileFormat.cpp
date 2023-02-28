#include "ModelFileFormat.h"

using namespace std;

bool ModeFileFormat::segyToFdm(string sgyFile, string fdmFile, int coorIdx,
        libSeismicFileIO::TraceHdrEntry* hdrEntry, GlobalTranspose* gTrans, libCommon::Grid3D* grid){

    // initial the segy reader
    libSeismicFileIO::RecordReader *reader = new libSeismicFileIO::RecordReader(sgyFile);

    // define the trace header
    if(hdrEntry)
        reader->setTraceHdrEntry(*hdrEntry);

    // read all the traces
    vector<libCommon::Trace *> traces, frame;
    while(reader->readNextFrame(frame)){
        traces.insert(traces.end(), frame.begin(), frame.end());
        frame.clear();
    }

    // release memeory
    delete reader;

    // 3 point world to local 
    if(gTrans)
        for(auto &p : traces)
            gTrans->worldToLocal(*p);

    // get location coordinates
    vector<libCommon::Point> locs;
    for(size_t i=0; i<traces.size(); i++){
        if(coorIdx == 0)
            locs.push_back(traces[i]->getShotLoc());
        else if(coorIdx == 1)
            locs.push_back(traces[i]->getRecvLoc());
        else if(coorIdx == 2)
            locs.push_back(traces[i]->getCdpLoc());
    }

    libCommon::Grid3D fdmGrid;
    if(grid){
        fdmGrid = *grid;
    }else{
        // get range
        libCommon::Range3D range;
        for(auto p : locs){
            range += p;
        }

        double x0 = round(range.xRange.begin);
        double y0 = round(range.yRange.begin);
        double dx = 1E20;
        double dy = 1E20;
        for(size_t i=1; i<locs.size(); i++) {
            double curDx = fabs(locs[i].x-locs[i-1].x);
            double curDy = fabs(locs[i].y-locs[i-1].y);
            dx = curDx > 2 ? min(dx, curDx) : dx;
            dy = curDy > 2 ? min(dy, curDy) : dy;
        }

        libCommon::Range1D xRange, yRange;
        int nx = 0;
        for(size_t i=0; i<locs.size(); i++){
            if(fabs(locs[i].y - y0)<0.5*dy){
                xRange += locs[i].x;
                nx++;
            }
        }
        dx = xRange.length()/(nx-1);
        dx = (int)(dx*10+0.5)/10.0;  // snap to 0.1

        int ny = 0;
        for(size_t i=0; i<locs.size(); i++){
            if(fabs(locs[i].x - x0)<0.5*dx){
                yRange += locs[i].y;
                ny++;
            }
        }
        dy = yRange.length()/(ny-1);
        dy = (int)(dy*10+0.5)/10.0;  // snap to 0.1

        // zGrid unit is m or ms, so if too small, need to convert from s->ms
        libCommon::Grid1D zGrid = traces[0]->getGrid();
     if(zGrid.step < 1 )
            zGrid = libCommon::Grid1D(zGrid.num, zGrid.begin, zGrid.step * 1000);

     // get the new grid now    
     fdmGrid = libCommon::Grid3D(libCommon::Grid1D(nx, x0, dx), libCommon::Grid1D(ny, y0, dy), zGrid);
    }

    float* data = new float[fdmGrid.size()]();
    for(size_t i=0; i<locs.size(); i++){
        int idx = fdmGrid.idx_snap(locs[i]);
        memcpy((void*)(data+idx), (void*)(traces[i]->getData()), traces[i]->getNSamples()*sizeof(float));
    }

    // fill holes
    libCommon::Utl::fillHole(fdmGrid.nx(), fdmGrid.ny(), fdmGrid.nz(), 0, data);


    if(gTrans){
        double x0 = fdmGrid.x0();
        double y0 = fdmGrid.y0();
        gTrans->localToWorld(x0, y0);
        fdmGrid.setx0(x0);
        fdmGrid.sety0(y0);
    }

    libCommon::FDM fdm;
    fdm.setData(data);
    fdm.setGrid(fdmGrid);
    fdm.save(fdmFile);

    return true;
}

