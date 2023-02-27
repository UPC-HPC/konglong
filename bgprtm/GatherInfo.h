/*
 * ImaingCondition.h
 *
 */

#ifndef GATHERINFO_H_
#define GATHERINFO_H_

#include <string>
#include <iostream>
#include <fstream>


struct GatherInfo {
public:
  int nz, nt, nx, ny;
  float dz, dt, dx, dy;
  float z0, t0, x0, y0;
public:
  GatherInfo(): nz(0), nt(1), nx(0), ny(0), dz(0), dt(0), dx(0), dy(0), z0(0), t0(0), x0(0), y0(0) {}; // default nt 1 in case of failing to load file
  GatherInfo(int innz, int innt, int innx, int inny, float inz0, float int0, float inx0, float iny0,
             float indz, float indt, float indx, float indy)
    : nz(innz), nt(innt), nx(innx), ny(inny), dz(indz), dt(indt), dx(indx), dy(indy), z0(inz0), t0(int0), x0(inx0), y0(iny0) {};

  void save(std::string fileName)const {
    std::ofstream ofs(fileName.c_str());
    ofs << "nz: " << nz << " nt: " << nt << " nx: " << nx << " ny: " << ny << std::endl;
    ofs << "z0: " << z0 << " t0: " << t0 << " x0: " << x0 << " y0: " << y0 << std::endl;
    ofs << "dz: " << dz << " dt: " << dt << " dx: " << dx << " dy: " << dy << std::endl;
    ofs.close();
  }
  void read(std::string fileName) {
    char tmp[128];
    std::ifstream ifs(fileName.c_str());
    ifs >> tmp >> nz >> tmp >> nt >> tmp >> nx >> tmp >> ny;
    ifs >> tmp >> z0 >> tmp >> t0 >> tmp >> x0 >> tmp >> y0;
    ifs >> tmp >> dz >> tmp >> dt >> tmp >> dx >> tmp >> dy;
    ifs.close();
  }
  void print(std::string space = "", std::ostream &os = std::cout)const {
    os << space << "-------------- Gather Information ---------------------" << std::endl;
    os << space << "nz: " << nz << " nt: " << nt << " nx: " << nx << " ny: " << ny << std::endl;
    os << space << "z0: " << z0 << " t0: " << t0 << " x0: " << x0 << " y0: " << y0 << std::endl;
    os << space << "dz: " << dz << " dt: " << dt << " dx: " << dx << " dy: " << dy << std::endl;
    os << std::endl << std::endl << std::flush;
  }

};


#endif /* GATHERINFO_H_ */

