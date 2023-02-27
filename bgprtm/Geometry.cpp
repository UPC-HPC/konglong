#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <iostream>
#include <fstream>

#include "Geometry.h"
#include "Grid.h"

Geometry::Geometry() {
  header = NULL;
}

Geometry::~Geometry() {
  if(header) {
    free(header);
    header = NULL;
  };
}
void Geometry::read(const char *fileName) {
  std::ifstream ifs(fileName, std::ios::binary);

  if(!ifs.is_open()) {
    char cwd[FILENAME_MAX + 1];
    getcwd(cwd, FILENAME_MAX);
    std::cout << "ERROR: Can't read the file " << fileName << "(cwd: " << cwd << ")" << std::endl;
    exit(-1);
  }
  if(!header) header = (GeomHeader*)malloc(sizeof(GeomHeader));
  ifs.read((char*)header, sizeof(GeomHeader));

  if(header->gridType != RECTANGLE) {
    if(zgrid.size() < (size_t)header->nz) zgrid.resize(header->nz);
    ifs.read((char*)&zgrid[0], sizeof(float) * header->nz);

    if(dzgrid.size() < (size_t)header->nz) dzgrid.resize(header->nz);
    ifs.read((char*)&dzgrid[0], sizeof(float) * header->nz);
  }
  ifs.close();
}
void Geometry::write(const char *fileName) {
  std::ofstream ofs(fileName, std::ios::binary);

  if(!ofs.is_open()) {
    char cwd[FILENAME_MAX + 1];
    getcwd(cwd, FILENAME_MAX);
    std::cout << "ERROR: Can't write the file " << fileName << "(cwd: " << cwd << ")" << std::endl;
    exit(-1);
  }
  ofs.write((char*)header, sizeof(GeomHeader));

  if(header->gridType != RECTANGLE) {
    ofs.write((char*)&zgrid[0], sizeof(float) * header->nz);
    ofs.write((char*)&dzgrid[0], sizeof(float) * header->nz);
  }
  ofs.close();
}

void Geometry::setHeader(float x0, float y0, float z0, int nx, int ny, int nz, float dx, float dy, float dz, int nzbnd, int nzuppad,
    int nxbnd, int nybnd, int gridType) {
  if(!header) header = (GeomHeader*)malloc(sizeof(GeomHeader));

  header->x0 = x0;
  header->y0 = y0;
  header->z0 = z0;

  header->nx = nx;
  header->ny = ny;
  header->nz = nz;

  header->dx = dx;
  header->dy = dy;
  header->dz = dz;
  header->nzbnd = nzbnd;
  header->nzuppad = nzuppad;
  header->nxbnd = nxbnd;
  header->nybnd = nybnd;
  header->gridType = gridType;
}

void Geometry::setZGrid(vector<float> &zgrid) {
  setZGrid(&zgrid[0]);
}

void Geometry::setDzGrid(vector<float> &dzgrid) {
  setDzGrid(&dzgrid[0]);
}

void Geometry::setZGrid(float *zgrid0) {
  if(zgrid.size() < (size_t)header->nz) zgrid.resize(header->nz);
  memcpy(&zgrid[0], zgrid0, sizeof(float) * header->nz);
}

void Geometry::setDzGrid(float *dzgrid0) {
  if(dzgrid.size() < (size_t)header->nz) dzgrid.resize(header->nz);
  memcpy(&dzgrid[0], dzgrid0, sizeof(float) * header->nz);
}

void Geometry::printHeader() {
  printf("Geometry info:\n");
  if(header->gridType == RECTANGLE) printf("          gridType: Rectangle \n");
  else if(header->gridType == IRREGULAR) printf("          gridType: IrregularZ \n");
  else if(header->gridType == XPYRAMID) printf("          gridType: XPYRAMID \n");
  else if(header->gridType == YPYRAMID) printf("          gridType: YPYRAMID \n");
  else if(header->gridType == XYPYRAMID) printf("          gridType: XYPYRAMID \n");
  else printf("          unknown gridType!\n");

  printf("          orig: (x0,y0,z0) = %f,%f,%f\n", header->x0, header->y0, header->z0);
  printf("          step: (dx,dy,dz) = %f,%f,%f\n", header->dx, header->dy, header->dz);
  printf("          size: (nx,ny,nz) = %d,%d,%d\n", header->nx, header->ny, header->nz);
  printf("          boundary(nzbnd,nzuppad): = %d,%d\n", header->nzbnd, header->nzuppad);
  fflush(stdout);
}

void Geometry::printTable() {
  if(header->gridType != RECTANGLE) {
    printf("zgrid table:\n");
    for(int i = 0; i < header->nz; i++)
      printf("iz=%4d, zgrid=%5.4f dz=%5.4f\n", i, zgrid[i], dzgrid[i]);
  } else {
    printf("It is rectangle grid. Unable to print the table. \n");
  }

  fflush(stdout);
}

void Geometry::read(string &path) {
  read(path.c_str());
}

void Geometry::write(string &path) {
  write(path.c_str());
}

