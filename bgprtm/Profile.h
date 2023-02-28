#ifndef PROFILE_H
#define PROFILE_H

class Fdm;

class Profile {
public:
  int             nz;
  float           dz;
  float           z0;
  float       velmin;
  float      maxfreq;
  float     *profile;
  Fdm         *myvel;
  Profile(Fdm *vel, float maxfreq0) {maxfreq = maxfreq0; myvel = vel; create();}
  ~Profile();
  void  getVprofile(float x, float y);
  void  getVprofile(int x, int y);
  void  MinVelForZGrid();
  float getValue(float z);
  float TimeGetDepth(float t);
  float getVmin(float z00, float zmax0);
  void create();
  float getMinValue();
  float getMinValue_backup();
  void MinVelSmooth();

private:
};


#endif

