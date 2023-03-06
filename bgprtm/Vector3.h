#ifndef VECTOR3_H
#define VECTOR3_H

#include <math.h>
#include <float.h>

class vector2 {
public:
  float x;
  union {
    float y;
    float z;
  };
  vector2()
    : x(0.0f), z(0.0f)
  {}
  vector2(float x_, float z_)
    : x(x_), z(z_)
  {}

  vector2 operator+(const vector2 &v01) const {vector2 rev; rev.x = x + v01.x; rev.z = z + v01.z; return rev;};
  vector2 operator-(const vector2 &v01) const {vector2 rev; rev.x = x - v01.x; rev.z = z - v01.z; return rev;};
  float   operator^(const vector2 &v01) const {return x * v01.z - z * v01.x;};
  float   operator*(const vector2 &v01) const {return x * v01.x + z * v01.z;};
  vector2 operator*(float   v01) const {vector2 rev; rev.x = x * v01; rev.z = z * v01; return rev;};
  vector2 operator*(double  v01) const {vector2 rev; rev.x = x * v01; rev.z = z * v01; return rev;};
  vector2 operator/(float   v01) const {vector2 rev; rev.x = x / v01; rev.z = z / v01; return rev;};
  vector2 operator/(double  v01) const {vector2 rev; rev.x = x / v01; rev.z = z / v01; return rev;};
  //void    operator=(const vector2 &v1) {x=v1.x;z=v1.z;};
  void    operator+=(const vector2 &v1) {x += v1.x; z += v1.z;};
  void    operator-=(const vector2 &v1) {x -= v1.x; z -= v1.z;};
  void    operator*=(const float sl) {x *= sl; z *= sl;};
  void    operator*=(const double sl) {x *= sl; z *= sl;};
  void    operator/=(const float sl) {x /= sl; z /= sl;};
  void    operator/=(const double sl) {x /= sl; z /= sl;};
  void    setvec(float a, float b) {x = a; z = b;};
  float   normal0() const {return sqrtf(x * x + z * z);};
  void    snormal() {float vv = normal0(); if(vv) vv = 1.0 / vv; x *= vv; z *= vv;};
  float   azimuth180() const;
  float   azimuth360() const;
  vector2 vnormal() const {vector2 rev; rev.setvec(0.0f, 0.0f); if(normal0()) rev = (*this) / normal0(); return rev;};
  vector2 rotatef(float c, float s) const {vector2 r; r.x = c * x + s * z; r.z = -s * x + c * z; return r;};
  vector2 rotatei(float c, float s) const {vector2 r; r.x = c * x - s * z; r.z = s * x + c * z; return r;};
  void    srotatef(float c, float s) {float t; t = c * x + s * z; z = -s * x + c * z; x = t;};
  void    srotatei(float c, float s) {float t; t = c * x - s * z; z = s * x + c * z; x = t;};
  vector2 rotatef(float the) const {float s = sin(the); float c = cos(the); return rotatef(c, s);};
  vector2 rotatei(float the) const {float s = sin(the); float c = cos(the); return rotatei(c, s);};
  void    srotatef(float the) {float s = sin(the); float c = cos(the); this->srotatef(c, s);};
  void    srotatei(float the) {float s = sin(the); float c = cos(the); this->srotatei(c, s);};
};

class vector3 {
public:
  float  x,  y,  z;
  vector3()
    : x(0.0f), y(0.0f), z(0.0f)
  {}
  vector3(const float x_, float y_, const float z_)
    : x(x_), y(y_), z(z_)
  {}
  void initMin();
  void initMax();
  void updateMin(const vector3 &v);
  void updateMax(const vector3 &v);

  vector3 operator+(const vector3 &v01) const;
  vector3 operator-(const vector3 &v01) const;
  vector3 operator^(const vector3 &v01) const;
  float   operator*(const vector3 &v01) const {return x * v01.x + y * v01.y + z * v01.z;};
  vector3 operator*(float scale) const {return vector3(x * scale, y * scale, z * scale);};
  vector3 operator*(double scale) const {return vector3((float)(x * scale), (float)(y * scale), (float)(z * scale));};
  //void    operator=(const vector3 &v1) {x=v1.x;y=v1.y;z=v1.z;};
  void    operator+=(const vector3 &v1) {x += v1.x; y += v1.y; z += v1.z;};
  void    operator-=(const vector3 &v1) {x -= v1.x; y -= v1.y; z -= v1.z;};
  void    operator*=(const float sl) {x *= sl; y *= sl; z *= sl;}
  void    operator*=(const double sl) {x *= sl; y *= sl; z *= sl;}
  void    operator/=(const float sl) {x /= sl; y /= sl; z /= sl;};
  void    operator/=(const double sl) {x /= sl; y /= sl; z /= sl;};
  vector3 operator+(const vector2 &v01) const {vector3 rev; rev.x = x + v01.x; rev.y = y + v01.y; rev.z = z; return rev;};
  vector3 operator-(const vector2 &v01) const {vector3 rev; rev.x = x - v01.x; rev.y = y - v01.y; rev.z = z; return rev;};
  vector3 reverse() const {return vector3(-x, -y, -z);};
  vector3 project(const vector3 &v01) const;
  vector3 nproject(const vector3 &v01) const {return ((*this) * ((*this) * v01));};
  void    sreverse() {x = -x, y = -y, z = -z;};
  vector3 vnormal() const;
  vector2 get2dvec() const {vector2 rev; rev.x = x, rev.y = y; return rev;};
  void    snormal();
  vector3 vnormal2() const;
  float   normal0() const {return sqrtf(x * x + y * y + z * z);};
  float   mod() const {return (x * x + y * y + z * z);};
  float   imod() const {return 1.f / (x * x + y * y + z * z);};
  float   h2() const {return (x * x + y * y);};
  void    setvec(float a, float b, float c) {x = a; y = b; z = c;};
  void    setvec(vector2 x2, float c) {x = x2.x; y = x2.y; z = c;};
  vector3 rotatef(float cthe, float sthe, float cphi, float sphi) const;
  vector3 rotatei(float cthe, float sthe, float cphi, float sphi) const;
  void    srotatef(float cthe, float sthe, float cphi, float sphi);
  void    srotatei(float cthe, float sthe, float cphi, float sphi);
  void    srotatef(float cthe, float sthe, float cphi, float sphi, float calpha, float salpha);
  void    srotatei(float cthe, float sthe, float cphi, float sphi, float calpha, float salpha);
  vector3 rotatef(const vector3 &norm) const;
  vector3 rotatei(const vector3 &norm) const;
  void    getaxis(float theta, float phi);
  float   dipangle() const;
};
class vector6 {
public:
  vector3  x,  p;
  vector6()
    : x(), p()
  {}
  vector6(const vector3 x_, const vector3 p_)
    : x(x_), p(p_)
  {}
#if 0
  vector6(const vector6 &that)
    : x(that.x), p(that.p)
  {}
  vector6 operator=(const vector6 &that)
  { x = that.x; p = that.p; return *this; }
#endif

  vector6 operator+(const vector6 &v01) const {return vector6(x + v01.x, p + v01.p);};
  vector6 operator-(const vector6 &v01) const {return vector6(x - v01.x, p - v01.p);};
  vector6 operator*(float   v01) const {vector6 rev; rev.x = x * v01; rev.p = p * v01; return rev;};
  vector6 operator*(double  v01) const {vector6 rev; rev.x = x * v01; rev.p = p * v01; return rev;};
  void    operator=(const vector6 &v1) {x = v1.x; p = v1.p;};
  void    operator+=(const vector6 &v1) {x += v1.x; p += v1.p;};
  void    operator-=(const vector6 &v1) {x -= v1.x; p -= v1.p;};
  void    operator*=(const float sl) {x *= sl; p *= sl;};
  void    operator*=(const double sl) {x *= sl; p *= sl;}
  void    setvec(const vector3 &x0, const vector3 &p0) {x = x0; p = p0;};
};

#endif

