#include "Vector3.h"
#include "Util.h"

float vector2::azimuth180() const {
  vector2 tmp = vnormal();
  float angle = PI_2_degree * acos(tmp.x);
  if(tmp.y < 0) angle = 180.f - angle;
  return angle;
}

float vector2::azimuth360() const {
  vector2 tmp = vnormal();
  float angle = PI_2_degree * acos(tmp.x);
  if(tmp.y < 0) angle = 360.f - angle;
  return angle;
}

void vector3::initMin() {
  x = FLT_MAX;
  y = FLT_MAX;
  z = FLT_MAX;
}

void vector3::initMax() {
  x = -FLT_MAX;
  y = -FLT_MAX;
  z = -FLT_MAX;
}

void vector3::updateMin(const vector3 &v) {
  x = std::min(v.x, x);
  y = std::min(v.y, y);
  z = std::min(v.z, z);
}

void vector3::updateMax(const vector3 &v) {
  x = std::max(v.x, x);
  y = std::max(v.y, y);
  z = std::max(v.z, z);
}


vector3 vector3::operator+(const vector3 &otherv) const {
  vector3 rev;
  rev.x  = x  + otherv.x;
  rev.y  = y  + otherv.y;
  rev.z  = z  + otherv.z;
  return rev;
}

vector3 vector3::operator-(const vector3 &otherv) const {
  vector3 rev;
  rev.x  = x  - otherv.x;
  rev.y  = y  - otherv.y;
  rev.z  = z  - otherv.z;
  return rev;
}

vector3 vector3::operator^(const vector3 &otherv) const {
  vector3 rev;
  rev.x  = y * otherv.z - z * otherv.y;
  rev.y  = -x * otherv.z + z * otherv.x;
  rev.z  = x * otherv.y - y * otherv.x;
  return rev;
}

vector3 vector3::vnormal() const {
  vector3 rev;
  float value = normal0();
  if(value > 0) value = 1. / value;
  rev.x  = x * value;
  rev.y  = y * value;
  rev.z  = z * value;
  return rev;
}

vector3 vector3::vnormal2() const {
  vector3 rev;
  rev.x  = x * x;
  rev.y  = y * y;
  rev.z  = z * z;
  float value = rev.x + rev.y + rev.z;
  float valuei = value ? (1.0 / value) : 0.0;
  return rev * valuei;
}


void vector3::snormal() {
  ;
  float value = normal0();
  if(value > 0) value = 1. / value;
  x  *= value;
  y  *= value;
  z  *= value;
}

vector3 vector3::rotatef(const vector3 &norm) const {
  vector3 rev;
  float cphi = norm.z;
  float sphi = sqrt(norm.x * norm.x + norm.y * norm.y);
  float cthe = 1;
  float sthe = 0;
  if(sphi != 0) {
    cthe = norm.x / sphi;
    sthe = norm.y / sphi;
  }
  rev.x = x * cphi * cthe + y * sphi * cthe + z * sthe;
  rev.y = -x * sphi      + y * cphi;
  rev.z = -x * cphi * sthe - y * sphi * sthe + z * cthe;
  return rev;
}

vector3 vector3::rotatei(const vector3 &norm) const {
  vector3 rev;
  float cphi = norm.z;
  float sphi = sqrt(norm.x * norm.x + norm.y * norm.y);
  float cthe = 1;
  float sthe = 0;
  if(sphi != 0) {
    cthe = norm.x / sphi;
    sthe = norm.y / sphi;
  }
  rev.x = x * cphi * cthe - y * sphi - z * cphi * sthe;
  rev.y = x * sphi * cthe + y * cphi - z * sphi * sthe;
  rev.z = x * sthe               + z * cthe;
  return rev;
}

vector3 vector3::rotatef(float cthe, float sthe, float cphi, float sphi) const {
  vector3 rev;
  rev.x = x * cphi * cthe + y * sphi * cthe + z * sthe;
  rev.y = -x * sphi      + y * cphi;
  rev.z = -x * cphi * sthe - y * sphi * sthe + z * cthe;
  return rev;
}

vector3 vector3::rotatei(float cthe, float sthe, float cphi, float sphi) const {
  vector3 rev;
  rev.x = x * cphi * cthe - y * sphi - z * cphi * sthe;
  rev.y = x * sphi * cthe + y * cphi - z * sphi * sthe;
  rev.z = x * sthe               + z * cthe;
  return rev;
}

void vector3::srotatef(float cthe, float sthe, float cphi, float sphi) {
  float myx = x * cphi * cthe + y * sphi * cthe + z * sthe;
  float myy = -x * sphi      + y * cphi;
  float myz = -x * cphi * sthe - y * sphi * sthe + z * cthe;
  x = myx;
  y = myy;
  z = myz;
}

void vector3::srotatei(float cthe, float sthe, float cphi, float sphi) {
  float myx = x * cphi * cthe - y * sphi - z * cphi * sthe;
  float myy = x * sphi * cthe + y * cphi - z * sphi * sthe;
  float myz = x * sthe               + z * cthe;
  x = myx;
  y = myy;
  z = myz;
}


void vector3::srotatef(float cthe, float sthe, float cphi, float sphi, float calpha, float salpha) {
  ;  // to be written
}

void vector3::srotatei(float cthe, float sthe, float cphi, float sphi, float calpha, float salpha) {
  ; // to be written
}


float vector3::dipangle() const {
  float length = this->normal0();
  float lh     = sqrt(x * x + y * y);
  float dangle = asin(lh / length) * 180. / 3.14159265;
  return dangle;
}

vector3 vector3::project(const vector3 &v01) const {
  vector3 normal = this->vnormal();
  float length = normal * v01;
  return normal * length;
}

void vector3::getaxis(float theta, float phi) {
  x = sin(theta) * cos(phi);
  y = sin(theta) * sin(phi);
  z = cos(theta);
}

