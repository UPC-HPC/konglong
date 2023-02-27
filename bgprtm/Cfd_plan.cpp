#include "Cfd_plan.h"


CfdPlan::CfdPlan(int inN, float dd) {
  n = inN;
  bm1 =  new float[5 * n];
  bl1 =  new float[4 * n];
  am1 =  new float[12 * 5];

  bm2 =  new float[5 * n];
  bl2 =  new float[4 * n];
  am2 =  new float[12 * 5];
  if(n > 1) {
    init_cfd(n, dd, bm1, bl1, am1, bm2, bl2, am2);
  }
}

CfdPlan::~CfdPlan() {
  delete [] bm1;
  delete [] bl1;
  delete [] am1;
  delete [] bm2;
  delete [] bl2;
  delete [] am2;
}
void CfdPlan::init_cfd(int nz, float dz, float *bmz1, float *blz1, float *amz1, float *bmz2, float *blz2, float *amz2) {
  float *bz = (float *)malloc(9 * nz * sizeof(float *));
  init_cfddev1(bz, blz1, amz1, nz, dz, 1);
  bandec(bz, nz, 4, blz1);
  reducematrix(bz, bmz1, nz, 9, 5);
  cfd_transpose(bmz1, nz, 5);
  cfd_transpose(blz1, nz, 4);

  init_cfddev1(bz, blz2, amz2, nz, dz, 2);
  bandec(bz, nz, 4, blz2);
  reducematrix(bz, bmz2, nz, 9, 5);
  cfd_transpose(bmz2, nz, 5);
  cfd_transpose(blz2, nz, 4);

  free(bz);
}


void CfdPlan::reducematrix(float *a, float *b, int n1, int n2a, int n2b) {
  if(n2b > n2a) {
    printf("error in reducematrix ! %d > %d \n", n2b, n2a);
    exit(-1);
  }
  for(int j = 0; j < n2b; ++j) {
    for(int i = 0; i < n1; ++i) {
      b[j * n1 + i] = a[j * n1 + i];
    }
  }

}
void CfdPlan::cfd_transpose(float *a, int n1, int n2) {
  float *work  = (float *) calloc(n1 * n2, sizeof(float));

  for(int j = 0; j < n2; ++j) {
    for(int i = 0; i < n1; ++i) {
      work[j + n2 * i] = a[i + j * n1];
    }
  }
  for(int i = 0; i < n1 * n2; ++i) {
    a[i] = work[i];
  }
  free(work);
}

void CfdPlan::bandec(float *a, int n, int m1, float *al) {
  int mm = 2 * m1 + 1;
  int l = m1;

  for(int i = 0; i < m1; ++i) {
    for(int j = m1 - i; j < mm; ++j) {
      //      a[j-l][i] = a[j][i];
      a[(j - l)*n + i] = a[j * n + i];
    }
    l = l - 1;
    for(int j = mm - l - 1; j < mm; ++j) {
      a[j * n + i] = 0.0;
    }
  }
  l = m1;
 for(int k = 1; k <= n; ++k) {
    float dum = a[k - 1];
    if(l < n) l++;
    if(dum == 0.0) a[k - 1] = 1.e-20;
    for(int i = k + 1; i <= l; ++i) {
      dum = a[i - 1] / a[k - 1];
      al[(i - k - 1)*n + k - 1] = dum;
      for(int j = 2; j <= mm; ++j) {
        a[(j - 2)*n + i - 1] = a[(j - 1) * n + i - 1] - dum * a[(j - 1) * n + k - 1];
      }
      a[(mm - 1)*n + i - 1] = 0.0;
    }
  }
}
void CfdPlan::init_cfddev1(float *b, float *bl, float *a, int n, float dx, int isgn) {

  float ca5[5], cb5[5];
  float ca[11], cb[9];

  //  10th order FD coefficients
  /*  ca5[0] = 0.890907049;
    ca5[1] = -0.311680973;
    ca5[2] = 0.10999693;
    ca5[3] = -0.0301858205;
    ca5[4] =  0.004671995;*/

  ca5[0] = 0.861085998189822;
  ca5[1] = 0.753782983214427;
  ca5[2] = 0.257072513289755;
  ca5[3] = 0.031326722562826;
  ca5[4] = 0.000410456919800;

  cb5[0] = 1.0;
  cb5[1] = 0.753205475977546;
  cb5[2] = 0.312203852849021;
  cb5[3] = 0.063580085296350;
  cb5[4] = 0.004597763362256;
 for(int j = 0; j < 4; ++j) {
    for(int i = 0; i < n; ++i) {
      bl[j * n + i] = 0.0;
    }
  }
  for(int j = 0; j < 9; ++j) {
    for(int i = 0; i < n; ++i) {
      b[j * n + i] = 0.0;
    }
  }

  for(int i = 0; i < 60; ++i) {
    a[i] = 0.0;
  }

  cb[0] = cb5[4];
  cb[1] = cb5[3];
  cb[2] = cb5[2];
  cb[3] = cb5[1];
  cb[4] = cb5[0];
  cb[5] = cb5[1];
  cb[6] = cb5[2];
  cb[7] = cb5[3];
  cb[8] = cb5[4];
if(isgn == 1) {
    //    b[4][0]=cb5[0];

    for(int i = 0; i < 4; ++i) {
      for(int j = 0; j < 9; ++j) {
        int j1 = i - 4 + j;
        if(j1 < 0) {
          int j2 = 4 - i - j1;
          b[j2 * n + i] = b[j2 * n + i] - cb[j];
        } else {
          b[j * n + i] = b[j * n + i] + cb[j];
        }
      }
    }
  } else {
    for(int i = 0; i < 4; ++i) {
      for(int j = 0; j < 9; ++j) {
        int j1 = i - 4 + j;
        if(j1 < 0) {
          int j2 = 4 - i - j1;
          b[j2 * n + i] = b[j2 * n + i] + cb[j];
        } else {
          b[j * n + i] = b[j * n + i] + cb[j];
        }
      }
    }

  }

  for(int i = 4; i < n - 4; ++i) {
    for(int j = 0; j < 9; ++j) {
      b[j * n + i] = cb[j];
    }
  }
 if(isgn == 1) {
    //    b[4][n-1]=cb5[0];

    for(int i = n - 4; i < n; ++i) {
      for(int j = 0; j < 9; ++j) {
        int j1 = i - 4 + j;
        if(j1 > n - 1) {
          int j2 = (n + 3 - i) - (j1 - n + 1);
          b[j2 * n + i] = b[j2 * n + i] - cb[j];

        } else {
          b[j * n + i] = b[j * n + i] + cb[j];
        }
      }
    }
  } else {
    for(int i = n - 4; i < n; ++i) {
      for(int j = 0; j < 9; ++j) {
        int j1 = i - 4 + j;
        if(j1 > n - 1) {
          int j2 = (n + 3 - i) - (j1 - n + 1);
          b[j2 * n + i] = b[j2 * n + i] + cb[j];

        } else {
          b[j * n + i] = b[j * n + i] + cb[j];
        }
      }
    }

  }

ca[0] = -ca5[4];
  ca[1] = -ca5[3];
  ca[2] = -ca5[2];
  ca[3] = -ca5[1];
  ca[4] = -ca5[0];
  ca[5] = 0.0;
  ca[6] = ca5[0];
  ca[7] = ca5[1];
  ca[8] = ca5[2];
  ca[9] = ca5[3];
  ca[10] = ca5[4];

  if(isgn == 1) {
    for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < 11; ++j) {
        int j1 = i - 5 + j;
        if(j1 < 0) {
          int j2 = 5 - i - j1;
          a[j2 * 5 + i] = a[j2 * 5 + i] + ca[j];
        } else {
          a[j * 5 + i] = a[j * 5 + i] + ca[j];
        }
      }
    }

  } else {
    for(int i = 0; i < 5; ++i) {
      for(int j = 0; j < 11; ++j) {
        int j1 = i - 5 + j;
        if(j1 < 0) {
          int j2 = 5 - i - j1;
          a[j2 * 5 + i] = a[j2 * 5 + i] - ca[j];
        } else {
          a[j * 5 + i] = a[j * 5 + i] + ca[j];
	          }
      }
    }

  }

  for(int i = 0; i < 5; ++i) {
    a[11 * 5 + i] = ca5[i];
  }


  for(int i = 0; i < 5; ++i) {
    for(int j = 0; j < 12; ++j) {
      a[j * 5 + i] = a[j * 5 + i] / dx * 0.5;
    }
  }

}

