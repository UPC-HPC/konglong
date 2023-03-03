#include "common_util.h"
void set_data_3d(float *data,
                 const int nx, const int ny, const int nz,
                 const int pattern_type)
{


    int numElem = nx*ny*nz;
    printf("NumElem = %d\n",numElem);

    for(int iz=0;iz<nz;iz++)
    {
        for(int iy=0;iy<ny;iy++)
        {
            for(int ix=0;ix<nx;ix++)
            {
                float value ;
                if(pattern_type==0)
                    value = 1.f;
                else if(pattern_type==1)
                    value = ix*1.f;
                else if(pattern_type==2)
                    value = iy*1.f;
                else if(pattern_type==3)
                    value = iz*1.f;
                else if(pattern_type==4)
                    value = 0.2*iz+0.3*iy+0.5*ix;
                else if(pattern_type==5)
                    value = 0.2*(nz-iz-1)+0.3*(ny-iy-1)+0.5*(nx-ix-1);
                else if(pattern_type==6)
                    value = 0.1*(iz%32)*(iz%32)+0.3*(iy%32)*(iy%32)+0.5*(ix%32)*(ix%32);
                data[ix+iy*nx+iz*nx*ny]=value;
            }
        }
    }

    return;
}



int compare_data_3d(float *h_data, float*d_data,
                 const int nx, const int ny, const int nz,
                 float threshold, int nerr_threshold)
{

    int numElem = nx*ny*nz;

    float *d2h_data = (float*) malloc(sizeof(float)*numElem);
    cudaMemcpy(d2h_data, d_data, sizeof(float)*numElem, cudaMemcpyDefault);

    int i=0;
    int nerr=0;
    for(int iz=0;iz<nz;iz++)
    {
        for(int iy=0;iy<ny;iy++)
        {
            for(int ix=0;ix<nx;ix++)
            {
                if( ((fabs(h_data[i])<1.E-12)&&(fabs(d2h_data[i])>2.E-12))
                    || ((fabs(h_data[i])>=1.E-12)&&(fabs((h_data[i]-d2h_data[i])/h_data[i])>threshold)) )
                {
                    printf("Mismatch at %d %d %d: %8.6E,%8.6E\n",ix,iy,iz,h_data[i],d2h_data[i]);
                    nerr++;   
                }    
                if(nerr>=nerr_threshold) return 1;
                i++;
            }
        }
    }

    free(d2h_data);
    return 0;
    /*
    if(pointer_type==DEVICE)
    { 
        cudaMemcpy(d_data, h_data, sizeof(float)*numElem, cudaMemcpyHostToDevice);
    }
*/
}



