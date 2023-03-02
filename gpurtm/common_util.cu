#include "common_util.h"
void set_data_3d(float *data,
                 const int nx, const int ny, const int nz,
                 const int pattern_type)
{


    int numElem = nx*ny*nz;
    printf("NumElem = %d\n",numElem);

    /*
    float *h_data = NULL;

    if(pointer_type==HOST) 
        h_data = d_data;
    else          
        h_data= (float*) malloc(sizeof(float)*numElem);
*/
    for(int iz=0;iz<nz;iz++)
    {
        for(int iy=0;iy<ny;iy++)
        {
            for(int ix=0;ix<nx;ix++)
            {
                float value ;
                if(pattern_type==1)
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

    /*
    if(pointer_type==DEVICE)
    { 
        cudaMemcpy(d_data, h_data, sizeof(float)*numElem, cudaMemcpyHostToDevice);
        free(h_data);
    }
*/
    return;
}

