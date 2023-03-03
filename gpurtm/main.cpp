#include <stdio.h>
#include <stdlib.h>
#include "Wavefield.h"
#include "gpu_kernel.h"
#include "cpu_kernel.h"

int main()
{
    int nx=48;
    int ny=32;
    int nz=108;

    int nt = 1477;
    int nts = 187;
    int it0=187;
    int it01=0;
    int it02= nts;
    int itinc=1;

    int it = it01;

    int ita =0;

    Wavefield *mywf = new Wavefield(nx,ny,nz);

    while(it!=it02)
    {
        int its = (it-it0);

        mywf->set_data();
        //run the kernel
        cpu_kernel(mywf);
        gpu_kernel(mywf);

        mywf->compare_host_dev();
        
        exit(0);
    }

    delete mywf;

	return 0;
}
