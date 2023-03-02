#include <stdio.h>
#include <stdlib.h>
#include "gpu_kernel.h"
#include "cpu_kernel.h"

int main()
{

    int nt = 1477;
    int nts = 187;
    int it0=187;
    int it01=0;
    int it02= nts;
    int itinc=1;

    int it = it01;

    int ita =0;

    while(it!=it02)
    {
        int its = (it-it0);

        //run the kernel
        cpu_kernel();
        gpu_kernel();
        exit(0);
    }

	return 0;
}
