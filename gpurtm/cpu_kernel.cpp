#include "cpu_kernel.h"
#include "Wavefield.h"

void cpu_kernel(const int nx, const int ny, const int nz)
{

    Wavefield *mywf = new Wavefield(nx,ny,nz);

    delete mywf;
    return;
}
