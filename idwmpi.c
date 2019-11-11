#include <stdio.h>
#include <openmpi/mpi.h>
#include <sys/sysinfo.h>

unsigned int processors = 1;

int main(int argc, char* argv[]) {
    processors = get_nprocs();
    MPI_Init(argc, argv);

    

    MPI_Finalize();
    return 0;
}