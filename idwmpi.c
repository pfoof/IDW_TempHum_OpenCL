#include <stdio.h>
#include <openmpi/mpi.h>
#include <sys/sysinfo.h>
#include "input.h"

#define TAG_NPROC 100

unsigned int processors = 1;

void idw2(float* _input, int* _input_size, float* output, int* output_size) {

}

int main(int argc, char* argv[]) {
    processors = get_nprocs();
    MPI_Init(&argc, &argv);

    int tid, nthreads;
    MPI_Comm_rank(MPI_COMM_WORLD, &tid);

    if(tid == 0) {
        
        //Main Thread
        MPI_Comm_size(MPI_COMM_WORLD, &nthreads);
        
        printf("Getting processor counts...\n");
        int cores[nthreads], allcores;
        cores[0] = processors; allcores = processors;
        int i;
        for(i = 0; i < nthreads - 1; ++i) {
            int r; MPI_Status recvStatus;
            MPI_Recv(&r, 1, MPI_INT, MPI_ANY_SOURCE, TAG_NPROC, MPI_COMM_WORLD, &recvStatus);
            cores[recvStatus.MPI_SOURCE] = r;
            allcores += r;
        }
        printf("Total of %d cores in %d processes\n", allcores, nthreads);
        


        //end tid == 0
    } else {
        
        //Other threads
        MPI_Send(&processors, 1, MPI_INT, 0, TAG_NPROC, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}