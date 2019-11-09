#define PROGRAM_FILE "projekt_opencl.cl"
#define KERNEL_FUNC "idw2"
#define COLOR_FUNC "colorize"

#include <CL/cl.h>
#include <stdio.h>

#include "input.h"

const int output_size[2] = { 1024, 1024 };
const int global_size[2] = { 16, 16 };

cl_device_id create_device() {

   cl_platform_id platform;
   cl_device_id dev;
   int err;

   /* Identify a platform */
   err = clGetPlatformIDs(1, &platform, NULL);
   if(err < 0) {
      perror("Couldn't identify a platform");
      exit(1);
   } 

   // Access a device
   // GPU
   err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
   if(err == CL_DEVICE_NOT_FOUND) {
      // CPU
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
   }
   if(err < 0) {
      perror("Couldn't access any devices");
      exit(1);   
   }

   return dev;
}

cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

   cl_program program;
   FILE *program_handle;
   char *program_buffer, *program_log;
   size_t program_size, log_size;
   int err;

   /* Read program file and place content into buffer */
   program_handle = fopen(filename, "r");
   if(program_handle == NULL) {
      perror("Couldn't find the program file");
      exit(1);
   }
   fseek(program_handle, 0, SEEK_END);
   program_size = ftell(program_handle);
   rewind(program_handle);
   program_buffer = (char*)malloc(program_size + 1);
   program_buffer[program_size] = '\0';
   fread(program_buffer, sizeof(char), program_size, program_handle);
   fclose(program_handle);

   /* Create program from file 
   Creates a program from the source code in the add_numbers.cl file. 
   Specifically, the code reads the file's content into a char array 
   called program_buffer, and then calls clCreateProgramWithSource.
   */
   program = clCreateProgramWithSource(ctx, 1, 
      (const char**)&program_buffer, &program_size, &err);
   if(err < 0) {
      perror("Couldn't create the program");
      exit(1);
   }
   free(program_buffer);

   /* Build program 
   The fourth parameter accepts options that configure the compilation. 
   These are similar to the flags used by gcc. For example, you can 
   define a macro with the option -DMACRO=VALUE and turn off optimization 
   with -cl-opt-disable.
   */
   err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
   if(err < 0) {

      /* Find size of log and print to std output */
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            0, NULL, &log_size);
      program_log = (char*) malloc(log_size + 1);
      program_log[log_size] = '\0';
      clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 
            log_size + 1, program_log, NULL);
      printf("%s\n", program_log);
      free(program_log);
      exit(1);
   }

   return program;
}

int main() {


   /* OpenCL structures */
   cl_device_id device;
   cl_context context;
   cl_program program;
   cl_kernel kernel_idw, kernel_colorize;
   cl_command_queue queue;
   cl_int err;

   device = create_device();
   context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
   if(err < 0) {
      perror("Couldn't create a context");
      exit(1);   
   }

   /* Build program */
   program = build_program(context, device, PROGRAM_FILE);

   queue = clCreateCommandQueue(context, device, 0, &err);
   if(err < 0) {
      perror("Couldn't create a command queue");
      exit(1);   
   };

   /* Create kernels */
   kernel_idw = clCreateKernel(program, KERNEL_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel idw");
      exit(1);
   };
   kernel_colorize = clCreateKernel(program, COLOR_FUNC, &err);
   if(err < 0) {
      perror("Couldn't create a kernel colorize");
      exit(1);
   };

    float* output = (float*)malloc(output_size[0]*output_size[1]*sizeof(float));
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, output_size[0]*output_size[1]*sizeof(float), output, &err);
   if(err < 0) {
      perror("Error creating buffer 1");
      exit(1);
   };
    cl_mem output_size_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * sizeof(int), output_size, &err);
    if(err < 0) {
      perror("Error creating buffer 2");
      exit(1);
   };

    //Verify this!
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, input_size[0]*input_size[1]*sizeof(float), input, &err);
   if(err < 0) {
      perror("Error creating buffer 3");
      exit(1);
   };
    cl_mem input_size_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * sizeof(int), input_size, &err);
   if(err < 0) {
      perror("Error creating buffer 4");
      exit(1);
   };
    
  /* Create kernel arguments */
   err = clSetKernelArg(kernel_idw, 0, sizeof(cl_mem), &input_buffer);
   err |= clSetKernelArg(kernel_idw, 1, sizeof(cl_mem), &input_size_buffer);
   err |= clSetKernelArg(kernel_idw, 2, sizeof(cl_mem), &output_buffer);
   err |= clSetKernelArg(kernel_idw, 3, sizeof(cl_mem), &output_size_buffer);
   if(err < 0) {
      perror("Couldn't create a kernel argument");
      exit(1);
   }

   err = clEnqueueNDRangeKernel(queue, kernel_idw, 1, NULL, &global_size, 
         NULL, 0, NULL, NULL); 
   if(err < 0) {
      perror("Couldn't enqueue the kernel");
      exit(1);
   }

   /* Read the kernel's output    */
   err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, 
         output_size[0]*output_size[1]*sizeof(float), output, 0, NULL, NULL);
   if(err < 0) {
      perror("Couldn't read the buffer");
      exit(1);
   }

    FILE* output_file;
    output_file = fopen("out.csv", "w");
    int i,j;
    for(i = 0; i < output_size[1]; ++i) {
        for(j = 0; j < output_size[0]; ++j) {
            fprintf(output_file, "%.3f, ", output[i * output_size[0] + j]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);

   clReleaseMemObject(output_buffer);
   clReleaseMemObject(input_buffer);
   clReleaseMemObject(output_size_buffer);
   clReleaseMemObject(input_size_buffer);
   clReleaseKernel(kernel_idw);
   clReleaseKernel(kernel_colorize);
   clReleaseCommandQueue(queue);
   clReleaseProgram(program);
   clReleaseContext(context);
   return 0;
}
