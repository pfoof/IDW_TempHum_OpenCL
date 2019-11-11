#include <omp.h>
#include "input.h"
#include <math.h>
#include <stdio.h>

float distance(float x1, float y1, float x2, float y2) {
    return sqrt( (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2) );
}

void idw2(float* _input, unsigned int* _input_size, float* output, int* output_size) {
       unsigned int iw = _input_size[0];
       unsigned int ih = _input_size[1];
       unsigned int ow = output_size[0];
       unsigned int oh = output_size[1];
       float ppp_v = (float)ow / (float)iw; //pixel_per_point vertical
       float ppp_h = (float)oh / (float)ih; //pixel_per_point horizontal
       float z = 0.0f, dist = 1.0f, distsum = 1.0f;
       float pos_x = 0.0f; float pos_y = 0.0f;
    #pragma omp parallel for firstprivate(z, dist, distsum, pos_x, pos_y)
        for(int x = 0; x < ow; ++x)
            for(int y = 0; y < oh; ++y) {
                z = 0.0f;
                distsum = 0.0f;
                pos_x = (float)x; pos_y = (float)y;
                
                for(int u = 0; u < iw; ++u)
                    for(int v = 0; v < ih; ++v) {
                        dist = distance(u * ppp_v, v * ppp_h, pos_x, pos_y);
                        if(dist < 0.001f) {
                            output[y * ow + x] = input[v * iw + u];
                            break; break;
                        } //end if dist
                        dist *= dist;
                        z += input[v * iw + u] / dist;
                        distsum += 1.0f/dist;
                    } //end for v

                    output[y * ow + x] = z / distsum;
            } //end for y
} //end void

int main(int argc, char* argv[]) {

    if(argc < 3) {
        printf("Usage %s <output_w> <output_h>\n", argv[0]);
        exit(2);
    }

    const int output_size[2] = { atoi(argv[1]), atoi(argv[2]) };
    float* output = (float*)malloc( output_size[0] * output_size[1] * sizeof(float) );
    
    idw2(input, input_size, output, output_size);

    FILE* output_file;
    output_file = fopen("out_omp.csv", "w");
    int i,j;
    for(i = 0; i < output_size[1]; ++i) {
        for(j = 0; j < output_size[0]; ++j) {
            fprintf(output_file, "%.3f, ", output[i * output_size[0] + j]);
        }
        fprintf(output_file, "\n");
    }
    fclose(output_file);

}
