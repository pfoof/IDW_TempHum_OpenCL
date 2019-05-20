__kernel void idw2(__global float* input, __global unsigned int* input_size, __global float* output, __global unsigned int* output_size) {
    
    unsigned int w = get_global_size(0);
    unsigned int h = get_global_size(1);
    unsigned int ow = output_size[0];
    unsigned int oh = output_size[1];
    unsigned int iw = input_size[0];
    unsigned int ih = input_size[1];
    
    float ppp_v = (float)ow/(float)iw; //pixel_per_point vertical
    float ppp_h = (float)oh/(float)ih; //pixel_per_point horizontal
    float z = 0.0f, dist = 1.0f, distsum = 1.0f;
    
    for(int x = 0; x < ow; x += w)
        for(int y = 0; y < oh; y += h) {
            z = 0.0f;
            distsum = 0.0f;
            float2 pos = (float2)((float)x, (float)y);
            for(int u = 0; u < iw; ++u)
                for(int v = 0; v < ih; ++v) {
                    dist = distance( (float2)((float)u * ppp_v, (float)v * ppp_h), pos);
                    dist *= dist;
                    z += input[u * ih + v] / dist;
                    distsum += dist;
                }
            output[x * ow + y] = z / distsum;
        }
}

__kernel void idw(__global float* input, __global float* output) {

}