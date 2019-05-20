__kernel void idw2(__global float* input, __global unsigned int* input_size, __global float* output, __global unsigned int* output_size) {
    
    unsigned int w = get_global_size(0);
    unsigned int h = get_global_size(1);
    unsigned int ow = output_size[0];
    unsigned int oh = output_size[1];
    unsigned int iw = input_size[0];
    unsigned int ih = input_size[1];
    
    float ppp_v = (float)ow / (float)iw; //pixel_per_point vertical
    float ppp_h = (float)oh / (float)ih; //pixel_per_point horizontal
    float z = 0.0f, dist = 1.0f, distsum = 1.0f;
    float2 pos = (float2)(0.0f, 0.0f);
    
    for(int x = get_global_id(0); x < ow; x += w)
        for(int y = get_global_id(1); y < oh; y += h) {
            
            z = 0.0f;
            distsum = 0.0f;
            pos = (float2)((float)x, (float)y);
            
            for(int u = 0; u < iw; ++u)
                for(int v = 0; v < ih; ++v) {
                    dist = distance( (float2)((float)u * ppp_v, (float)v * ppp_h), pos);
                    if(dist < 0.001f) {
                        output[x * ow + y] = input[u * ih + v];
                        break; break;
                    }
                    dist *= dist;
                    z += input[u * ih + v] / dist;
                    distsum += 1.0f/dist;
                }

            output[x * ow + y] = z / distsum;
        }
    
}

__kernel void idw(__global float* input, __global float* output) {

}