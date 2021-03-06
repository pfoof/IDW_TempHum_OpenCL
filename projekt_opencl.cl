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
                        output[y * ow + x] = input[v * iw + u];
                        break; break;
                    }
                    dist *= dist;
                    z += input[v * iw + u] / dist;
                    distsum += 1.0f/dist;
                }

            output[y * ow + x] = z / distsum;
        }
    
}

__kernel void colorize(
        __global float* values,
        __global uchar4* colors,
        __global unsigned int* size) {
    unsigned int w = get_global_size(0);
    unsigned int h = get_global_size(1);
    unsigned int image_w = size[0];
    unsigned int image_h = size[1];
    
    float edge = 0.5f;
    unsigned char r, gb;
    unsigned int x,y;
    for(x = get_global_id(0); x < image_w; x += w)
        for(y = get_global_id(1); y < image_h; y += h) {
            edge = smoothstep(-40.0f, 40.0f, values[y * image_w + x]);
            r = (unsigned char)(255.0f * edge);
            gb = (unsigned char)(255.0f * (1.0f-edge));
            colors[y * image_w + x] = (uchar4)(r, gb, gb, 255);
        }
}

__kernel void idw(__global float* input, __global float* output) {

}
