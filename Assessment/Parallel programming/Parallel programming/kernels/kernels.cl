kernel void histogram_kernel(__global const uchar* image, __global int* histogram, const int imageSize) {
    int id = get_global_id(0);
    if (id < imageSize) {
        atomic_inc(&histogram[image[id]]);
    }
}

kernel void cumulative_histogram_kernel(__global int* histogram, __global int* cumHistogram, const int numBins) {
    int id = get_global_id(0);
    if (id == 0) {
        cumHistogram[id] = histogram[id];
    }
    else {
        int cumSum = 0;
        for (int i = 0; i <= id; i++){
            cumSum += histogram[i];
    }
        cumHistogram[id] = cumSum;
}

kernel void normalise_cumulative_histogram_kernel(__global int* cumHistogram, __global uchar * LUT, const int numPixels) {
        int id = get_global_id(0);
        LUT[id] = (uchar)(255.0f * cumHistogram[id] / numPixels);
    }

kernel void apply_lut_kernel(__global uchar* image, __global const uchar LUT, const int imageSize) {
    int id = get_global_id(0);
    if (id < imageSize) {
        image[id] = LUT[image[id]];
    }
}