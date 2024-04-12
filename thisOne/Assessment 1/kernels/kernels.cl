// Kernel to compute the intensity histogram from an image
kernel void create_intensity_histogram(global const uchar* input, global int* output) {
    const int GID = get_global_id(0);
    const int PIXEL_COUNT = get_global_size(0);

    // Indexes for the color channels
    int red_index = GID;
    int green_index = GID + PIXEL_COUNT;
    int blue_index = GID + PIXEL_COUNT * 2;

    // Red channel histogram
    atomic_inc(&output[input[red_index]]);
    // Green channel histogram, offset by 256 to separate the histogram data
    atomic_inc(&output[input[green_index] + 256]);
    // Blue channel histogram, offset by 512 to separate the histogram data
    atomic_inc(&output[input[blue_index] + 512]);
}


// Kernel to compute a cumulative histogram using the Blelloch algorithm (exclusive scan)
kernel void cumulate_histogram_bl(global int* data) {
    int GID = get_global_id(0);
    int N = get_global_size(0);

    // Up-sweep (reduce) phase
    for (int stride = 1; stride < N; stride *= 2) {
        int index = (GID + 1) * stride * 2 - 1;
        if (index < N) {
            data[index] += data[index - stride];
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Clear the last element to initiate exclusive scan
    if (GID == 0) {
        data[N - 1] = 0;
    }
    barrier(CLK_GLOBAL_MEM_FENCE);

    // Down-sweep phase
    for (int stride = N / 2; stride > 0; stride /= 2) {
        int index = (GID + 1) * stride * 2 - 1;
        if (index < N) {
            int temp = data[index];
            data[index] += data[index - stride];
            data[index - stride] = temp;
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}


// Kernel to map cumulative histogram to output image
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image) {
    const int GID = get_global_id(0);
    const int PIXEL_COUNT = get_global_size(0);

    // Calculate new pixel values based on histogram equalization
    int red_value = (int)(((float)histogram[input_image[GID]] / (float)PIXEL_COUNT) * 255);
    int green_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT] + 256] / (float)PIXEL_COUNT) * 255);
    int blue_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT * 2] + 512] / (float)PIXEL_COUNT) * 255);

    // Write new values back to the output image
    output_image[GID] = (uchar)red_value;
    output_image[GID + PIXEL_COUNT] = (uchar)green_value;
    output_image[GID + (2 * PIXEL_COUNT)] = (uchar)blue_value;
}

