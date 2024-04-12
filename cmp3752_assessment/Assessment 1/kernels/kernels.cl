// Kernel to compute the intensity histogram from an image
kernel void create_intensity_histogram(global const uchar* input, global int* output) {
    const int GID = get_global_id(0);
    const int PIXEL_COUNT = get_global_size(0);

    // Indexes for the color channels
    int red_index = GID;
    int green_index = GID + PIXEL_COUNT;
    int blue_index = GID + PIXEL_COUNT * 2;

    // Increase the histogram counts atomically to avoid race conditions
    atomic_inc(&output[input[red_index]]);                  // Red channel histogram
    atomic_inc(&output[input[green_index] + 256]);          // Green channel histogram, offset by 256
    atomic_inc(&output[input[blue_index] + 512]);           // Blue channel histogram, offset by 512
}

// Kernel to compute a cumulative histogram using the Hillis-Steele algorithm
kernel void cumulate_histogram(global int* input, global int* output) {
    int gid = get_global_id(0);
    int bin_count = get_global_size(0);
    int total_channels = 3 * bin_count;  // Assuming three color channels (RGB)
    global int* temp_buffer;

    // Copy the input to output for the first iteration
    for (int i = 0; i < total_channels; i += bin_count) {
        output[gid + i] = input[gid + i];
    }

    barrier(CLK_GLOBAL_MEM_FENCE);  // Synchronize before starting the reduction

    for (int stride = 1; stride < bin_count; stride *= 2) {
        int previous_index = gid - stride;
        if (previous_index >= 0) {  // Check if the previous index is within bounds
            for (int i = 0; i < total_channels; i += bin_count) {
                output[gid + i] += input[previous_index + i];
            }
        }

        barrier(CLK_GLOBAL_MEM_FENCE);  // Ensure all writes are done before next iteration

        // Swap the pointers to swap roles of input and output
        temp_buffer = input;
        input = output;
        output = temp_buffer;
    }

    // Final swap to ensure the output contains the latest results if necessary
    if (bin_count % 2 == 1) {
        for (int i = 0; i < total_channels; i += bin_count) {
            output[gid + i] = input[gid + i];
        }
    }
}


// Kernel to compute a cumulative histogram using the Blelloch algorithm
kernel void cumulate_histogram_bl(global int* data) {
    int GID = get_global_id(0);
    int N = get_global_size(0);

    // Up-sweep phase
    for (int stride = 1; stride < N; stride *= 2) {
        int index = (GID + 1) * stride * 2 - 1;
        if (index < N)
            data[index] += data[index - stride];
        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    // Down-sweep phase
    if (GID == 0) data[N - 1] = 0; // Set the last element to zero
    barrier(CLK_GLOBAL_MEM_FENCE);

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

// Kernel to compute a cumulative histogram using atomic addition
kernel void cumulative_histogram_at(global int* data) {
    int GID = get_global_id(0);
    int N = get_global_size(0);

    for (int i = GID + 1; i < N; i++) {
        atomic_add(&data[i], data[GID]);
    }
}

// Kernel to map cumulative histogram to output image
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image) {
    const int GID = get_global_id(0);
    const int PIXEL_COUNT = get_global_size(0);
    int red_value = (int)(((float)histogram[input_image[GID]] / (float)PIXEL_COUNT) * 255);
    int green_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT] + 256] / (float)PIXEL_COUNT) * 255);
    int blue_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT * 2] + 512] / (float)PIXEL_COUNT) * 255);
    output_image[GID] = red_value;
    output_image[GID + PIXEL_COUNT] = green_value;
    output_image[GID + (2 * PIXEL_COUNT)] = blue_value;
}

