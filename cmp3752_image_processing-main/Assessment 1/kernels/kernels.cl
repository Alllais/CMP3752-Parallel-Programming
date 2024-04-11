// Create histograms containing image intensities for each of the 3 colour channels.
// The output array is comprised of 3 contiguous intensity histograms
kernel void create_intensity_histogram(global const uchar* input, global int* output) {
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	int red_intensity = input[GID];
	int green_intensity = input[GID + PIXEL_COUNT];
	int blue_intensity = input[GID + PIXEL_COUNT*2];
	atomic_inc(&output[red_intensity]);
	atomic_inc(&output[green_intensity + 256]);
	atomic_inc(&output[blue_intensity + 256*2]);
}

// Creates a cumulative histogram from a non-cumulative histogram by employing a Hillis-Steele scan
kernel void cumulate_histogram(global int* input, global int* output) {
	int GID = get_global_id(0);
	int BIN_COUNT = get_global_size(0);
	global int* swap_buffer;
	for (int stride = 1; stride <= BIN_COUNT; stride *= 2) {
		output[GID]                 = input[GID];                  // R Channel
		output[GID + BIN_COUNT]     = input[GID + BIN_COUNT];     // G Channel
		output[GID + BIN_COUNT * 2] = input[GID + BIN_COUNT * 2]; // B Channel
		if (GID >= stride) {
			output[GID]                 += input[GID - stride];                   // R Channel
			output[GID + BIN_COUNT]     += input[GID + BIN_COUNT - stride];       // G Channel
			output[GID + BIN_COUNT * 2] += input[GID + (BIN_COUNT * 2) - stride]; // B Channel
		}

		barrier(CLK_GLOBAL_MEM_FENCE);

		swap_buffer = input;
		input = output;
		output = swap_buffer;
	}
}

// Normalize cumulative histogram
// Map image pixels to CDF intensities
kernel void map_cumulative_histogram_to_image(global const uchar* input_image, global const int* histogram, global uchar* output_image) {
	const int GID = get_global_id(0);
	const int PIXEL_COUNT = get_global_size(0);
	int red_value = (int)(((float)histogram[input_image[GID]] / (float)PIXEL_COUNT) * 255);
	int green_value = (int)(((float)histogram[input_image[GID+PIXEL_COUNT] + 256] / (float)PIXEL_COUNT) * 255);
	int blue_value = (int)(((float)histogram[input_image[GID + PIXEL_COUNT*2] + 512] / (float)PIXEL_COUNT) * 255);
	output_image[GID] = red_value;
	output_image[GID + PIXEL_COUNT] = green_value;
	output_image[GID + (2 * PIXEL_COUNT)] = blue_value;
}