#include <iostream>
#include <vector>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

void print_help() {
    cerr << "Application usage:" << endl
        << "  -p : select platform " << endl
        << "  -d : select device" << endl
        << "  -l : list all platforms and devices" << endl
        << "  -f : input image file (default: test.ppm)" << endl
        << "  -h : print this message" << endl;
}

vector<int> create_intensity_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, CImg<unsigned char> img) {
    const int BIN_COUNT = 256 * 3; // Three channels each with 256 bins
    vector<int> histogram(BIN_COUNT, 0);
    size_t img_size = img.size(); // Total number of color components
    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, img_size);
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * histogram.size());
    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, img_size, img.data());

    cl::Kernel kernel(program, "create_intensity_histogram");
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(img_size / 3));
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data());

    return histogram;
}

vector<int> cumulate_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, vector<int> histogram) {
    const size_t hist_size = histogram.size(); // Size of the histogram
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(int) * hist_size);
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(int) * hist_size, histogram.data());

    cl::Kernel kernel(program, "cumulate_histogram_bl");
    kernel.setArg(0, buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(hist_size));
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int) * hist_size, histogram.data());

    return histogram;
}

CImg<unsigned char> map_histogram_to_image(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, CImg<unsigned char> img, vector<int> histogram) {
    size_t img_size = img.size();
    cl::Buffer img_buffer(context, CL_MEM_READ_ONLY, img_size);
    cl::Buffer hist_buffer(context, CL_MEM_READ_ONLY, sizeof(int) * histogram.size());
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, img_size);
    queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, img_size, img.data());
    queue.enqueueWriteBuffer(hist_buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data());

    cl::Kernel kernel(program, "map_cumulative_histogram_to_image");
    kernel.setArg(0, img_buffer);
    kernel.setArg(1, hist_buffer);
    kernel.setArg(2, output_buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(img_size / 3));

    vector<unsigned char> output_data(img.size());
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, img_size, output_data.data());

    return CImg<unsigned char>(output_data.data(), img.width(), img.height(), img.depth(), img.spectrum());
}

int main(int argc, char** argv) {
    string image_filename = "test.ppm";
    int platform_id = 0, device_id = 0;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-p" && i < argc - 1) platform_id = atoi(argv[++i]);
        else if (arg == "-d" && i < argc - 1) device_id = atoi(argv[++i]);
        else if (arg == "-l") cout << ListPlatformsDevices() << endl;
        else if (arg == "-f" && i < argc - 1) image_filename = argv[++i];
        else if (arg == "-h") { print_help(); return 0; }
    }

    try {
        CImg<unsigned char> image_input(image_filename.c_str());
        cl::Context context = GetContext(platform_id, device_id);
        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
        cl::Program::Sources sources;
        AddSources(sources, "kernels.cl");
        cl::Program program(context, sources);
        program.build();

        vector<int> intensity_histogram = create_intensity_histogram(program, context, queue, image_input);
        vector<int> cumulative_histogram = cumulate_histogram(program, context, queue, intensity_histogram);
        CImg<unsigned char> output_image = map_histogram_to_image(program, context, queue, image_input, cumulative_histogram);

        CImgDisplay disp_input(image_input, "Input");
        CImgDisplay disp_output(output_image, "Output");
        while (!disp_input.is_closed() && !disp_input.is_keyESC()) {
            disp_input.wait(1);
            disp_output.wait(1);
        }
    }
    catch (cl::Error& err) {
        cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << endl;
    }
    catch (CImgException& err) {
        cerr << "ERROR: " << err.what() << endl;
    }

    return 0;
}
