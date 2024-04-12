#pragma once

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
    const int BIN_COUNT = 256;
    vector<int> histogram(BIN_COUNT * 3, 0);
    cl::Event write_event, kernel_event, read_event;

    // Buffers setup
    cl::Buffer input_buffer(context, CL_MEM_READ_ONLY, img.size());
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * histogram.size());
    queue.enqueueWriteBuffer(input_buffer, CL_TRUE, 0, img.size(), img.data(), nullptr, &write_event);

    // Kernel setup
    cl::Kernel kernel(program, "create_intensity_histogram");
    kernel.setArg(0, input_buffer);
    kernel.setArg(1, output_buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(img.size() / 3), cl::NullRange, nullptr, &kernel_event);
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data(), nullptr, &read_event);

    // Performance info
    cout << "---------------CREATE INTENSITY HISTOGRAM---------------" << endl
        << "Load image buffer: " << GetFullProfilingInfo(write_event, PROF_US) << endl
        << "Generate intensity histogram: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl
        << "Retrieve histogram: " << GetFullProfilingInfo(read_event, PROF_US) << endl;

    return histogram;
}

vector<int> cumulate_histogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, vector<int> histogram) {
    // Buffers setup
    cl::Event write_event, kernel_event, read_event;
    cl::Buffer buffer(context, CL_MEM_READ_WRITE, sizeof(int) * histogram.size());
    queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data(), nullptr, &write_event);

    // Kernel setup
    cl::Kernel kernel(program, "cumulate_histogram");
    kernel.setArg(0, buffer);
    kernel.setArg(1, buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size() / 3), cl::NullRange, nullptr, &kernel_event);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data(), nullptr, &read_event);

    // Performance info
    cout << "---------------CUMULATE HISTOGRAM---------------" << endl
        << "Load histogram buffer: " << GetFullProfilingInfo(write_event, PROF_US) << endl
        << "Generate cumulative histogram: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl
        << "Retrieve cumulative histogram: " << GetFullProfilingInfo(read_event, PROF_US) << endl;

    return histogram;
}

CImg<unsigned char> map_histogram_to_image(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, CImg<unsigned char> img, vector<int> histogram) {


    cl::Event write_img_event, write_hist_event, kernel_event, read_event;
    cl::Buffer img_buffer(context, CL_MEM_READ_ONLY, img.size());
    cl::Buffer hist_buffer(context, CL_MEM_READ_ONLY, sizeof(int) * histogram.size());
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY, img.size());
    queue.enqueueWriteBuffer(img_buffer, CL_TRUE, 0, img.size(), img.data(), nullptr, &write_img_event);
    queue.enqueueWriteBuffer(hist_buffer, CL_TRUE, 0, sizeof(int) * histogram.size(), histogram.data(), nullptr, &write_hist_event);

    // Kernel setup
    cl::Kernel kernel(program, "map_cumulative_histogram_to_image");
    kernel.setArg(0, img_buffer);
    kernel.setArg(1, hist_buffer);
    kernel.setArg(2, output_buffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(img.size() / 3), cl::NullRange, nullptr, &kernel_event);

    vector<unsigned char> output_data(img.size());
    queue.enqueueReadBuffer(output_buffer, CL_TRUE, 0, img.size(), output_data.data(), nullptr, &read_event);

    // Performance info
    cout << "---------------MAP CUMULATIVE HISTOGRAM TO IMAGE---------------" << endl
        << "Load image buffer: " << GetFullProfilingInfo(write_img_event, PROF_US) << endl
        << "Load histogram buffer: " << GetFullProfilingInfo(write_hist_event, PROF_US) << endl
        << "Generate modified image: " << GetFullProfilingInfo(kernel_event, PROF_US) << endl
        << "Retrieve modified image: " << GetFullProfilingInfo(read_event, PROF_US) << endl;

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
        CImgDisplay disp_input(image_input, "Input");

        cl::Context context = GetContext(platform_id, device_id);
        cl::CommandQueue queue(context, CL_QUEUE_PROFILING_ENABLE);
        cl::Program::Sources sources;
        AddSources(sources, "kernels.cl");
        cl::Program program(context, sources);
        program.build();

        vector<int> intensity_histogram = create_intensity_histogram(program, context, queue, image_input);
        vector<int> cumulative_histogram = cumulate_histogram(program, context, queue, intensity_histogram);
        CImg<unsigned char> output_image = map_histogram_to_image(program, context, queue, image_input, cumulative_histogram);

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
