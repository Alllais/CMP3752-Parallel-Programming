#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include "Utils.h"
#include "CImg.h"

using namespace cimg_library;
using namespace std;

const int COLOR_CHANNELS = 3;
const int BIN_COUNT = 256;

void printHelp() {
    cerr << "Application usage:" << endl;
    cerr << "  -p : select platform " << endl;
    cerr << "  -d : select device" << endl;
    cerr << "  -l : list all platforms and devices" << endl;
    cerr << "  -f : input image file (default: test.ppm)" << endl;
    cerr << "  -h : print this message" << endl;
}

vector<int> createIntensityHistogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, const CImg<unsigned char>& from) {
    cl::Buffer inputBuffer(context, CL_MEM_READ_ONLY, from.size());
    cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, BIN_COUNT * COLOR_CHANNELS * sizeof(int));
    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, from.size(), from.data());

    cl::Kernel kernel(program, "create_intensity_histogram");
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(from.size() / COLOR_CHANNELS), cl::NullRange);

    vector<int> histogram(BIN_COUNT * COLOR_CHANNELS);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, BIN_COUNT * COLOR_CHANNELS * sizeof(int), histogram.data());

    return histogram;
}

vector<int> cumulateHistogram(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, const vector<int>& histogram) {
    size_t bufferSize = histogram.size() * sizeof(int);
    cl::Buffer inputBuffer(context, CL_MEM_READ_WRITE, bufferSize);
    cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, bufferSize);
    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, bufferSize, histogram.data());

    cl::Kernel kernel(program, "cumulate_histogram");
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(histogram.size() / COLOR_CHANNELS), cl::NullRange);

    vector<int> cumulativeHistogram(histogram.size());
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, bufferSize, cumulativeHistogram.data());

    return cumulativeHistogram;
}

CImg<unsigned char> mapCumulativeHistogramToImage(cl::Program& program, cl::Context& context, cl::CommandQueue& queue, const CImg<unsigned char>& inputImage, const vector<int>& cumulativeHistogram) {
    cl::Buffer inputImageBuffer(context, CL_MEM_READ_ONLY, inputImage.size());
    cl::Buffer inputHistogramBuffer(context, CL_MEM_READ_ONLY, cumulativeHistogram.size() * sizeof(int));
    cl::Buffer outputBuffer(context, CL_MEM_READ_WRITE, inputImage.size());
    queue.enqueueWriteBuffer(inputImageBuffer, CL_TRUE, 0, inputImage.size(), inputImage.data());
    queue.enqueueWriteBuffer(inputHistogramBuffer, CL_TRUE, 0, cumulativeHistogram.size() * sizeof(int), cumulativeHistogram.data());

    cl::Kernel kernel(program, "map_cumulative_histogram_to_image");
    kernel.setArg(0, inputImageBuffer);
    kernel.setArg(1, inputHistogramBuffer);
    kernel.setArg(2, outputBuffer);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(inputImage.size() / COLOR_CHANNELS), cl::NullRange);

    vector<unsigned char> imageData(inputImage.size());
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, inputImage.size(), imageData.data());
    CImg<unsigned char> outputImage(imageData.data(), inputImage.width(), inputImage.height(), inputImage.depth(), inputImage.spectrum());

    return outputImage;
}

int main(int argc, char** argv) {
    string imageFilename = "test.ppm";
    int platformId = 0, deviceId = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) { platformId = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) { deviceId = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-l") == 0) { cout << ListPlatformsDevices() << endl; }
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) { imageFilename = argv[++i]; }
        else if (strcmp(argv[i], "-h") == 0) { printHelp(); return 0; }
    }

    try {
        CImg<unsigned char> inputImage(imageFilename.c_str());
        CImgDisplay dispInput(inputImage, "Input");

        cl::Context context = GetContext(platformId, deviceId);
        cl::CommandQueue queue(context);
        cl::Program::Sources sources;
        AddSources(sources, "kernels.cl");
        cl::Program program(context, sources);

        program.build();

        auto histogram = createIntensityHistogram(program, context, queue, inputImage);
        auto cumulativeHistogram = cumulateHistogram(program, context, queue, histogram);
        auto outputImage = mapCumulativeHistogramToImage(program, context, queue, inputImage, cumulativeHistogram);

        CImgDisplay dispOutput(outputImage, "Output");
        while (!dispInput.is_closed() && !dispInput.is_keyESC()) {
            dispInput.wait();
            dispOutput.wait();
        }
    }
    catch (const cl::Error& err) {
        cerr << "OpenCL Error: " << err.what() << ", " << getErrorString(err.err()) << endl;
    }
    catch (const CImgException& err) {
        cerr << "CImg Error: " << err.what() << endl;
    }
    catch (const exception& e) {
        cerr << "Standard Exception: " << e.what() << endl;
    }
    return 0;
}