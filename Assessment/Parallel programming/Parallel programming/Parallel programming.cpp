#include <CL/opencl.hpp>
#include <vector>
#include <iostream>
#include <fstream>

std::string loadKernelSource(const std::string& filename) {
	std::ifstream file(filename);
	if (!file.is_open()) {
		std:cerr << "Failed to open file: " << filename << std::endl;
		exit(1);
	}
	return std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

int main() {
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);
	cl::Platform platform = platforms.front();

	std::vector<cl::Device> devices;
	platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
	cl::Device device = devices.front();

	cl::Context context(device);
	cl::CommandQueue queue(context, device);

	std::string kernelSource = loadKernelSource("kernels.cl");
	cl::Program::Sources sources;
	sources.push_back({ kernelSource.c_str(), kernelSource.length() });
	cl::Program program(context, sources);
	if (program.build({ device }) != CL_SUCCESS{
		std::cerr << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << std::endl;
		exit(1);
		}
}