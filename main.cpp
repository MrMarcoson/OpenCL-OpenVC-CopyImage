#include <math.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <random>
#include <CL/cl2.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

cv::Mat load_image(string file_name)
{
    cv::Mat image = cv::imread(file_name);
    if (image.empty()) {
        std::cout << "Image not loaded" << endl;
    }

    cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
    return image;
}

string get_kernel(string file) {
    std::ifstream stream(file);
    std::stringstream kernel;
    kernel << stream.rdbuf();
    
    return kernel.str();
}

void setup_device(cl::Platform &platform, cl::Device &device) {
    std::vector<cl::Platform> all_platforms;
    cl::Platform::get(&all_platforms);
    if (all_platforms.size() == 0) {
        std::cout << " 0 platforms \n";
        exit(1);
    }

    platform = all_platforms[0];

    std::vector<cl::Device> all_devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
    if (all_devices.size() == 0) {
        std::cout << " 0 devices\n";
        exit(1);
    }

    device = all_devices[0];
}

int main() {
    long unsigned int image_size = 2;

    cl::Platform platform;
    cl::Device device;
    setup_device(platform, device);
    std::cout << platform.getInfo<CL_PLATFORM_NAME>() << ": " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context({ device });
    cl::CommandQueue queue(context, device, CL_QUEUE_PROFILING_ENABLE);

    cl::Program::Sources sources;
    string kernel_code = get_kernel("copy_image.cl");
    sources.push_back({ kernel_code.c_str(), kernel_code.length() });

    cl::Program program(context, sources);
    if(program.build({ device }) != CL_SUCCESS) {
        std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device) << "\n";
        exit(1);
    }

    cl::Kernel kernel(program, "copy_image");

    cv::Mat input_image = load_image("file.png");
    std::vector<uchar> input_arr(input_image.size().width * input_image.size().height * 4);
    std::vector<uchar> output_arr(image_size * image_size * 4);

    int step = 0;
    for(int i = 0; i < input_image.size().width ; i++) {
        for(int j = 0; j < input_image.size().height; j++) {
            input_arr[step] = (int)input_image.at<cv::Vec3b>(i, j)[0];
            input_arr[step + 1] = (int)input_image.at<cv::Vec3b>(i, j)[1];
            input_arr[step + 2] = (int)input_image.at<cv::Vec3b>(i, j)[2];
            input_arr[step + 3] = (int)input_image.at<cv::Vec3b>(i, j)[3];
            step += 4;
        }
    }

    cl::ImageFormat format(CL_RGBA, CL_UNORM_INT8);
	cl::Image2D Input_Image(context, CL_MEM_READ_ONLY, format, input_image.size().width, input_image.size().height);
	cl::Image2D Output_Image(context, CL_MEM_WRITE_ONLY, format, image_size, image_size);

	std::array<size_t, 3> origin { 0, 0, 0 };
    std::array<size_t, 3> input_region { input_image.size().width, input_image.size().height, 1 };
	std::array<size_t, 3> output_region { image_size, image_size, 1 };

	queue.enqueueWriteImage(Input_Image, CL_TRUE, origin, input_region, 0, 0, &input_arr[0]);
	
    kernel.setArg(0, Input_Image);
	kernel.setArg(1, Output_Image);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(image_size, image_size), cl::NullRange, NULL);
    queue.enqueueReadImage(Output_Image, CL_TRUE, origin, output_region, 0, 0, &output_arr[0]);

    cv::Mat output_image(image_size, image_size, CV_8UC4, output_arr.data());
    cv::cvtColor(output_image, output_image, cv::COLOR_RGBA2BGRA);
    cv::imwrite("output.png", output_image);

    return 0;
}
