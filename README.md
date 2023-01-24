# OpenCL-OpenVC-CopyImage
Template project for image processing in OpenCL. Uses OpenCV to input and output files. 
Program loads image to OpenCV Mat, then maps it as RGBA to 1D array of structure:

```
  pixel: 0, 0     pixel: 0, 1   pixel: n, n
{ r, g, b, a, r, g, b, a, ... r, g, b, a }
```
Image is enqueued to GPU device as OpenCL Image2D object, and copies it to output image.
GPU returns array of same structure as above, that is parsed to Mat object and converted again to BGR. 


# Dependencies

## OpenCL
```
sudo apt install opencl-headers ocl-icd-opencl-dev -y
```

## OpenCV
```
sudo apt install python3-opencv libopencv-dev 
```

## Compilaton and usage
```
g++ main.cpp -lOpenCL `pkg-config --cflags --libs opencv4`
./a.out
```
