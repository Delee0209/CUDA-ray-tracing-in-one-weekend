# CUDA ray tracing in one weekend
- Term project for the course "Introduction to CUDA Parallel Programming" at NTU
- In this implementation, we reference some of the code from [NVIDIA CUDA ray tracing in one weekend tutorial's code](https://developer.nvidia.com/blog/accelerated-ray-tracing-cuda/) with its random number generation and overall structure for set up the device memory, then port the original Ray tracing in one weekend host code to device code, and add support to some new feature like environment map for background, free camera and camera angle for ease of observation and draw a window for display.
- the result in present in real-time using some OpenGL library (glfw + glew).
    - we ulitize the `cudaGraphicsResource` and `cudaGraphicsGLRegisterImage` to share the GPU memory between CUDA and OPENGL, allow us to avoid any memory transfer and copy to the host side.
## Gallery
![image](https://hackmd.io/_uploads/r1n_kQ1vle.png)
![未命名](https://hackmd.io/_uploads/HkUNxmkvxl.jpg)
![image](https://hackmd.io/_uploads/rJXLxXkvel.png)
![未命名](https://hackmd.io/_uploads/BkwDZmkvgx.jpg)
