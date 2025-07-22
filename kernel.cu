#include <iostream>
#include <string>
#include <time.h>
#include <float.h>
#include <thread>
#include <chrono>

#include <curand_kernel.h>

#define GLFW_DLL

#include <GL\glew.h>
#include <GLFW\glfw3.h>

//#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <surface_indirect_functions.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hitable_list.h"
#include "camera.h"
#include "material.h"

#include "svpng.h"
#include "lodepng.h"
#include "lodepng.cpp"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
GLFWwindow *window;

vec3 frontVec = vec3(0, 0, 1);

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 getBackGround(vec3 &direction, unsigned char *background, unsigned int background_width, unsigned int background_height)
{
    vec3 tmp = direction, output;
    float u = (0.5 - atan2(tmp.x(), tmp.z()) / (M_PI * 2)) * background_width, v =
            (0.5 - tmp.y() * 0.5) * background_height;
    int base0, base1, base2, base3, base_u = u, base_v = v;
    if (u - base_u >= 0.0f && v - base_v >= 0.0f)
    {
        base0 = base_u + base_v * background_width;
        base1 = (base_u + 1) + base_v * background_width;
        base2 = base_u + (base_v + 1) * background_width;
        for (int i = 0; i < 3; i++)
        {
            output.e[i] = (u - base_u) * background[3 * base1 + i] + (1 - (u - base_u)) * background[3 * base0 + i] +
                          (v - base_v) * background[3 * base2 + i] + (1 - (v - base_v)) * background[3 * base0 + i];
        }
        output /= 2;
    }
    else if (u - base_u >= 0.0f)
    {
        base0 = base_u + base_v * background_width;
        base1 = (base_u + 1) + base_v * background_width;
        for (int i = 0; i < 3; i++)
            output.e[i] = (u - base_u) * background[3 * base1 + i] + (1 - (u - base_u)) * background[3 * base0 + i];
    }
    else if (v - base_v >= 0.0f)
    {
        base0 = base_u + base_v * background_width;
        base2 = base_u + (base_v + 1) * background_width;
        for (int i = 0; i < 3; i++)
            output.e[i] = (v - base_v) * background[3 * base2 + i] + (1 - (v - base_v)) * background[3 * base0 + i];
    }
    else
    {
        for (int i = 0; i < 3; i++)
            output.e[i] = background[3 * base0 + i];
    }

    output /= 255.99f;

    return output;
}

__device__ vec3 color(const ray &r, hitable **world, curandState *local_rand_state, unsigned char *background, unsigned int background_width, unsigned int background_height, float density)
{
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++)
    {
        hit_record rec;
        if ((*world)->hit(cur_ray, 0.0001f, FLT_MAX, rec))
        {
            ray scattered;
            vec3 attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state))
            {
                //if(curand_uniform(local_rand_state) < density * rec.t && !scattered.inside)
                //{
                //    rec.t *= curand_uniform(local_rand_state);
                //    rec.p = cur_ray.point_at_parameter(rec.t);
                //    scattered = ray(rec.p, unit_vector(vec3(0.0,11.0,5.0) + vec3(cos(2*curand_uniform(local_rand_state)*M_PI),0,sin(2*curand_uniform(local_rand_state)*M_PI)) - rec.p));
                //    attenuation = vec3(0.7,0.7,0.7);
                //}
                cur_attenuation *= attenuation;
                if (cur_ray.done)
                    return cur_attenuation;
                cur_ray = scattered;
            }
            else
            {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else
        {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            vec3 c = getBackGround(unit_direction, background, background_width, background_height);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__global__ void rand_init(curandState *rand_state)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
        curand_init(0, 0, 0, rand_state);
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(0, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(int max_x, int max_y, int ns, camera **cam, hitable **world, curandState *rand_state, unsigned char *background, unsigned int background_width, unsigned int background_height, cudaSurfaceObject_t surface, int nscount, float density)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y))
        return;
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];
    vec3 col(0, 0, 0);
    for (int s = 0; s < ns; s++)
    {
        float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v, &local_rand_state);
        col += color(r, world, &local_rand_state, background, background_width, background_height, density);
    }
    rand_state[pixel_index] = local_rand_state;
    col /= float(ns);
    float4 color = make_float4(col.x(), col.y(), col.z(), 1);
    float4 old;
    surf2Dread(&old, surface, (int) (i * sizeof(color)), j, cudaBoundaryModeTrap);

    int sampleCount = 0;
    if(nscount != 1)
        sampleCount = old.w;

    if(col.length() != 0)
    {
        sampleCount ++;
        color = make_float4(old.x + ((color.x - old.x) / sampleCount), old.y + ((color.y - old.y) / sampleCount), old.z + ((color.z - old.z) / sampleCount), sampleCount);
    }
    else if(nscount != 1)
        color = make_float4(old.x, old.y, old.z, sampleCount);
    color = make_float4(fmin(fmax(color.x,0.f),1.f),fmin(fmax(color.y,0.f),1.f),fmin(fmax(color.z,0.f),1.f),color.w); // capping color in range(0,1)
    surf2Dwrite(color, surface, i * sizeof(color), j, cudaBoundaryModeClamp);

    //color = make_float4(old.x + ((color.x - old.x) / nscount), old.y + ((color.y - old.y) / nscount), old.z + ((color.z - old.z) / nscount), sampleCount);
    //surf2Dwrite(color, surface, i * sizeof(color), j, cudaBoundaryModeClamp);
}

#define RND (curand_uniform(&local_rand_state))

__global__ void
create_IOW_scene(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state, vec3 lookfrom, vec3 lookat)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        int i = 0;
        d_list[i++] = new sphere(vec3(0, -1e3, -0), 1e3, new metal(vec3(1.0, 1.0, 1.0),0.8));
        for (int a = -5; a < 5; a++)
        {
            for (int b = -5; b < 5; b++)
            {
                float choose_mat = RND;
                vec3 center(a + RND * 0.5, 0.2 + RND * 2, b + RND * 0.5);
                if (choose_mat < 0.6f)
                {
                    d_list[i++] = new sphere(center, 0.2, new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.7f)
                {
                    d_list[i++] = new sphere(center, 0.2, new light(vec3(0.5f + RND * RND * 10.0f, 0.5f + RND * RND * 10.0f, 0.5f + RND * RND * 10.0f)));
                }
                else if (choose_mat < 0.9f)
                {
                    d_list[i++] = new sphere(center, 0.2, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else
                {
                    d_list[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
        d_list[i++] = new sphere(vec3(-6, 1, 0), 1.0, new dielectric(1.33));
        d_list[i++] = new sphere(vec3(6, 1, 0), 1.0, new metal(vec3(0.9, 0.9, 0.9), 0.0));
        d_list[i++] = new sphere(vec3(0, 3, 6), 1.0, new light(vec3(10.0f, 10.0f, 10.0f)));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);

        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.0;//0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 60.0, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

__global__ void
create_cornell_box(hitable **d_list, hitable **d_world, camera **d_camera, int nx, int ny, curandState *rand_state, vec3 lookfrom, vec3 lookat)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        curandState local_rand_state = *rand_state;
        int i = 0;
        d_list[i++] = new sphere(vec3(0, 10+5e2, 5), 5e2, new lambertian(vec3(0.7, 0.7, 0.7))); // roof
        d_list[i++] = new sphere(vec3(0, -5e2, 5), 5e2, new lambertian(vec3(0.7,0.7,0.7))); // floor
        d_list[i++] = new sphere(vec3(0, 5, 10+5e2), 5e2, new lambertian(vec3(0.7, 0.7, 0.7))); // back
        d_list[i++] = new sphere(vec3(0, 5, -10-5e2), 5e2, new lambertian(vec3(0.7, 0.7, 0.7))); // behind cam
        d_list[i++] = new sphere(vec3(5+5e2, 5, 5), 5e2, new lambertian(vec3(0.7, 0.0, 0.0))); // left
        d_list[i++] = new sphere(vec3(-5-5e2, 5, 5), 5e2, new lambertian(vec3(0.0, 0.7, 0.0))); // right

        //for (int a = -4; a < 5; a++)
        //{
        //    for (int b = 1; b < 10; b++)
        //    {
        //        float choose_mat = RND;
        //        vec3 center(a + RND * 0.5, 4.0 + RND * 4, b + RND * 0.5);
        //        if (choose_mat < 0.5f)
        //        {
        //            d_list[i++] = new sphere(center, 0.4, new lambertian(vec3(RND * RND, RND * RND, RND * RND)));
        //        }
        //        else if (choose_mat < 0.8f)
        //        {
        //            d_list[i++] = new sphere(center, 0.4, new metal(vec3(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
        //        }
        //        else
        //        {
        //            d_list[i++] = new sphere(center, 0.4, new dielectric(1.5));
        //        }
        //    }
        //}

        d_list[i++] = new sphere(vec3(-2, 3, 5), 2.5, new dielectric(1.45));
        d_list[i++] = new sphere(vec3(2, 1.5, 3), 1.5, new metal(vec3(184,159,141)/255.99, 0.8));
        d_list[i++] = new sphere(vec3(1.8, 5, 7.5), 1.2, new lambertian(vec3(0.7, 0.7, 0.7)));
        //d_list[i++] = new sphere(vec3(1, 8, 3), 0.7, new dielectric(2.41));
        //d_list[i++] = new sphere(vec3(-0, 4, 5), 2.5, new dielectric(2.41));
        //d_list[i++] = new sphere(vec3(-2, 7, 8), 0.5, new lambertian(vec3(0.7, 0.5, 0.2)));
        //d_list[i++] = new sphere(vec3(-3, 1.5, 3), 1.0, new metal(vec3(184,159,141)/255.99, 0.3));
        //d_list[i++] = new sphere(vec3(2, 2.5, 2), 0.6, new lambertian(vec3(0.7, 0.7, 0.7)));
        d_list[i++] = new sphere(vec3(0, 9.995+1e2, 5), 1e2, new light(vec3(2.5f, 2.5f, 2.5f)));
        *rand_state = local_rand_state;
        *d_world = new hitable_list(d_list, i);

        float dist_to_focus = 10.0;
        (lookfrom - lookat).length();
        float aperture = 0.0;//0.1;
        *d_camera = new camera(lookfrom, lookat, vec3(0, 1, 0), 60.0, float(nx) / float(ny), aperture, dist_to_focus);
    }
}

__global__ void resetCamera(camera **d_camera, int nx, int ny, vec3 lookfrom, vec3 lookat, vec3 move)
{
    float dist_to_focus = 10.0;
    float aperture = 0.0f;
    **d_camera = camera(lookfrom + move, lookat + move, vec3(0, 1, 0), 60.0, float(nx) / float(ny), aperture, dist_to_focus);
}

__global__ void free_world(hitable **d_list, hitable **d_world, camera **d_camera)
{
    for (int i = 0; i < 10 * 10 + 1 + 3; i++)
    {
        delete ((sphere *) d_list[i])->mat_ptr;
        delete d_list[i];
    }
    delete *d_world;
    delete *d_camera;
}

constexpr int nx = 1200;
constexpr int ny = 800;
bool canAccess = false;
bool needReset = false;

int nsCount = 0;

//player movement
vec3 moveCPU = vec3(0, 0, 0);
vec3 lookfromCPU = vec3(0, 5, -8);
vec3 lookatCPU = vec3(0, 5, -7);

float density = 25;

//set up background image
unsigned int background_width;
unsigned int background_height;
unsigned char *background;
unsigned char *device_background;

// set texture
GLuint viewGLTexture;
cudaGraphicsResource_t viewCudaResource;

unsigned char *runRT()
{
    canAccess = false;
    int ns = 1;
    int tx = 20;
    int ty = 32;
    clock_t start, stop;

    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << ns << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    start = clock();
    int num_pixels = nx * ny;

    curandState *d_rand_state;
    checkCudaErrors(cudaMalloc((void **) &d_rand_state, num_pixels * sizeof(curandState)));
    curandState *d_rand_state2;
    checkCudaErrors(cudaMalloc((void **) &d_rand_state2, 1 * sizeof(curandState)));

    rand_init<<<1, 1 >>>(d_rand_state2);
    stop = clock();
    double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took0: " << timer_seconds << " seconds.\n";

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    hitable **d_list;
    int num_hitables = 10 * 10 + 1 + 3; // default scene for raytracing in one weekend
    //int num_hitables = 85; // cornell box
    //int num_hitables = 10; // cornell box
    checkCudaErrors(cudaMalloc((void **) &d_list, num_hitables * sizeof(hitable *)));
    hitable **d_world;
    checkCudaErrors(cudaMalloc((void **) &d_world, sizeof(hitable *)));
    camera **d_camera;
    checkCudaErrors(cudaMalloc((void **) &d_camera, sizeof(camera *)));
    create_IOW_scene<<<1, 1 >>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, lookfromCPU, lookatCPU);
    //create_cornell_box<<<1, 1 >>>(d_list, d_world, d_camera, nx, ny, d_rand_state2, lookfromCPU, lookatCPU);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);
    render_init<<<blocks, threads >>>(nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    canAccess = false;

    while (!glfwWindowShouldClose(window))
    {
        while (canAccess)
        {
            std::chrono::duration<int, std::micro> timespan(1);
            std::this_thread::sleep_for(timespan);
        }
        start = clock();
        if (needReset)
        {
            nsCount = 0;
            resetCamera<<<1, 1>>>(d_camera, nx, ny, lookfromCPU, lookatCPU, moveCPU);
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            needReset = false;
        }

        checkCudaErrors(cudaGraphicsMapResources(1, &viewCudaResource, 0));

        cudaArray_t viewCudaArray;
        checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&viewCudaArray, viewCudaResource, 0, 0));

        cudaResourceDesc viewCudaArrayResourceDesc;
        memset(&viewCudaArrayResourceDesc, 0, sizeof(viewCudaArrayResourceDesc));
        viewCudaArrayResourceDesc.resType = cudaResourceTypeArray;
        viewCudaArrayResourceDesc.res.array.array = viewCudaArray;

        cudaSurfaceObject_t viewCudaSurfaceObject;
        checkCudaErrors(cudaCreateSurfaceObject(&viewCudaSurfaceObject, &viewCudaArrayResourceDesc));

        nsCount++;
        render<<<blocks, threads >>>(nx, ny, ns, d_camera, d_world, d_rand_state, device_background, background_width, background_height, viewCudaSurfaceObject, nsCount, density/1000);

        checkCudaErrors(cudaDestroySurfaceObject(viewCudaSurfaceObject));
        checkCudaErrors(cudaGraphicsUnmapResources(1, &viewCudaResource));
        checkCudaErrors(cudaStreamSynchronize(0));
        checkCudaErrors(cudaDeviceSynchronize());

        stop = clock();
        double timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;

        //show fps on title
        std::string fpsStr = "Cuda realTime ray tracing, fps: ";
        fpsStr += std::to_string(1.0 / timer_seconds);
        glfwSetWindowTitle(window, fpsStr.c_str());

        glBindTexture(GL_TEXTURE_2D, viewGLTexture);
        {
            glBegin(GL_QUADS);
            {
                glTexCoord2f(0.0f, 0.0f);
                glVertex2f(0.0f, 0.0f);
                glTexCoord2f(1.0f, 0.0f);
                glVertex2f(nx, 0.0f);
                glTexCoord2f(1.0f, 1.0f);
                glVertex2f(nx, ny);
                glTexCoord2f(0.0f, 1.0f);
                glVertex2f(0.0f, ny);
            }
            glEnd();
        }
        glBindTexture(GL_TEXTURE_2D, 0);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

//    FILE* output = fopen("output.png", "wb");
//    svpng(output, nx, ny, rgb, 0);
//    fclose(output);

    // clean up
    start = clock();
    checkCudaErrors(cudaDeviceSynchronize());
    free_world<<<1, 1 >>>(d_list, d_world, d_camera);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_list));
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(device_background));
    // useful for cuda-memcheck --leak-check full
    cudaDeviceReset();
    stop = clock();
    timer_seconds = ((double) (stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took3: " << timer_seconds << " seconds.\n";
}

double rlangle = 0;
double udangle = 0;

void processInput(GLFWwindow *window)
{
    while (true)
    {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);
        double scale = 0.000001;

        //WASD
        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) //W
        {
            //forward
            lookfromCPU += frontVec * scale;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)//A
        {
            vec3 leftVec = vec3(frontVec.z(), 0, -frontVec.x());
            lookfromCPU += leftVec * scale;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) //S
        {
            lookfromCPU -= frontVec * scale;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) //D
        {
            vec3 rightVec = vec3(-frontVec.z(), 0, frontVec.x());
            lookfromCPU += rightVec * scale;
            needReset = true;
        }

        if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS)
        {
            lookfromCPU[1] -= scale;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS)
        {
            lookfromCPU[1] += scale;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        {
            if (udangle <= 90)
            {
                udangle += 0.000005;
                needReset = true;
                frontVec = vec3(sin(rlangle * M_PI / 180), sin(udangle * M_PI / 180), cos(rlangle * M_PI / 180));
            }
        }
        if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        {
            if (udangle >= -90)
            {
                udangle -= 0.000005;
                needReset = true;
                frontVec = vec3(sin(rlangle * M_PI / 180), sin(udangle * M_PI / 180), cos(rlangle * M_PI / 180));
            }
        }
        if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        {
            rlangle += 0.000005;
            frontVec = vec3(sin(rlangle * M_PI / 180), sin(udangle * M_PI / 180), cos(rlangle * M_PI / 180));
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        {
            rlangle -= 0.000005;
            frontVec = vec3(sin(rlangle * M_PI / 180), sin(udangle * M_PI / 180), cos(rlangle * M_PI / 180));
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            if(density < 1000) density += 0.00004;
            else density = 1000;
            needReset = true;
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        {
            if(density > 0) density -= 0.00004;
            else density = 0;
            needReset = true;
        }
        lookatCPU = lookfromCPU + frontVec;
        //std::this_thread::sleep_for(std::chrono::nanoseconds(1));
    }
}

void generateDebugChecker(unsigned char *texture, unsigned int width, unsigned int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (i % 20 < 10 && j % 20 < 10)
            {
                for (int i = 0; i < 3; i++)
                    texture[3 * (i * width + j) + i] = 255;
            }
            else if (i % 20 >= 10 && j % 20 >= 10)
            {
                for (int i = 0; i < 3; i++)
                    texture[3 * (i * width + j) + i] = 255;
            }
            else
            {
                for (int i = 0; i < 3; i++)
                    texture[3 * (i * width + j) + i] = 0;
            }
        }
    }
}

/* program entry */
int main(int argc, char *argv[])
{
    cudaSetDevice(0);

    // load background image
    unsigned error;
    error = lodepng_decode24_file(&background, &background_width, &background_height, "comfy_cafe_16k.png");
    if (error)
        printf("error %u: %s\n", error, lodepng_error_text(error));
    else
        printf("load background\n");

    checkCudaErrors(cudaMalloc((void **) &device_background, background_width * background_height * 3 * sizeof(unsigned char)));
    checkCudaErrors(cudaMemcpy(device_background, background, background_width * background_height * 3 * sizeof(unsigned char), cudaMemcpyHostToDevice));

    int num_pixels = nx * ny;

    if (!glfwInit())
    {
        fprintf(stderr, "Failed to initialize GLFW\n");
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
    window = glfwCreateWindow(nx, ny, "Cuda realTime ray tracing, fps: ", NULL, NULL);
    if (!window)
    {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        exit(EXIT_FAILURE);
    }
    glfwMakeContextCurrent(window);
    GLenum err = glewInit();
    if (GLEW_OK != err) printf("glew fork up: %s\n", glewGetErrorString(err));

    if (GLEW_VERSION_2_0)
    {
        printf("support 2.0\n");
    }

    glEnable(GL_TEXTURE_2D);
    glGenTextures(1, &viewGLTexture);
    glBindTexture(GL_TEXTURE_2D, viewGLTexture);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, nx, ny, 0, GL_RGBA, GL_FLOAT, NULL);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    checkCudaErrors(cudaGraphicsGLRegisterImage(&viewCudaResource, viewGLTexture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsNone));

    glfwSwapInterval(1);

    // set up view
    glViewport(0, 0, nx, ny);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();

    glOrtho(0.0, nx, 0.0, ny, 0.0, 1.0); // this creates a canvas you can do 2D drawing on

    std::thread processInputThread(processInput, window);
    processInputThread.detach();

    runRT();

    // Terminate GLFW
    glfwTerminate();

    // Exit program
    exit(EXIT_SUCCESS);
}