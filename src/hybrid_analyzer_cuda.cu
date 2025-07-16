#include "../include/scene.h"
#include "../include/kernels.h"
#include "../include/cuda_utils.h"
#include "../include/common_math.h"
#include "../include/materials.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>

// GPU renderer function
extern "C" void render_gpu_cuda_wrapper(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed, int threads_x, int threads_y) {
    // Allocate device memory
    Vec3* d_framebuffer;
    gpuErrchk(cudaMalloc(&d_framebuffer, width * height * sizeof(Vec3)));
    
    // Initialize random states
    curandState* d_rand_states;
    int num_states = width * height;
    gpuErrchk(cudaMalloc(&d_rand_states, num_states * sizeof(curandState)));
    
    dim3 rand_block_size(256);
    dim3 rand_grid_size((num_states + rand_block_size.x - 1) / rand_block_size.x);
    
    init_random_states_kernel<<<rand_grid_size, rand_block_size>>>(d_rand_states, num_states, base_seed);
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy spheres to device
    SphereData_Device* d_spheres;
    gpuErrchk(cudaMalloc(&d_spheres, num_spheres * sizeof(SphereData_Device)));
    gpuErrchk(cudaMemcpy(d_spheres, spheres, num_spheres * sizeof(SphereData_Device), cudaMemcpyHostToDevice));
    
    // Launch kernel
    dim3 block_size(threads_x, threads_y);
    dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                   (height + block_size.y - 1) / block_size.y);
    
    render_kernel<<<grid_size, block_size>>>(d_framebuffer, width, height, camera,
                                            d_spheres, num_spheres,
                                            nullptr, 0,
                                            d_rand_states);
    
    gpuErrchk(cudaDeviceSynchronize());
    
    // Copy results back to host
    gpuErrchk(cudaMemcpy(output, d_framebuffer, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost));
    
    // Cleanup
    cudaFree(d_framebuffer);
    cudaFree(d_rand_states);
    cudaFree(d_spheres);
}
