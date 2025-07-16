#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <vector>
#include <fstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <omp.h>
#include <string>

// Include the headers from your project
#include "include/scene.h"
#include "include/kernels.h"
#include "include/cuda_utils.h"
#include "include/common_math.h"
#include "include/hybrid_render.h"
#include "include/materials.h"
#include "include/camera_shared.h"

// External CUDA wrapper function
extern "C" void render_gpu_cuda_wrapper(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed, int threads_x, int threads_y);

// Test configurations - EXACT MATCH WITH CUDA REFERENCE FOR CROSS-COMPARISON
// These values MUST match the CUDA-only analyzer for valid comparison
#define TEST_WIDTH 800
#define TEST_HEIGHT 600
#define NUM_ITERATIONS 5
// Use the same sampling as the CUDA reference
#define SAMPLES_PER_PIXEL 4

// Simple CPU ray tracer for comparison (same as CUDA version)
struct SimpleRay {
    Vec3 origin, direction;
};

struct SimpleHit {
    Vec3 point, normal;
    float t;
    Material_Device material;
    bool hit;
};

// Simple random number generator to match GPU behavior
class SimpleRandom {
private:
    unsigned long long state;
public:
    SimpleRandom(unsigned long long s) : state(s) {}
    
    float random_float() {
        state = state * 1103515245ULL + 12345ULL;
        return ((state >> 16) & 0xFFFFFFFF) / 4294967296.0f;
    }
    
    float random_float_range(float min, float max) {
        return min + (max - min) * random_float();
    }
    
    Vec3 random_in_unit_sphere() {
        while (true) {
            Vec3 p = vec3_create(random_float_range(-1.0f, 1.0f),
                                random_float_range(-1.0f, 1.0f),
                                random_float_range(-1.0f, 1.0f));
            if (vec3_length_squared(p) < 1.0f) return p;
        }
    }
    
    Vec3 random_unit_vector() {
        return vec3_normalize(random_in_unit_sphere());
    }
};

// Simple CPU sphere intersection (same as CUDA version)
SimpleHit intersect_sphere_cpu(const SimpleRay& ray, const SphereData_Device& sphere) {
    SimpleHit hit;
    hit.hit = false;
    
    Vec3 oc = vec3_sub(ray.origin, sphere.center);
    float a = vec3_dot(ray.direction, ray.direction);
    float b = 2.0f * vec3_dot(oc, ray.direction);
    float c = vec3_dot(oc, oc) - sphere.radius * sphere.radius;
    
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) return hit;
    
    float t = (-b - sqrtf(discriminant)) / (2.0f * a);
    if (t > 0.001f) {
        hit.hit = true;
        hit.t = t;
        hit.point = vec3_add(ray.origin, vec3_scale(ray.direction, t));
        hit.normal = vec3_normalize(vec3_sub(hit.point, sphere.center));
        hit.material = sphere.material;
    }
    
    return hit;
}

// CPU path tracing function (same as CUDA version)
Vec3 ray_color_cpu_path_trace(SimpleRay ray, const SphereData_Device* spheres, int num_spheres, SimpleRandom* rng) {
    Vec3 attenuation = vec3_create(1.0f, 1.0f, 1.0f);

    for (int depth = 0; depth < MAX_DEPTH; ++depth) {
        SimpleHit closest_hit;
        closest_hit.hit = false;
        closest_hit.t = INFINITY;

        for (int i = 0; i < num_spheres; i++) {
            SimpleHit hit = intersect_sphere_cpu(ray, spheres[i]);
            if (hit.hit && hit.t < closest_hit.t) {
                closest_hit = hit;
            }
        }

        if (closest_hit.hit) {
            Vec3 scatter_direction = vec3_add(closest_hit.normal, rng->random_unit_vector());
            if (vec3_length_squared(scatter_direction) < 1e-8f) {
                scatter_direction = closest_hit.normal;
            }
            
            ray.origin = closest_hit.point;
            ray.direction = vec3_normalize(scatter_direction);
            attenuation = vec3_mul(attenuation, closest_hit.material.albedo);
        } else {
            Vec3 unit_direction = vec3_normalize(ray.direction);
            float t = 0.5f * (unit_direction.y + 1.0f);
            Vec3 white = vec3_create(1.0f, 1.0f, 1.0f);
            Vec3 blue = vec3_create(0.5f, 0.7f, 1.0f);
            Vec3 background_color = vec3_add(vec3_scale(white, 1.0f - t), vec3_scale(blue, t));
            return vec3_mul(attenuation, background_color);
        }
    }

    return vec3_create(0.0f, 0.0f, 0.0f);
}

// CPU renderer with OpenMP (same camera/sampling logic as GPU)
void render_cpu_openmp(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed, int num_threads) {
    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = y * width + x;
            SimpleRandom rng(base_seed + pixel_index + omp_get_thread_num() * 1000);
            
            Vec3 pixel_color = vec3_create(0.0f, 0.0f, 0.0f);
            
            for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (width - 1);
                float v_img = (float)(y + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (height - 1);
                float v_cam = 1.0f - v_img;
                
                SimpleRay ray;
                ray.origin = camera.origin;
                ray.direction = vec3_normalize(vec3_sub(vec3_add(vec3_add(camera.lower_left_corner, 
                                                                          vec3_scale(camera.horizontal, u)), 
                                                                 vec3_scale(camera.vertical, v_cam)), 
                                                        camera.origin));
                
                pixel_color = vec3_add(pixel_color, ray_color_cpu_path_trace(ray, spheres, num_spheres, &rng));
            }
            
            pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);
            
            // Gamma correction
            pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
            pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
            pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
            
            output[pixel_index] = pixel_color;
        }
    }
}

// CPU renderer without OpenMP (serial version)
void render_cpu_serial(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_index = y * width + x;
            SimpleRandom rng(base_seed + pixel_index);
            
            Vec3 pixel_color = vec3_create(0.0f, 0.0f, 0.0f);
            
            for (int s = 0; s < SAMPLES_PER_PIXEL; s++) {
                float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (width - 1);
                float v_img = (float)(y + ((SAMPLES_PER_PIXEL > 1) ? rng.random_float() : 0.5f)) / (height - 1);
                float v_cam = 1.0f - v_img;
                
                SimpleRay ray;
                ray.origin = camera.origin;
                ray.direction = vec3_normalize(vec3_sub(vec3_add(vec3_add(camera.lower_left_corner, 
                                                                          vec3_scale(camera.horizontal, u)), 
                                                                 vec3_scale(camera.vertical, v_cam)), 
                                                        camera.origin));
                
                pixel_color = vec3_add(pixel_color, ray_color_cpu_path_trace(ray, spheres, num_spheres, &rng));
            }
            
            pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);
            
            // Gamma correction
            pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
            pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
            pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
            
            output[pixel_index] = pixel_color;
        }
    }
}

// GPU renderer function
void render_gpu_cuda(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed, int threads_x, int threads_y) {
    render_gpu_cuda_wrapper(output, width, height, spheres, num_spheres, camera, base_seed, threads_x, threads_y);
}

// Hybrid renderer that combines CPU and GPU workload
void render_hybrid_combined(Vec3* output, int width, int height, const SphereData_Device* spheres, int num_spheres, Camera_Device& camera, unsigned int base_seed, int num_threads, int gpu_threads_x, int gpu_threads_y) {
    // Split work: 50% GPU, 50% CPU
    int gpu_height = height / 2;
    int cpu_height = height - gpu_height;
    
    Vec3* gpu_result = (Vec3*)malloc(width * gpu_height * sizeof(Vec3));
    Vec3* cpu_result = (Vec3*)malloc(width * cpu_height * sizeof(Vec3));
    
    // GPU renders top half
    render_gpu_cuda(gpu_result, width, gpu_height, spheres, num_spheres, camera, base_seed, gpu_threads_x, gpu_threads_y);
    
    // CPU renders bottom half with adjusted camera
    Camera_Device cpu_camera = camera;
    // Adjust camera for bottom half
    float v_offset = (float)gpu_height / height;
    cpu_camera.lower_left_corner = vec3_add(cpu_camera.lower_left_corner, vec3_scale(cpu_camera.vertical, v_offset));
    
    render_cpu_openmp(cpu_result, width, cpu_height, spheres, num_spheres, cpu_camera, base_seed + width * gpu_height, num_threads);
    
    // Combine results
    memcpy(output, gpu_result, width * gpu_height * sizeof(Vec3));
    memcpy(output + width * gpu_height, cpu_result, width * cpu_height * sizeof(Vec3));
    
    free(gpu_result);
    free(cpu_result);
}

// Calculate MSE and RMSE
double calculate_mse(Vec3* img1, Vec3* img2, int width, int height) {
    double sum_squared_error = 0.0;
    int total_pixels = width * height;
    
    for (int i = 0; i < total_pixels; i++) {
        double dr = img1[i].x - img2[i].x;
        double dg = img1[i].y - img2[i].y;
        double db = img1[i].z - img2[i].z;
        
        sum_squared_error += dr * dr + dg * dg + db * db;
    }
    
    return sum_squared_error / (3.0 * total_pixels);
}

double calculate_rmse(Vec3* img1, Vec3* img2, int width, int height) {
    return sqrt(calculate_mse(img1, img2, width, height));
}

// Scene setup verification function - MUST match CUDA reference exactly
void verify_scene_consistency(const SphereData_Device* spheres, int num_spheres, const Camera_Device& camera) {
    printf("=== SCENE CONSISTENCY VERIFICATION ===\n");
    printf("Number of spheres: %d\n", num_spheres);
    
    for (int i = 0; i < num_spheres; i++) {
        printf("Sphere %d:\n", i);
        printf("  Center: (%.3f, %.3f, %.3f)\n", spheres[i].center.x, spheres[i].center.y, spheres[i].center.z);
        printf("  Radius: %.3f\n", spheres[i].radius);
        printf("  Material albedo: (%.3f, %.3f, %.3f)\n", spheres[i].material.albedo.x, spheres[i].material.albedo.y, spheres[i].material.albedo.z);
    }
    
    printf("Camera:\n");
    printf("  Origin: (%.3f, %.3f, %.3f)\n", camera.origin.x, camera.origin.y, camera.origin.z);
    printf("  Lower left corner: (%.3f, %.3f, %.3f)\n", camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z);
    printf("  Horizontal: (%.3f, %.3f, %.3f)\n", camera.horizontal.x, camera.horizontal.y, camera.horizontal.z);
    printf("  Vertical: (%.3f, %.3f, %.3f)\n", camera.vertical.x, camera.vertical.y, camera.vertical.z);
    printf("=======================================\n");
}

// Serial vs Hybrid performance test function
void run_hybrid_performance_test(int num_threads, int gpu_threads_x, int gpu_threads_y, FILE* csv_file) {
    printf("\n=== Testing Hybrid Configuration ===\n");
    printf("CPU threads: %d, GPU threads: %dx%d\n", num_threads, gpu_threads_x, gpu_threads_y);
    
    // Calculate GPU configuration details (same as CUDA reference)
    int threads_per_block = gpu_threads_x * gpu_threads_y;
    int grid_x = (TEST_WIDTH + gpu_threads_x - 1) / gpu_threads_x;
    int grid_y = (TEST_HEIGHT + gpu_threads_y - 1) / gpu_threads_y;
    int total_blocks = grid_x * grid_y;
    int total_gpu_threads = threads_per_block * total_blocks;
    
    printf("GPU Configuration Analysis:\n");
    printf("  Block size: %dx%d (%d threads per block)\n", gpu_threads_x, gpu_threads_y, threads_per_block);
    printf("  Grid size: %dx%d (%d blocks total)\n", grid_x, grid_y, total_blocks);
    printf("  Total GPU threads: %d\n", total_gpu_threads);
    printf("  Threads per warp: 32 (GPU has %d warps per block)\n", (threads_per_block + 31) / 32);
    printf("CPU Configuration: %d OpenMP threads\n", num_threads);
    
    // Allocate result buffers
    Vec3* cpu_serial_result = (Vec3*)malloc(TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3));
    Vec3* hybrid_result = (Vec3*)malloc(TEST_WIDTH * TEST_HEIGHT * sizeof(Vec3));
    
    // Setup camera - EXACT MATCH WITH CUDA REFERENCE
    Camera_Device cam;
    cam.origin = vec3_create(0, 0, 0);
    cam.lower_left_corner = vec3_create(-2, -1.5, -1);
    cam.horizontal = vec3_create(4, 0, 0);
    cam.vertical = vec3_create(0, 3, 0);
    
    // Create test scene - EXACT MATCH WITH CUDA REFERENCE
    SphereData_Device test_spheres[3];
    // Ground sphere
    test_spheres[0].center = vec3_create(0, -100.5f, -1);
    test_spheres[0].radius = 100.0f;
    test_spheres[0].material = material_lambertian_create_host(vec3_create(0.5f, 0.5f, 0.5f));
    
    // Center sphere
    test_spheres[1].center = vec3_create(0, 0, -1);
    test_spheres[1].radius = 0.5f;
    test_spheres[1].material = material_lambertian_create_host(vec3_create(0.7f, 0.3f, 0.3f));
    
    // Left sphere
    test_spheres[2].center = vec3_create(-1, 0, -1);
    test_spheres[2].radius = 0.5f;
    test_spheres[2].material = material_lambertian_create_host(vec3_create(0.8f, 0.8f, 0.0f));
    
    // Verify scene consistency with CUDA reference
    verify_scene_consistency(test_spheres, 3, cam);
    
    unsigned int base_seed = 1234; // Fixed seed for reproducibility - SAME AS CUDA REFERENCE
    
    // CPU Serial timing
    double cpu_serial_total_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        render_cpu_serial(cpu_serial_result, TEST_WIDTH, TEST_HEIGHT, test_spheres, 3, cam, base_seed);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        cpu_serial_total_time += time_ms;
    }
    double avg_cpu_serial_time = cpu_serial_total_time / NUM_ITERATIONS;
    
    // Hybrid timing (CPU OpenMP + GPU combined)
    double hybrid_total_time = 0.0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        render_hybrid_combined(hybrid_result, TEST_WIDTH, TEST_HEIGHT, test_spheres, 3, cam, base_seed, num_threads, gpu_threads_x, gpu_threads_y);
        auto end = std::chrono::high_resolution_clock::now();
        
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        hybrid_total_time += time_ms;
    }
    double avg_hybrid_time = hybrid_total_time / NUM_ITERATIONS;
    
    // Calculate accuracy metrics (Serial vs Hybrid)
    double mse_hybrid_vs_serial = calculate_mse(hybrid_result, cpu_serial_result, TEST_WIDTH, TEST_HEIGHT);
    double rmse_hybrid_vs_serial = calculate_rmse(hybrid_result, cpu_serial_result, TEST_WIDTH, TEST_HEIGHT);
    
    // Calculate speedup
    double hybrid_speedup = avg_cpu_serial_time / avg_hybrid_time;
    
    // Calculate PSNR
    double psnr_hybrid = 10.0 * log10(3.0 / mse_hybrid_vs_serial);
    
    // Print results
    printf("Serial Time: %.2f ms\n", avg_cpu_serial_time);
    printf("Hybrid Time (%d CPU threads + %dx%d GPU): %.2f ms\n", num_threads, gpu_threads_x, gpu_threads_y, avg_hybrid_time);
    printf("Hybrid Speedup: %.2fx\n", hybrid_speedup);
    printf("Accuracy - RMSE: %.6f, PSNR: %.2f dB\n", rmse_hybrid_vs_serial, psnr_hybrid);
    
    // Write to CSV
    fprintf(csv_file, "%d,%dx%d,%.2f,%.2f,%.2f,%.6f,%.6f,%.2f\n",
            num_threads, gpu_threads_x, gpu_threads_y,
            avg_cpu_serial_time, avg_hybrid_time, hybrid_speedup,
            mse_hybrid_vs_serial, rmse_hybrid_vs_serial, psnr_hybrid);
    
    // Cleanup
    free(cpu_serial_result);
    free(hybrid_result);
    
    printf("----------------------\n");
}

int main() {
    printf("Hybrid Ray Tracing Performance Analysis\n");
    printf("=======================================\n");
    printf("Image Resolution: %dx%d\n", TEST_WIDTH, TEST_HEIGHT);
    printf("Number of iterations per test: %d\n", NUM_ITERATIONS);
    printf("Samples per pixel: %d\n", SAMPLES_PER_PIXEL);
    
    // Query device properties
    cudaDeviceProp prop;
    int device;
    gpuErrchk(cudaGetDevice(&device));
    gpuErrchk(cudaGetDeviceProperties(&prop, device));
    
    printf("\nSystem Information:\n");
    printf("GPU Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max block dimensions: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max grid dimensions: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    printf("Warp size: %d\n", prop.warpSize);
    printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Total max GPU threads: %d\n", prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor);
    printf("CPU threads available: %d\n", omp_get_max_threads());
    printf("==================================================\n");
    
    // Initialize CUDA
    gpuErrchk(cudaFree(0));
    size_t stack_size = 16384;
    gpuErrchk(cudaDeviceSetLimit(cudaLimitStackSize, stack_size));
    
    // Test different configurations - EXACT MATCH WITH CUDA REFERENCE
    // Force OpenMP to use available threads
    omp_set_num_threads(8); // Set to system max threads
    int max_threads = omp_get_max_threads();
    printf("Detected max OpenMP threads: %d\n", max_threads);
    
    std::vector<int> thread_counts = {1, 2, 4, 8}; // OpenMP thread counts based on system capability
    // Filter to only include valid thread counts
    std::vector<int> valid_thread_counts;
    for (int t : thread_counts) {
        if (t <= max_threads) {
            valid_thread_counts.push_back(t);
        }
    }
    thread_counts = valid_thread_counts;
    
    // GPU thread configurations - IDENTICAL TO CUDA REFERENCE FOR DIRECT COMPARISON
    // These are the EXACT same configurations tested in the CUDA-only analyzer
    std::vector<std::pair<int, int>> gpu_configs = {
        {1, 1},     // Serial-like execution (1 thread per block) - many blocks
        {2, 2},     // 4 threads per block
        {4, 4},     // 16 threads per block
        {8, 8},     // 64 threads per block (2 warps)
        {16, 16},   // 256 threads per block (8 warps) - good for MX330
        {16, 32},   // 512 threads per block (16 warps) - good for MX330
        {32, 16},   // 512 threads per block (16 warps, different aspect ratio)
        {8, 16},    // 128 threads per block (4 warps)
        {16, 8},    // 128 threads per block (4 warps, different aspect ratio)
        {4, 16},    // 64 threads per block (2 warps, tall)
        {16, 4},    // 64 threads per block (2 warps, wide)
        {32, 8},    // 256 threads per block (8 warps, wide)
        {8, 32},    // 256 threads per block (8 warps, tall)
        {32, 4},    // 128 threads per block (4 warps, very wide)
        {4, 32},    // 128 threads per block (4 warps, very tall)
        // Avoiding 32x32 and other high thread counts as they may cause issues on MX330
    };
    
    // Open CSV file for results
    FILE* csv_file = fopen("hybrid_performance_analysis.csv", "w");
    fprintf(csv_file, "CPU_Threads,GPU_Config,Serial_Time_ms,Parallel_Time_ms,Speedup,MSE,RMSE,PSNR_dB\n");
    
    printf("\n=== Hybrid Performance Testing ===\n");
    printf("Testing hybrid CPU+GPU performance with different configurations...\n");
    
    // Test hybrid performance
    for (int cpu_threads : thread_counts) {
        for (const auto& gpu_config : gpu_configs) {
            int threads_per_block = gpu_config.first * gpu_config.second;
            
            // Check device constraints - same as CUDA reference
            bool valid_config = true;
            valid_config &= (threads_per_block <= prop.maxThreadsPerBlock);
            valid_config &= (gpu_config.first <= prop.maxThreadsDim[0]);
            valid_config &= (gpu_config.second <= prop.maxThreadsDim[1]);
            
            // For MX330, avoid very high thread counts per block as they may not perform well
            if (prop.multiProcessorCount <= 4) { // Low-end GPU
                valid_config &= (threads_per_block <= 512); // Conservative limit
            }
            
            if (valid_config) {
                run_hybrid_performance_test(cpu_threads, gpu_config.first, gpu_config.second, csv_file);
            } else {
                printf("Skipping CPU:%d + GPU:%dx%d configuration (not suitable for device: %d threads per block)\n", 
                       cpu_threads, gpu_config.first, gpu_config.second, threads_per_block);
            }
        }
    }
    
    fclose(csv_file);
    
    // Generate comprehensive summary report
    printf("\n=== COMPREHENSIVE ANALYSIS SUMMARY ===\n");
    printf("Performance analysis complete!\n");
    printf("Results saved to: hybrid_performance_analysis.csv\n");
    
    return 0;
}
