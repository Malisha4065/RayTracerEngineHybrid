#include "../include/hybrid_render.h"
#include "../include/scene.h"
#include "../include/kernels.h"
#include "../include/cuda_utils.h"
#include "../include/materials.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <algorithm> // for std::min

// Global variables for hybrid rendering
static bool hybrid_initialized = false;

// Initialize hybrid rendering resources
void init_hybrid_rendering(int max_width, int max_height) {
    if (hybrid_initialized) return;
    
    // Initialize OpenMP
    omp_set_dynamic(0);
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    
    hybrid_initialized = true;
    printf("Hybrid rendering initialized with %d OpenMP threads\n", max_threads);
}

// Cleanup hybrid rendering resources
void cleanup_hybrid_rendering() {
    if (!hybrid_initialized) return;
    
    hybrid_initialized = false;
    printf("Hybrid rendering cleanup complete\n");
}

// We'll use the render_kernel_region directly from kernels.cu instead of duplicating code

// Host-side random number generation (simple LCG)
float random_float_host(unsigned int* seed) {
    *seed = *seed * 1664525u + 1013904223u;
    return ((float)(*seed) / 4294967296.0f);
}

Vec3 random_unit_vector_host(unsigned int* seed) {
    float theta = 2.0f * M_PI * random_float_host(seed);
    float phi = acosf(1.0f - 2.0f * random_float_host(seed));
    float sin_phi = sinf(phi);
    return vec3_create(sin_phi * cosf(theta), sin_phi * sinf(theta), cosf(phi));
}

Vec3 random_in_unit_sphere_host(unsigned int* seed) {
    Vec3 p;
    do {
        p = vec3_create(2.0f * random_float_host(seed) - 1.0f,
                       2.0f * random_float_host(seed) - 1.0f,
                       2.0f * random_float_host(seed) - 1.0f);
    } while (vec3_length_squared(p) >= 1.0f);
    return p;
}

// Host-side hit record helper
void hit_record_set_face_normal_host(HitRecord_Device* rec, const Ray* r, const Vec3* outward_normal) {
    rec->front_face = vec3_dot(r->direction, *outward_normal) < 0;
    rec->normal = rec->front_face ? *outward_normal : vec3_scale(*outward_normal, -1.0f);
}

// Host-side material functions
bool material_scatter_host(const Material_Device* self, const Ray* r_in, 
                          const HitRecord_Device* rec, Vec3* attenuation, 
                          Ray* scattered_ray, unsigned int* seed) {
    switch (self->type) {
        case MAT_LAMBERTIAN_DEVICE: {
            Vec3 scatter_direction = vec3_add(rec->normal, random_unit_vector_host(seed));
            if (vec3_length_squared(scatter_direction) < 1e-8f) {
                scatter_direction = rec->normal;
            }
            *scattered_ray = ray_create(rec->p, vec3_normalize(scatter_direction));
            *attenuation = self->albedo;
            return true;
        }
        case MAT_METAL_DEVICE: {
            Vec3 reflected_dir = vec3_reflect(vec3_normalize(r_in->direction), rec->normal);
            Vec3 fuzzed_dir = vec3_add(reflected_dir, vec3_scale(random_in_unit_sphere_host(seed), self->fuzz));
            *scattered_ray = ray_create(rec->p, vec3_normalize(fuzzed_dir));
            *attenuation = self->albedo;
            return (vec3_dot(scattered_ray->direction, rec->normal) > 0.0f);
        }
        case MAT_EMISSIVE_DEVICE:
            return false;
        default:
            return false;
    }
}

Vec3 material_emitted_host(const Material_Device* self, const HitRecord_Device* rec) {
    (void)rec;
    if (self->type == MAT_EMISSIVE_DEVICE) {
        return self->emission;
    }
    return vec3_create(0,0,0);
}

// Host-side hit detection functions
bool sphere_hit_host(const SphereData_Device* sphere, const Ray* r, 
                     float t_min, float t_max, HitRecord_Device* rec) {
    Vec3 oc = vec3_sub(r->origin, sphere->center);
    float a = vec3_length_squared(r->direction);
    float half_b = vec3_dot(oc, r->direction);
    float c = vec3_length_squared(oc) - sphere->radius * sphere->radius;
    float discriminant = half_b * half_b - a * c;
    
    if (discriminant < 0.0f) return false;
    
    float sqrtd = sqrtf(discriminant);
    float root = (-half_b - sqrtd) / a;
    if (root < t_min || root > t_max) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || root > t_max) {
            return false;
        }
    }
    
    rec->t = root;
    rec->p = ray_at(*r, rec->t);
    Vec3 outward_normal = vec3_div(vec3_sub(rec->p, sphere->center), sphere->radius);
    hit_record_set_face_normal_host(rec, r, &outward_normal);
    rec->material = sphere->material;
    return true;
}

bool cube_hit_host(const CubeData_Device* cube, const Ray* r, 
                   float t_min, float t_max, HitRecord_Device* rec) {
    Vec3 inv_dir = vec3_create(1.0f / r->direction.x, 1.0f / r->direction.y, 1.0f / r->direction.z);
    
    float t0x = (cube->min_corner.x - r->origin.x) * inv_dir.x;
    float t1x = (cube->max_corner.x - r->origin.x) * inv_dir.x;
    if (inv_dir.x < 0.0f) { float temp = t0x; t0x = t1x; t1x = temp; }
    
    float t0y = (cube->min_corner.y - r->origin.y) * inv_dir.y;
    float t1y = (cube->max_corner.y - r->origin.y) * inv_dir.y;
    if (inv_dir.y < 0.0f) { float temp = t0y; t0y = t1y; t1y = temp; }
    
    float t0z = (cube->min_corner.z - r->origin.z) * inv_dir.z;
    float t1z = (cube->max_corner.z - r->origin.z) * inv_dir.z;
    if (inv_dir.z < 0.0f) { float temp = t0z; t0z = t1z; t1z = temp; }
    
    float t_enter = fmaxf(fmaxf(t0x, t0y), t0z);
    float t_exit = fminf(fminf(t1x, t1y), t1z);
    
    if (t_enter >= t_exit || t_exit < t_min || t_enter > t_max) {
        return false;
    }
    
    float t_hit = (t_enter > t_min) ? t_enter : t_exit;
    if (t_hit < t_min || t_hit > t_max) return false;
    
    rec->t = t_hit;
    rec->p = ray_at(*r, rec->t);
    rec->material = cube->material;
    
    // Determine normal based on which face was hit
    Vec3 center = vec3_scale(vec3_add(cube->min_corner, cube->max_corner), 0.5f);
    Vec3 local_point = vec3_sub(rec->p, center);
    Vec3 size = vec3_sub(cube->max_corner, cube->min_corner);
    Vec3 abs_local = vec3_create(fabsf(local_point.x), fabsf(local_point.y), fabsf(local_point.z));
    Vec3 half_size = vec3_scale(size, 0.5f);
    
    Vec3 outward_normal;
    if (abs_local.x > abs_local.y && abs_local.x > abs_local.z) {
        outward_normal = vec3_create((local_point.x > 0) ? 1.0f : -1.0f, 0.0f, 0.0f);
    } else if (abs_local.y > abs_local.z) {
        outward_normal = vec3_create(0.0f, (local_point.y > 0) ? 1.0f : -1.0f, 0.0f);
    } else {
        outward_normal = vec3_create(0.0f, 0.0f, (local_point.z > 0) ? 1.0f : -1.0f);
    }
    
    hit_record_set_face_normal_host(rec, r, &outward_normal);
    return true;
}

bool world_hit_host(const SphereData_Device* spheres, int num_spheres,
                    const CubeData_Device* cubes, int num_cubes,
                    const Ray* r, float t_min, float t_max, HitRecord_Device* rec) {
    HitRecord_Device temp_rec;
    bool hit_anything = false;
    float closest_so_far = t_max;
    
    // Check spheres in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_spheres; i++) {
        HitRecord_Device local_rec;
        if (sphere_hit_host(&spheres[i], r, t_min, closest_so_far, &local_rec)) {
            #pragma omp critical
            {
                if (local_rec.t < closest_so_far) {
                    hit_anything = true;
                    closest_so_far = local_rec.t;
                    temp_rec = local_rec;
                }
            }
        }
    }
    
    // Check cubes in parallel
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_cubes; i++) {
        HitRecord_Device local_rec;
        if (cube_hit_host(&cubes[i], r, t_min, closest_so_far, &local_rec)) {
            #pragma omp critical
            {
                if (local_rec.t < closest_so_far) {
                    hit_anything = true;
                    closest_so_far = local_rec.t;
                    temp_rec = local_rec;
                }
            }
        }
    }
    
    if (hit_anything) {
        *rec = temp_rec;
    }
    return hit_anything;
}

// Host-side ray color function
Vec3 ray_color_host(const Ray* r,
                    const SphereData_Device* spheres, int num_spheres,
                    const CubeData_Device* cubes, int num_cubes,
                    int depth, unsigned int* seed) {
    HitRecord_Device rec;
    
    if (depth <= 0) {
        return vec3_create(0, 0, 0);
    }
    
    if (world_hit_host(spheres, num_spheres, cubes, num_cubes, r, 0.001f, INFINITY, &rec)) {
        Ray scattered_ray;
        Vec3 attenuation;
        Vec3 emitted_light = material_emitted_host(&rec.material, &rec);
        
        if (material_scatter_host(&rec.material, r, &rec, &attenuation, &scattered_ray, seed)) {
            Vec3 scattered_color = ray_color_host(&scattered_ray, spheres, num_spheres, cubes, num_cubes, depth - 1, seed);
            Vec3 final_color;
            final_color.x = emitted_light.x + attenuation.x * scattered_color.x;
            final_color.y = emitted_light.y + attenuation.y * scattered_color.y;
            final_color.z = emitted_light.z + attenuation.z * scattered_color.z;
            return final_color;
        } else {
            return emitted_light;
        }
    }
    
    // Background
    Vec3 unit_direction = vec3_normalize(r->direction);
    float t = 0.5f * (unit_direction.y + 1.0f);
    Vec3 white = vec3_create(1.0f, 1.0f, 1.0f);
    Vec3 blue = vec3_create(0.5f, 0.7f, 1.0f);
    return vec3_add(vec3_scale(white, 1.0f - t), vec3_scale(blue, t));
}

// Performance tracking for dynamic load balancing
static float gpu_split_ratio = 0.7f; // GPU handles 70% of work initially
static double last_gpu_time = 0.0;
static double last_cpu_time = 0.0;
static int next_tile_atomic = 0; // Atomic counter for work distribution

// Enhanced tile-based hybrid rendering function
void render_frame_hybrid(SDL_Renderer *renderer, SDL_Texture *texture, int width, int height) {
    // Update resolution if it changed
    if (width != g_current_width || height != g_current_height) {
        resize_gpu_buffers(width, height);
    }
    
    // Initialize hybrid rendering if not done
    if (!hybrid_initialized) {
        init_hybrid_rendering(width, height);
    }
    
    // Setup camera
    camera_init_host(&d_camera, g_camera_pos_host, g_camera_lookat_host, g_camera_vup_host, g_fov_y_degrees_host, (float)width / height);
    
    printf("Using TRUE HYBRID rendering (GPU+CPU concurrent)\n");
    
    // Calculate tile grid
    int tile_size_x = TILE_SIZE;
    int tile_size_y = TILE_SIZE;
    int num_tiles_x = (width + tile_size_x - 1) / tile_size_x;
    int num_tiles_y = (height + tile_size_y - 1) / tile_size_y;
    int total_tiles = num_tiles_x * num_tiles_y;
    
    // Reset atomic counter for work distribution
    next_tile_atomic = 0;
    
    printf("Image size: %dx%d, Tile size: %dx%d, Total tiles: %d\n", 
           width, height, tile_size_x, tile_size_y, total_tiles);
    
    // Allocate host memory for final image
    Vec3* h_pixels = (Vec3*)malloc(width * height * sizeof(Vec3));
    if (!h_pixels) {
        printf("Error: Failed to allocate host memory\n");
        return;
    }
    
    // Create events for GPU timing
    cudaEvent_t gpu_start_event, gpu_end_event;
    cudaEventCreate(&gpu_start_event);
    cudaEventCreate(&gpu_end_event);
    
    // Split tiles between GPU and CPU based on current performance ratio
    int gpu_tiles = (int)(total_tiles * gpu_split_ratio);
    int cpu_tiles = total_tiles - gpu_tiles;
    
    // Ensure minimum work for both processors
    if (gpu_tiles < 1) gpu_tiles = 1;
    if (cpu_tiles < 1) cpu_tiles = 1;
    
    // Adjust tile count
    if (gpu_tiles > total_tiles) gpu_tiles = total_tiles;
    cpu_tiles = total_tiles - gpu_tiles;
    
    printf("GPU handling %d tiles, CPU handling %d tiles (ratio: %.2f)\n", 
           gpu_tiles, cpu_tiles, gpu_split_ratio);
    
    // Start timing
    cudaEventRecord(gpu_start_event, 0);
    double cpu_start_time = omp_get_wtime();
    
    // Process tiles in parallel with both GPU and CPU
    #pragma omp parallel sections
    {
        // GPU SECTION
        #pragma omp section
        {
            for (int i = 0; i < gpu_tiles; i++) {
                // Get next tile to process
                int tile_idx;
                #pragma omp atomic capture
                tile_idx = next_tile_atomic++;
                
                if (tile_idx >= total_tiles) continue;
                
                // Calculate tile coordinates
                int tile_x = tile_idx % num_tiles_x;
                int tile_y = tile_idx / num_tiles_x;
                
                int tile_start_x = tile_x * tile_size_x;
                int tile_start_y = tile_y * tile_size_y;
                int tile_end_x = std::min(tile_start_x + tile_size_x, width);
                int tile_end_y = std::min(tile_start_y + tile_size_y, height);
                
                // Launch kernel for this tile
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks(
                    (tile_end_x - tile_start_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (tile_end_y - tile_start_y + threadsPerBlock.y - 1) / threadsPerBlock.y
                );
                
                render_kernel_region<<<numBlocks, threadsPerBlock>>>(
                    d_pixel_data, width, height,
                    tile_start_x, tile_start_y, 
                    tile_end_x, tile_end_y,
                    d_camera, d_spheres_data, h_num_spheres,
                    d_cubes_data, h_num_cubes, d_rand_states
                );
            }
            
            // Wait for GPU to complete
            cudaDeviceSynchronize();
            
            // Copy GPU results back to host
            cudaMemcpy(h_pixels, d_pixel_data, width * height * sizeof(Vec3), cudaMemcpyDeviceToHost);
            
            // Record GPU time
            cudaEventRecord(gpu_end_event, 0);
            cudaEventSynchronize(gpu_end_event);
            float gpu_time_ms;
            cudaEventElapsedTime(&gpu_time_ms, gpu_start_event, gpu_end_event);
            last_gpu_time = gpu_time_ms / 1000.0; // Convert to seconds
            
            // Dynamic work stealing - process more tiles if available
            while (true) {
                int tile_idx;
                #pragma omp atomic capture
                tile_idx = next_tile_atomic++;
                
                if (tile_idx >= total_tiles) break;
                
                // Calculate tile coordinates
                int tile_x = tile_idx % num_tiles_x;
                int tile_y = tile_idx / num_tiles_x;
                
                int tile_start_x = tile_x * tile_size_x;
                int tile_start_y = tile_y * tile_size_y;
                int tile_end_x = std::min(tile_start_x + tile_size_x, width);
                int tile_end_y = std::min(tile_start_y + tile_size_y, height);
                
                // Launch kernel for this tile
                dim3 threadsPerBlock(16, 16);
                dim3 numBlocks(
                    (tile_end_x - tile_start_x + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (tile_end_y - tile_start_y + threadsPerBlock.y - 1) / threadsPerBlock.y
                );
                
                render_kernel_region<<<numBlocks, threadsPerBlock>>>(
                    d_pixel_data, width, height,
                    tile_start_x, tile_start_y, 
                    tile_end_x, tile_end_y,
                    d_camera, d_spheres_data, h_num_spheres,
                    d_cubes_data, h_num_cubes, d_rand_states
                );
                
                // Wait for this tile to complete
                cudaDeviceSynchronize();
                
                // Copy just this tile's results back
                for (int y = tile_start_y; y < tile_end_y; y++) {
                    cudaMemcpy(
                        &h_pixels[y * width + tile_start_x], 
                        &d_pixel_data[y * width + tile_start_x], 
                        (tile_end_x - tile_start_x) * sizeof(Vec3), 
                        cudaMemcpyDeviceToHost
                    );
                }
            }
        }

        // CPU SECTION
        #pragma omp section
        {
            #pragma omp parallel
            {
                // Each thread gets its own random seed
                unsigned int seed = (unsigned int)(time(NULL) + omp_get_thread_num() * 1000);
                
                // Process CPU tiles in parallel
                #pragma omp for schedule(dynamic, 1)
                for (int i = 0; i < cpu_tiles; i++) {
                    int tile_idx;
                    #pragma omp atomic capture
                    tile_idx = next_tile_atomic++;
                    
                    if (tile_idx >= total_tiles) continue;
                    
                    // Calculate tile coordinates
                    int tile_x = tile_idx % num_tiles_x;
                    int tile_y = tile_idx / num_tiles_x;
                    
                    int tile_start_x = tile_x * tile_size_x;
                    int tile_start_y = tile_y * tile_size_y;
                    int tile_end_x = std::min(tile_start_x + tile_size_x, width);
                    int tile_end_y = std::min(tile_start_y + tile_size_y, height);
                    
                    // Render this tile
                    for (int y = tile_start_y; y < tile_end_y; y++) {
                        for (int x = tile_start_x; x < tile_end_x; x++) {
                            Vec3 pixel_color = vec3_create(0,0,0);
                            
                            for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
                                float u = (float)(x + ((SAMPLES_PER_PIXEL > 1) ? random_float_host(&seed) : 0.5f)) / (width - 1);
                                float v_cam = (float)((height - 1 - y) + ((SAMPLES_PER_PIXEL > 1) ? random_float_host(&seed) : 0.5f)) / (height - 1);
                                
                                Ray r = camera_get_ray_host(&d_camera, u, v_cam);
                                pixel_color = vec3_add(pixel_color, ray_color_host(&r, h_spheres_data, h_num_spheres, h_cubes_data, h_num_cubes, MAX_DEPTH, &seed));
                            }
                            pixel_color = vec3_div(pixel_color, (float)SAMPLES_PER_PIXEL);
                            
                            // Gamma correction
                            pixel_color.x = sqrtf(fmaxf(0.0f, pixel_color.x));
                            pixel_color.y = sqrtf(fmaxf(0.0f, pixel_color.y));
                            pixel_color.z = sqrtf(fmaxf(0.0f, pixel_color.z));
                            
                            // Store in host buffer
                            h_pixels[y * width + x] = pixel_color;
                        }
                    }
                }
            }
            
            double cpu_end_time = omp_get_wtime();
            last_cpu_time = cpu_end_time - cpu_start_time;
        }
    }
    
    // Dynamic load balancing: Adjust split ratio based on performance
    if (last_gpu_time > 0.0 && last_cpu_time > 0.0) {
        // Calculate pixels processed per second
        int gpu_pixels = gpu_tiles * tile_size_x * tile_size_y;
        int cpu_pixels = cpu_tiles * tile_size_x * tile_size_y;
        
        if (gpu_pixels > 0 && cpu_pixels > 0) {
            float gpu_pixels_per_sec = gpu_pixels / last_gpu_time;
            float cpu_pixels_per_sec = cpu_pixels / last_cpu_time;
            
            float total_performance = gpu_pixels_per_sec + cpu_pixels_per_sec;
            if (total_performance > 0) {
                // New split ratio based on relative performance
                float new_gpu_ratio = gpu_pixels_per_sec / total_performance;
                
                // Apply smoothed adjustment
                gpu_split_ratio = 0.7f * gpu_split_ratio + 0.3f * new_gpu_ratio;
                
                // Clamp to reasonable bounds
                if (gpu_split_ratio < 0.1f) gpu_split_ratio = 0.1f;
                if (gpu_split_ratio > 0.9f) gpu_split_ratio = 0.9f;
                
                printf("Performance: GPU=%.1f Mpix/s (%.1fs), CPU=%.1f Mpix/s (%.1fs), New ratio=%.2f\n", 
                       gpu_pixels_per_sec/1e6, last_gpu_time, cpu_pixels_per_sec/1e6, last_cpu_time, gpu_split_ratio);
            }
        }
    }
    
    // Convert to SDL format
    void *sdl_pixels_locked;
    int pitch;
    SDL_LockTexture(texture, NULL, &sdl_pixels_locked, &pitch);
    unsigned char *pixel_data_sdl = (unsigned char *)sdl_pixels_locked;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3 p_color = h_pixels[y * width + x];
            int ir = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.x)));
            int ig = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.y)));
            int ib = (int)(255.999f * fminf(1.0f, fmaxf(0.0f, p_color.z)));
            
            int index = y * pitch + x * 4;
            pixel_data_sdl[index + 0] = (unsigned char)ib; // Blue
            pixel_data_sdl[index + 1] = (unsigned char)ig; // Green
            pixel_data_sdl[index + 2] = (unsigned char)ir; // Red
            pixel_data_sdl[index + 3] = 255;               // Alpha
        }
    }
    
    // Cleanup
    free(h_pixels);
    cudaEventDestroy(gpu_start_event);
    cudaEventDestroy(gpu_end_event);
    SDL_UnlockTexture(texture);
    
    SDL_RenderCopy(renderer, texture, NULL, NULL);
    SDL_RenderPresent(renderer);
}