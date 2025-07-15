#ifndef HYBRID_RENDER_H
#define HYBRID_RENDER_H

#include "common_math.h"
#include "hittables.h"
#include "camera_shared.h"
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <SDL2/SDL.h>

// Hybrid rendering configuration
#define TILE_SIZE 64  // Size of each tile (64x64 pixels)
#define MAX_STREAMS 8 // Maximum number of CUDA streams

// Structure for tile information
typedef struct {
    int start_x, start_y;
    int end_x, end_y;
    int width, height;
    int stream_id;
} TileInfo;

// Hybrid rendering functions
void render_frame_hybrid(SDL_Renderer *renderer, SDL_Texture *texture, int width, int height);
void init_hybrid_rendering(int max_width, int max_height);
void cleanup_hybrid_rendering();

// CUDA kernel for tile-based rendering
__global__ void render_tile_kernel(Vec3* fb, int fb_width, int fb_height,
                                   int tile_start_x, int tile_start_y,
                                   int tile_end_x, int tile_end_y,
                                   Camera_Device cam,
                                   SphereData_Device* spheres, int num_spheres,
                                   CubeData_Device* cubes, int num_cubes,
                                   curandState *rand_states);

// Host-side ray tracing function for OpenMP fallback
Vec3 ray_color_host(const Ray* r,
                    const SphereData_Device* spheres, int num_spheres,
                    const CubeData_Device* cubes, int num_cubes,
                    int depth, unsigned int* seed);

// Host-side hit detection functions
bool world_hit_host(const SphereData_Device* spheres, int num_spheres,
                    const CubeData_Device* cubes, int num_cubes,
                    const Ray* r, float t_min, float t_max, HitRecord_Device* rec);

bool sphere_hit_host(const SphereData_Device* sphere, const Ray* r, 
                     float t_min, float t_max, HitRecord_Device* rec);

bool cube_hit_host(const CubeData_Device* cube, const Ray* r, 
                   float t_min, float t_max, HitRecord_Device* rec);

// Material functions for host
bool material_scatter_host(const Material_Device* self, const Ray* r_in, 
                          const HitRecord_Device* rec, Vec3* attenuation, 
                          Ray* scattered_ray, unsigned int* seed);

Vec3 material_emitted_host(const Material_Device* self, const HitRecord_Device* rec);

// Random number generation for host
float random_float_host(unsigned int* seed);
Vec3 random_unit_vector_host(unsigned int* seed);
Vec3 random_in_unit_sphere_host(unsigned int* seed);

#endif // HYBRID_RENDER_H
