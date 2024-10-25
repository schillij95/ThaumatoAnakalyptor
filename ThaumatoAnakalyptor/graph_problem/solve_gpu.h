// solve_gpu.h
#ifndef SOLVE_GPU_H
#define SOLVE_GPU_H

#include <vector>
#include "node_structs.h"  // Assuming Node and Edge are declared here
#include <string>

// Declaration of the GPU solver function
void solve_gpu_session(std::vector<Node>& graph, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, int num_iterations, float o, float spring_factor, int steps, std::vector<float>& spring_constants, std::vector<size_t>& valid_indices, int iterations_factor, float o_factor, int estimated_windings, const std::string& histogram_dir);
// void solve_gpu(std::vector<Node>& graph, int i, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, bool first_estimated_iteration, int estimated_windings, Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, int num_nodes);

float min_f_star(const std::vector<Node>& graph, bool use_gt = false);
float max_f_star(const std::vector<Node>& graph, bool use_gt = false);

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000);
void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps = 10);

#endif // SOLVE_GPU_H
