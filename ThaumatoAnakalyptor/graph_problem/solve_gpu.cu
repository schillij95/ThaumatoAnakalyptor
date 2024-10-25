/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/
#include "solve_gpu.h"
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <stack>
#include <cmath>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <random>
#include <queue>
#include <numeric>


// Kernel to update nodes on the GPU
__global__ void update_nodes_kernel(Node* d_graph, size_t* d_valid_indices, float o, float spring_constant, int num_valid_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    float sum_w_f_tilde_k = 0.0f;
    float sum_w = 0.0f;

    Edge* edges = node.edges;
    for (int j = 0; j < node.num_edges; ++j) {
        const Edge& edge = edges[j];
        if (d_graph[edge.target_node].deleted) continue;

        sum_w_f_tilde_k += edge.certainty * (d_graph[edge.target_node].f_tilde - spring_constant * edge.k);
        sum_w += edge.certainty;
    }

    node.f_star = (sum_w_f_tilde_k + o * node.f_tilde) / (sum_w + o);
}

// Kernel to update f_tilde with f_star on the GPU
__global__ void update_f_tilde_kernel(Node* d_graph, size_t* d_valid_indices, int num_valid_nodes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_valid_nodes) return;

    size_t i = d_valid_indices[idx];
    Node& node = d_graph[i];
    if (node.deleted) return;

    // Update f_tilde with the computed f_star
    node.f_tilde = node.f_star;
}

// Copy edges from CPU to GPU with batched allocation
void copy_edges_to_gpu(Node* h_graph, Node* d_graph, size_t num_nodes, Edge** d_all_edges_ptr) {
    size_t total_edges = 0;

    // Step 1: Calculate the total number of edges
    for (size_t i = 0; i < num_nodes; ++i) {
        total_edges += h_graph[i].num_edges;
    }

    // Step 2: Allocate memory for all edges at once on the GPU
    Edge* d_all_edges;
    cudaMalloc(&d_all_edges, total_edges * sizeof(Edge));
    *d_all_edges_ptr = d_all_edges;  // Store the pointer for later use when freeing

    // Step 3: Copy all edges in one go
    Edge* h_all_edges = new Edge[total_edges];  // Temporary array to hold all edges
    size_t offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].num_edges > 0) {
            memcpy(&h_all_edges[offset], h_graph[i].edges, h_graph[i].num_edges * sizeof(Edge));
            offset += h_graph[i].num_edges;
        }
    }

    cudaMemcpy(d_all_edges, h_all_edges, total_edges * sizeof(Edge), cudaMemcpyHostToDevice);
    delete[] h_all_edges;

    // Step 4: Update the d_graph[i].edges pointers
    offset = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (h_graph[i].num_edges > 0) {
            // Use a device pointer (d_all_edges + offset), not host pointer (&d_all_edges[offset])
            Edge* d_edges_offset = d_all_edges + offset;
            cudaMemcpyAsync(&d_graph[i].edges, &d_edges_offset, sizeof(Edge*), cudaMemcpyHostToDevice); // many small transfers, but async
            offset += h_graph[i].num_edges;
        }
    }
    cudaDeviceSynchronize();  // Synchronize at the end, after all async transfers
}

// Function to free the memory
void free_edges_from_gpu(Edge* d_all_edges) {
    cudaFree(d_all_edges); // Free the entire batch of edges in one go
}

__device__ float atomicMinFloat(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fminf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__device__ float atomicMaxFloat(float* address, float value) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(fmaxf(value, __int_as_float(assumed))));
    } while (assumed != old);

    return __int_as_float(old);
}

__global__ void min_f_star_kernel(Node* graph, float* min_f_star_out, int num_nodes) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with max float value
    sdata[tid] = FLT_MAX;

    if (idx < num_nodes && !graph[idx].deleted) {
        sdata[tid] = graph[idx].f_star;
    }

    __syncthreads();

    // Perform parallel reduction to find the minimum f_star
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result from block to global memory
    if (tid == 0) {
        atomicMinFloat(min_f_star_out, sdata[0]);  // Replace atomicMin with atomicMinFloat
    }
}

__global__ void max_f_star_kernel(Node* graph, float* max_f_star_out, int num_nodes) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Initialize shared memory with min float value
    sdata[tid] = -FLT_MAX;

    if (idx < num_nodes && !graph[idx].deleted) {
        sdata[tid] = graph[idx].f_star;
    }

    __syncthreads();

    // Perform parallel reduction to find the maximum f_star
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Write result from block to global memory
    if (tid == 0) {
        atomicMaxFloat(max_f_star_out, sdata[0]);  // Replace atomicMax with atomicMaxFloat
    }
}

float min_f_star(const std::vector<Node>& graph, bool use_gt) {
    float min_f = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            if (node.gt_f_star < min_f) {
                min_f = node.gt_f_star;
            }
        } else {
            if (node.f_star < min_f) {
                min_f = node.f_star;
            }
        }
    }

    return min_f;
}

float max_f_star(const std::vector<Node>& graph, bool use_gt) {
    float max_f = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            if (node.gt_f_star > max_f) {
                max_f = node.gt_f_star;
            }
        } else {
            if (node.f_star > max_f) {
                max_f = node.f_star;
            }
        }
    }

    return max_f;
}

std::pair<float, float> min_max_percentile_f_star(const std::vector<Node>& graph, float percentile, bool use_gt = false) {
    std::vector<float> f_star_values;
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (use_gt) {
            f_star_values.push_back(node.gt_f_star);
        } else {
            f_star_values.push_back(node.f_star);
        }
    }

    std::sort(f_star_values.begin(), f_star_values.end());

    size_t num_values = f_star_values.size();
    size_t min_index = static_cast<size_t>(std::floor(percentile * num_values));
    size_t max_index = static_cast<size_t>(std::floor((1.0f - percentile) * num_values));
    return std::make_pair(f_star_values[min_index], f_star_values[max_index]);
}

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename, int num_buckets) {
    // Find min and max f_star values
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);

    // Calculate bucket size
    float bucket_size = (max_f - min_f) / num_buckets;

    // Initialize the histogram with 0 counts
    std::vector<int> histogram(num_buckets, 0);

    // Fill the histogram
    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        int bucket_index = static_cast<int>((node.f_star - min_f) / bucket_size);
        if (bucket_index >= 0 && bucket_index < num_buckets) {
            histogram[bucket_index]++;
        }
    }

    // Create a blank image for the histogram with padding on the left
    int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
    int hist_h = 800;  // height of the histogram image
    int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel
    int left_padding = 50;  // Add 50 pixels of padding on the left side

    cv::Mat hist_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels and padding

    // Normalize the histogram to fit in the image
    int max_value = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < num_buckets; ++i) {
        histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
    }

    // Draw the histogram with left padding
    for (int i = 0; i < num_buckets; ++i) {
        cv::rectangle(hist_image, 
                      cv::Point(left_padding + i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
                      cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
                      cv::Scalar(0, 0, 0), 
                      cv::FILLED);
    }

    // Add x-axis labels
    std::string min_label = "Min: " + std::to_string(min_f);
    std::string max_label = "Max: " + std::to_string(max_f);
    cv::putText(hist_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(hist_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Save the histogram image to a file if string not empty
    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    // Display the histogram
    cv::imshow("Histogram of f_star values", hist_image);
    cv::waitKey(1);
}

void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps) {
    std::vector<cv::String> filenames;
    cv::glob(directory + "/*.png", filenames);

    // Sort the filenames in ascending order
    std::sort(filenames.begin(), filenames.end());

    if (filenames.empty()) {
        std::cerr << "No images found in directory: " << directory << std::endl;
        return;
    }

    // Read the first image to get the frame size
    cv::Mat first_image = cv::imread(filenames[0]);
    if (first_image.empty()) {
        std::cerr << "Error reading image: " << filenames[0] << std::endl;
        return;
    }

    // Create a VideoWriter object
    cv::VideoWriter video(output_file, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, first_image.size());

    for (const auto& file : filenames) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "Error reading image: " << file << std::endl;
            continue;
        }
        video.write(img);
    }

    video.release();
    std::cout << "Video created successfully: " << output_file << std::endl;
}

// Main GPU solver function
void solve_gpu(std::vector<Node>& graph, int i, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, float o, float spring_constant, int num_iterations, std::vector<size_t>& valid_indices, bool first_estimated_iteration, int estimated_windings, Node* d_graph, size_t* d_valid_indices, int num_valid_nodes, int num_nodes) {
    std::cout << "Solving on GPU..." << std::endl;
    // CUDA kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_valid_nodes + threadsPerBlock - 1) / threadsPerBlock;

    // Run the iterations
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Launch the kernel to update nodes
        update_nodes_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, o, spring_constant, num_valid_nodes);

        cudaError_t err = cudaGetLastError(); // Check for errors during kernel launch
        if (err != cudaSuccess) {
            std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize(); // Check for errors during kernel execution

        // Launch the kernel to update f_tilde with f_star
        update_f_tilde_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_graph, d_valid_indices, num_valid_nodes);

        err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA Synchronization error: " << cudaGetErrorString(err) << std::endl;
        }

        // Synchronize to ensure all threads are finished
        cudaDeviceSynchronize();

        // std::cout << "Iteration: " << iter << std::endl;

        if (iter % 100 == 0) {
            // Copy results back to the host
            cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);
            // Generate filename with zero padding
            std::ostringstream filename;
            filename << "histogram/histogram_" 
                    << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                    << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                    << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
            // Calculate and display the histogram of f_star values
            if (video_mode) {
                calculate_histogram(graph, filename.str());
            }

            // Print
            std::cout << "\rIteration: " << iter << std::flush;  // Updates the same line

            // escape if estimated windings reached
            if (first_estimated_iteration) {
                // float min_f = min_f_star(graph);
                // float max_f = max_f_star(graph);
                auto [min_percentile, max_percentile] = min_max_percentile_f_star(graph, 0.02f);
                std::cout << " Min percentile: " << min_percentile << ", Max percentile: " << max_percentile << std::endl;
                if (max_percentile - min_percentile > 1.0f * 360.0f * estimated_windings) {
                    break;
                }
            }
        }
    }
}

void solve_gpu_session(std::vector<Node>& graph, int edges_deletion_round, bool video_mode, int max_index_digits, int max_iter_digits, int num_iterations, float o, float spring_factor, int steps, std::vector<float>& spring_constants, std::vector<size_t>& valid_indices, int iterations_factor, float o_factor, int estimated_windings, const std::string& histogram_dir) {
    // Allocate space for min and max f_star values on the GPU
    size_t num_nodes = graph.size();
    size_t num_valid_nodes = valid_indices.size();
    // original edges pointers
    std::vector<Edge*> original_edges(num_nodes);
    for (size_t i = 0; i < num_nodes; ++i) {
        original_edges[i] = graph[i].edges;
    }

    // Allocate memory on the GPU
    Node* d_graph;
    size_t* d_valid_indices;
    cudaMalloc(&d_graph, num_nodes * sizeof(Node));
    cudaMalloc(&d_valid_indices, num_valid_nodes * sizeof(size_t));

    // Copy graph and valid indices to the GPU
    cudaMemcpy(d_graph, graph.data(), num_nodes * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_valid_indices, valid_indices.data(), num_valid_nodes * sizeof(size_t), cudaMemcpyHostToDevice);

    // Allocate and copy edges to GPU
    Edge* d_all_edges;
    copy_edges_to_gpu(graph.data(), d_graph, num_nodes, &d_all_edges);

    std::cout << "Copied data to GPU" << std::endl;
    
    // Solve for each spring constant
    for (int64_t i = -1; i < steps+1; ++i) {
        int num_iterations_iteration = num_iterations;
        float o_iteration = o;
        float spring_constant_iteration = i == -1 ? spring_constants[0] : spring_constants[i];
        if (i == -1 && edges_deletion_round == 0) {
            // Use a warmup iteration with 10x the spring constant
            num_iterations_iteration *= iterations_factor;
            o_iteration = o * o_factor;
            spring_constant_iteration = spring_factor;
        }
        // Skip the first iterations if the warmup is already done
        else if (i == -1) {
            // Skip the warmup iteration for subsequent rounds
            continue;
        }
        else if (i == steps && edges_deletion_round >= 1) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations_iteration *= 3.0f;
            spring_constant_iteration = 1.0f;
        }
        else if (i == steps) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations_iteration *= 1.5f;
        }
        std::cout << "Spring Constant " << i << ": " << std::setprecision(10) << spring_constant_iteration << std::endl;
        bool first_estimated_iteration = i == -1 && edges_deletion_round == 0 && estimated_windings > 0;

        // Run GPU solver
        solve_gpu(graph, i, edges_deletion_round, video_mode, max_index_digits, max_iter_digits, o_iteration, spring_constant_iteration, num_iterations_iteration, valid_indices, first_estimated_iteration, estimated_windings, d_graph, d_valid_indices, num_valid_nodes, num_nodes);
        
        // endline
        std::cout << std::endl;

        // After generating histograms, create a video from the images
        if (video_mode) {
            create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
        }
    }

    // Copy results back to the host
    cudaMemcpy(graph.data(), d_graph, num_nodes * sizeof(Node), cudaMemcpyDeviceToHost);

    // Restore the original edges pointers (no need to copy edges back)
    for (int i = 0; i < num_nodes; ++i) {
        // Just set the edges pointer back to its original location on the CPU
        graph[i].edges = original_edges[i]; // Assuming original_edges[i] points to the correct Edge array on the CPU
    }

    // Free GPU memory
    free_edges_from_gpu(d_all_edges);

    cudaFree(d_graph);
    cudaFree(d_valid_indices);
}