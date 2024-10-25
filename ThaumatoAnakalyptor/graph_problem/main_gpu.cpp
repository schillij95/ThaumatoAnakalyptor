/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/
#include "solve_gpu.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <stack>
#include <cmath>
#include <omp.h>
#include <limits>
#include <iomanip>
#include <filesystem>
#include <argparse.hpp>
#include <random>
#include <queue>
#include <numeric>

namespace fs = std::filesystem;

std::pair<std::vector<Node>, float> load_graph_from_binary(const std::string &file_name, bool clip_z = false, float z_min = 0.0f, float z_max = 0.0f, float same_winding_factor = 1.0f, bool fix_same_block_edges = false) {
    std::vector<Node> graph;
    std::ifstream infile(file_name, std::ios::binary);

    if (!infile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return std::make_pair(graph, 0.0f);
    }

    // Read the number of nodes
    unsigned int num_nodes;
    infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(unsigned int));
    std::cout << "Number of nodes in graph: " << num_nodes << std::endl;

    // Prepare the graph with empty nodes
    graph.resize(num_nodes);

    // Read each node's winding angle and other attributes
    for (unsigned int i = 0; i < num_nodes; ++i) {
        infile.read(reinterpret_cast<char*>(&graph[i].z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].gt), sizeof(bool));
        infile.read(reinterpret_cast<char*>(&graph[i].gt_f_star), sizeof(float));
        graph[i].f_tilde = graph[i].f_init;
        graph[i].f_star = graph[i].f_init;
        graph[i].deleted = false;
        graph[i].fixed = false;
    }
    std::cout << "Nodes loaded successfully." << std::endl;

    int count_same_block_edges = 0;

    // Read the adjacency list and edges for each node
    for (unsigned int i = 0; i < num_nodes; ++i) {
        unsigned int node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(unsigned int));

        unsigned int num_edges;
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(unsigned int));

        // Allocate memory for edges in the node
        graph[node_id].edges = new Edge[num_edges];
        graph[node_id].num_edges = num_edges;

        for (unsigned int j = 0; j < num_edges; ++j) {
            Edge& edge = graph[node_id].edges[j];
            infile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(unsigned int));
            infile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(float));
            
            // Set the certainty factored value
            edge.certainty_factored = edge.certainty;

            infile.read(reinterpret_cast<char*>(&edge.k), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(bool));
            edge.fixed = false;  // Default initialization

            // Clip Z coordinates if required
            if (clip_z) {
                if (graph[edge.target_node].z < z_min || graph[edge.target_node].z > z_max) {
                    graph[edge.target_node].deleted = true;
                    continue;
                }
                if (graph[node_id].z < z_min || graph[node_id].z > z_max) {
                    graph[node_id].deleted = true;
                    continue;
                }
            }

            // Fix edges between nodes in the same block if needed
            if (fix_same_block_edges) {
                if (std::abs(edge.k) > 180) {
                    edge.same_block = true;
                }
            }

            // Apply same winding factor if necessary
            if (edge.same_block) {
                edge.certainty *= same_winding_factor;
                edge.certainty_factored *= same_winding_factor;
                count_same_block_edges++;
                if (std::abs(edge.k) > 450) {
                    std::cout << "Edge with k > 450: " << edge.k << std::endl;
                }
                if (std::abs(edge.k) < 180) {
                    std::cout << "Edge with k < 180: " << edge.k << std::endl;
                }
            }
        }
    }

    std::cout << "Same block edges: " << count_same_block_edges << std::endl;
    std::cout << "Graph loaded successfully." << std::endl;

    // Find the largest certainty value for display
    float max_certainty = 0;
    for (const auto& node : graph) {
        if (node.deleted) continue;
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (edge.certainty > max_certainty) {
                max_certainty = edge.certainty;
            }
        }
    }

    std::cout << "Max Certainty: " << max_certainty << std::endl;

    infile.close();
    // Return the graph and the max certainty value
    return std::make_pair(graph, max_certainty);
}


void save_graph_to_binary(const std::string& file_name, const std::vector<Node>& graph) {
    std::ofstream outfile(file_name, std::ios::binary);

    if (!outfile.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Write the number of nodes
    unsigned int num_nodes = graph.size();
    outfile.write(reinterpret_cast<const char*>(&num_nodes), sizeof(unsigned int));

    // Write each node's f_star and deleted status
    for (const auto& node : graph) {
        outfile.write(reinterpret_cast<const char*>(&node.f_star), sizeof(float));
        outfile.write(reinterpret_cast<const char*>(&node.deleted), sizeof(bool));
    }

    outfile.close();
}

void set_z_range_graph(std::vector<Node>& graph, float z_min, float z_max) {
    for (auto& node : graph) {
        if (node.z < z_min || node.z > z_max) {
            node.deleted = true;
        }
    }
}

float closest_valid_winding_angle(float f_init, float f_target) {
    int x = static_cast<int>(std::round((f_target - f_init) / 360.0f));
    float result = f_init + x * 360.0f;
    if (std::abs(f_target - result) > 10.0f) {
        std::cout << "Difference between f_target and result: " << std::abs(f_target - result) << std::endl;
    }
    if (std::abs(x - (f_target - f_init) / 360.0f) > 1e-4) {
        std::cout << "Difference between x and (f_target - f_init) / 360.0f: " << std::abs(x - (f_target - f_init) / 360.0f) << std::endl;
        std::cout << "f_init: " << f_init << ", f_target: " << f_target << ", x: " << x << ", result: " << result << std::endl;
    }
    return result;
}

void dfs(size_t node_index, const std::vector<Node>& graph, std::vector<bool>& visited, std::vector<size_t>& component) {
    std::stack<size_t> stack;
    stack.push(node_index);
    visited[node_index] = true;

    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();
        component.push_back(current);

        // Traverse through the edges of the current node
        for (int i = 0; i < graph[current].num_edges; ++i) {
            const Edge& edge = graph[current].edges[i];

            // Skip the edge if the target node is deleted
            if (graph[edge.target_node].deleted) {
                continue;
            }

            // If the target node has not been visited, mark it and push it to the stack
            if (!visited[edge.target_node]) {
                visited[edge.target_node] = true;
                stack.push(edge.target_node);
            }
        }
    }
}

void find_largest_connected_component(std::vector<Node>& graph) {
    size_t num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);
    std::vector<size_t> largest_component;

    size_t initial_non_deleted = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            initial_non_deleted++;
        }
        if (!visited[i] && !graph[i].deleted) {
            std::vector<size_t> current_component;
            dfs(i, graph, visited, current_component);

            if (current_component.size() > largest_component.size()) {
                largest_component = current_component;
            }
        }
    }

    // Flag nodes not in the largest connected component as deleted
    std::vector<bool> in_largest_component(num_nodes, false);
    for (size_t node_index : largest_component) {
        in_largest_component[node_index] = true;
    }

    size_t remaining_nodes = 0;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!in_largest_component[i]) {
            graph[i].deleted = true;
        }
        if (!graph[i].deleted) {
            remaining_nodes++;
        }
    }
    std::cout << "Remaining nodes: " << remaining_nodes << " out of " << initial_non_deleted << " initial non deleted nodes, of total edge: " << num_nodes << std::endl;
}

using EdgeWithCertainty = std::pair<float, int>;  // {certainty, target_node}

void prim_mst_assign_f_star(size_t start_node, std::vector<Node>& graph, float scale) {
    size_t num_nodes = graph.size();
    std::vector<bool> in_mst(num_nodes, false);
    std::vector<float> min_k_delta(num_nodes, std::numeric_limits<float>::max());
    std::vector<size_t> parent(num_nodes, 0);
    std::vector<bool> valid(num_nodes, false);
    std::vector<float> k_values(num_nodes, 0.0);

    // Priority queue to pick the edge with the minimum k delta
    std::priority_queue<EdgeWithCertainty, std::vector<EdgeWithCertainty>, std::greater<EdgeWithCertainty>> pq;

    pq.push({0.0f, start_node});
    min_k_delta[start_node] = 0.0f;

    while (!pq.empty()) {
        size_t u = pq.top().second;
        pq.pop();

        if (in_mst[u]) continue;
        in_mst[u] = true;

        for (int i = 0; i < graph[u].num_edges; ++i) {
            const Edge& edge = graph[u].edges[i];

            if (graph[u].deleted) {
                continue;
            }

            size_t v = edge.target_node;
            if (graph[v].deleted) {
                continue;
            }

            // Calculate k_delta (difference between BP solution and estimated k from the graph)
            float k_delta = std::abs(scale * (graph[v].f_tilde - graph[u].f_tilde) - edge.k);
            if (edge.fixed && edge.certainty > 0.0f) {
                k_delta = -1.0f;
            }

            // Check if this edge has a smaller k_delta and update
            if (!in_mst[v] && k_delta < min_k_delta[v]) {
                min_k_delta[v] = k_delta;
                pq.push({k_delta, v});
                parent[v] = u;
                k_values[v] = edge.k;
                valid[v] = true;
            }
        }
    }

    // Set f_star for the root node (start_node)
    graph[start_node].f_tilde = graph[start_node].f_init;
    graph[start_node].f_star = graph[start_node].f_init;

    // Create children structures to store each node's children and their k_values
    std::vector<std::vector<size_t>> children(num_nodes);
    std::vector<std::vector<float>> children_k_values(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i]) {
            children[parent[i]].push_back(i);
            children_k_values[parent[i]].push_back(k_values[i]);
        }
    }

    // Traverse the MST to assign f_star values (DFS style traversal)
    std::stack<size_t> stack;
    stack.push(start_node);

    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();

        if (graph[current].deleted) {
            continue;
        }

        for (size_t i = 0; i < children[current].size(); ++i) {
            size_t child = children[current][i];
            if (graph[child].deleted) {
                continue;
            }

            // Calculate the f_star for the child node
            float k = children_k_values[current][i];
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + k);
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[child].f_star);

            graph[child].f_tilde = graph[child].f_star;

            // Push child onto the stack for further processing
            stack.push(child);
        }
    }
}

void assign_winding_angles(std::vector<Node>& graph, float scale) {
    size_t num_nodes = graph.size();
    
    // Find a non-deleted node in the largest connected component to start the MST
    size_t start_node = 0;
    bool found_start_node = false;
    for (size_t i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            found_start_node = true;
            break;
        }
    }

    if (!found_start_node) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
        return;
    }

    // Perform MST to assign f_star values
    prim_mst_assign_f_star(start_node, graph, scale);

    // check the winding angles on f_star
    for (size_t i = 0; i < num_nodes; ++i) {
        if (graph[i].deleted) {
            continue;
        }
        closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }
}

float certainty_factor(float error) { // certainty factor based on the error. exp (-) as function. 1 for 0 error, 0.1 for 360 error (1 winding off)
    // x = - log (0.1) / 360
    float winding_off_factor = 0.4f;
    float x = - std::log(winding_off_factor) / 360.0f;
    float factor = std::exp(-x * error);
    // clip to range winding_off_factor to 1
    if (factor < winding_off_factor) {
        factor = winding_off_factor;
    }
    return factor;
}

// float min_f_star(const std::vector<Node>& graph, bool use_gt = false) {
//     float min_f = std::numeric_limits<float>::max();

//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         if (use_gt) {
//             if (node.gt_f_star < min_f) {
//                 min_f = node.gt_f_star;
//             }
//         } else {
//             if (node.f_star < min_f) {
//                 min_f = node.f_star;
//             }
//         }
//     }

//     return min_f;
// }

// float max_f_star(const std::vector<Node>& graph, bool use_gt = false) {
//     float max_f = std::numeric_limits<float>::min();

//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         if (use_gt) {
//             if (node.gt_f_star > max_f) {
//                 max_f = node.gt_f_star;
//             }
//         } else {
//             if (node.f_star > max_f) {
//                 max_f = node.f_star;
//             }
//         }
//     }

//     return max_f;
// }

float calculate_scale(const std::vector<Node>& graph, int estimated_windings) {
    if (estimated_windings <= 0) {
        return 1.0f;
    }
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);
    return std::abs((360.0f * estimated_windings) / (max_f - min_f));
}

float exact_matching_score(std::vector<Node>& graph) {
    // Copy the graph and assign the closest valid winding angle to f_star based on f_init
    std::vector<Node> graph_copy = graph;

    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        graph_copy[i].f_star = closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }

    float score = 0.0f;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        for (int j = 0; j < graph_copy[i].num_edges; ++j) {
            const Edge& edge = graph_copy[i].edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            float diff = graph_copy[edge.target_node].f_star - graph_copy[i].f_star;
            if (std::abs(diff - edge.k) < 1e-5) {
                score += edge.certainty;
            }
        }
    }

    return score;
}

float approximate_matching_loss(const std::vector<Node>& graph, float a = 1.0f) {
    float loss = 0.0f;

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        for (int j = 0; j < node.num_edges; ++j) {
            const Edge& edge = node.edges[j];
            if (graph[edge.target_node].deleted) {
                continue;
            }
            float diff = graph[edge.target_node].f_star - node.f_star;
            float l = diff - edge.k;
            loss += edge.certainty * std::exp(-a * std::abs(l));
        }
    }

    return loss;
}

std::tuple<float, float, float, float, float, float> computeErrorStats(const std::vector<Node>& graph, const std::vector<size_t>& valid_gt_indices, int n_pairs = 10'000) {
    if (valid_gt_indices.size() < 2) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f); // Not enough valid ground truth nodes to form pairs
    }

    std::vector<float> errors; // To store individual errors
    
    std::random_device rd;
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine for random number generation
    std::uniform_int_distribution<> dis(0, valid_gt_indices.size() - 1);

    // Loop for a maximum of n_pairs random pairs
    for (int i = 0; i < n_pairs; ++i) {
        // Randomly pick two distinct nodes with valid ground truth
        size_t idx_i = valid_gt_indices[dis(gen)];
        size_t idx_j = valid_gt_indices[dis(gen)];

        // Ensure we don't compare a node with itself
        while (idx_i == idx_j) {
            idx_j = valid_gt_indices[dis(gen)];
        }

        const Node& node_i = graph[idx_i];
        const Node& node_j = graph[idx_j];

        // Compute the distance1 (ground truth distances)
        float dist1 = node_i.gt_f_star - node_j.gt_f_star;

        // Compute the distance2 (computed f_star distances)
        float dist2 = node_i.f_star - node_j.f_star;

        // Compute the absolute error
        float error = std::abs(dist1 - dist2);

        // Store the error
        errors.push_back(error);
    }

    // If no valid pairs are found, return all zeros to avoid division by zero
    if (errors.empty()) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Sort the error values to compute statistics
    std::sort(errors.begin(), errors.end());

    // Compute the mean
    float total_error = std::accumulate(errors.begin(), errors.end(), 0.0f);
    float mean_error = total_error / errors.size();

    // Min and Max
    float min_error = errors.front();
    float max_error = errors.back();

    // Quartiles
    float q1 = errors[errors.size() / 4];
    float median = errors[errors.size() / 2];
    float q3 = errors[(errors.size() * 3) / 4];

    // Return the tuple of statistics
    return std::make_tuple(mean_error, min_error, q1, median, q3, max_error);
}

// Perform a breadth-first search (BFS) to gather a patch of non-deleted nodes around a seed node, limited by breadth distance
std::vector<size_t> bfsExpand(const std::vector<Node>& graph, size_t seed_idx, size_t breadth) {
    std::vector<size_t> patch;  // Stores the indices of nodes in the patch
    std::queue<std::pair<size_t, size_t>> node_queue;  // Pair of (node index, current distance)
    std::vector<bool> visited(graph.size(), false);

    // Start BFS from the seed node, with distance 0
    node_queue.push({seed_idx, 0});
    visited[seed_idx] = true;

    while (!node_queue.empty()) {
        auto [current_idx, current_breadth] = node_queue.front();
        node_queue.pop();

        // Add the current node to the patch if it's not deleted and contains gt
        if (!graph[current_idx].deleted && graph[current_idx].gt) {
            patch.push_back(current_idx);
        }

        // Stop expanding further if we have reached the maximum breadth level
        if (current_breadth >= breadth) {
            continue;
        }

        // Explore neighbors (edges) of the current node
        for (int j = 0; j < graph[current_idx].num_edges; ++j) {
            const Edge& edge = graph[current_idx].edges[j];
            if (!visited[edge.target_node] && !graph[edge.target_node].deleted && !edge.same_block) {
                visited[edge.target_node] = true;
                node_queue.push({edge.target_node, current_breadth + 1});  // Push neighbor with incremented breadth level
            }
        }
    }

    return patch;  // Return the indices of non-deleted nodes in the patch
}

// Function to compute errors between the seed node and nodes in its patch
std::tuple<float, float, float, float, float, float> computeLocalizedError(const std::vector<Node>& graph, const std::vector<size_t>& valid_gt_indices, int N = 100, int L = 10) {
    if (valid_gt_indices.size() < 2) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);  // Not enough valid ground truth nodes
    }

    std::vector<float> patch_errors;
    std::random_device rd;
    std::mt19937 gen(rd());  // Random number generator
    std::uniform_int_distribution<> dis(0, valid_gt_indices.size() - 1);

    // Loop for N patches
    for (int i = 0; i < N; ++i) {
        // Randomly pick a seed node
        size_t seed_idx = valid_gt_indices[dis(gen)];
        const Node& seed_node = graph[seed_idx];

        // Perform BFS to gather a patch around the seed node
        std::vector<size_t> patch = bfsExpand(graph, seed_idx, L);

        // Compute the error between the seed node and each node in the patch
        for (size_t patch_idx : patch) {
            if (patch_idx == seed_idx) continue;  // Skip the seed node itself

            const Node& patch_node = graph[patch_idx];

            // Compute the distance1 (ground truth distances)
            float dist1 = seed_node.gt_f_star - patch_node.gt_f_star;

            // Compute the distance2 (computed f_star distances)
            float dist2 = seed_node.f_star - patch_node.f_star;

            // Compute the absolute error
            float error = std::abs(dist1 - dist2);

            // Store the error
            patch_errors.push_back(error);
        }
    }

    // If no errors were calculated, return 0 values
    if (patch_errors.empty()) {
        return std::make_tuple(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Sort the error values to compute statistics
    std::sort(patch_errors.begin(), patch_errors.end());

    // Compute the mean
    float total_error = std::accumulate(patch_errors.begin(), patch_errors.end(), 0.0f);
    float mean_error = total_error / patch_errors.size();

    // Min and Max
    float min_error = patch_errors.front();
    float max_error = patch_errors.back();

    // Quartiles
    float q1 = patch_errors[patch_errors.size() / 4];
    float median = patch_errors[patch_errors.size() / 2];
    float q3 = patch_errors[(patch_errors.size() * 3) / 4];

    // Return the tuple of statistics
    return std::make_tuple(mean_error, min_error, q1, median, q3, max_error);
}

// void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000) {
//     // Find min and max f_star values
//     float min_f = min_f_star(graph);
//     float max_f = max_f_star(graph);

//     // Calculate bucket size
//     float bucket_size = (max_f - min_f) / num_buckets;

//     // Initialize the histogram with 0 counts
//     std::vector<int> histogram(num_buckets, 0);

//     // Fill the histogram
//     for (const auto& node : graph) {
//         if (node.deleted) {
//             continue;
//         }
//         int bucket_index = static_cast<int>((node.f_star - min_f) / bucket_size);
//         if (bucket_index >= 0 && bucket_index < num_buckets) {
//             histogram[bucket_index]++;
//         }
//     }

//     // Create a blank image for the histogram with padding on the left
//     int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
//     int hist_h = 800;  // height of the histogram image
//     int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel
//     int left_padding = 50;  // Add 50 pixels of padding on the left side

//     cv::Mat hist_image(hist_h, hist_w + left_padding + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels and padding

//     // Normalize the histogram to fit in the image
//     int max_value = *std::max_element(histogram.begin(), histogram.end());
//     for (int i = 0; i < num_buckets; ++i) {
//         histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
//     }

//     // Draw the histogram with left padding
//     for (int i = 0; i < num_buckets; ++i) {
//         cv::rectangle(hist_image, 
//                       cv::Point(left_padding + i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
//                       cv::Point(left_padding + (i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
//                       cv::Scalar(0, 0, 0), 
//                       cv::FILLED);
//     }

//     // Add x-axis labels
//     std::string min_label = "Min: " + std::to_string(min_f);
//     std::string max_label = "Max: " + std::to_string(max_f);
//     cv::putText(hist_image, min_label, cv::Point(left_padding + 10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
//     cv::putText(hist_image, max_label, cv::Point(left_padding + hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

//     // Save the histogram image to a file if string not empty
//     if (!filename.empty()) {
//         cv::imwrite(filename, hist_image);
//     }

//     // Display the histogram
//     cv::imshow("Histogram of f_star values", hist_image);
//     cv::waitKey(1);
// }

bool is_edge_valid(const std::vector<Node>& graph, const Edge& edge, const Node& current_node, float threshold = 0.1) {
    float diff = graph[edge.target_node].f_star - current_node.f_star;
    bool valid =true;
    if (edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 * threshold;
    }
    else if (!edge.same_block) {
        valid = std::abs(diff - edge.k) < 360 * threshold;
    }
    return valid;
}

bool remove_invalid_edges(std::vector<Node>& graph, float threshold = 0.1) {
    int erased_edges = 0;
    int remaining_edges = 0;

    for (auto& node : graph) {
        if (node.deleted || node.num_edges == 0) {
            continue;
        }

        int edges_before = node.num_edges;
        int valid_edge_count = 0;

        // Create a temporary array to store valid edges
        Edge* valid_edges = new Edge[edges_before];

        // Copy valid edges to the temporary array
        for (int j = 0; j < edges_before; ++j) {
            const Edge& edge = node.edges[j];

            if (!edge.fixed && !is_edge_valid(graph, edge, node, threshold)) {
                // Edge is invalid and should be erased
                erased_edges++;
            } else {
                // Edge is valid and should be kept
                valid_edges[valid_edge_count] = edge;
                valid_edge_count++;
            }
        }

        // Free the old edges array
        delete[] node.edges;

        // If there are valid edges, update the node's edges and count
        if (valid_edge_count > 0) {
            node.edges = new Edge[valid_edge_count];
            std::copy(valid_edges, valid_edges + valid_edge_count, node.edges);
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, reset the edges
            node.edges = nullptr;
            node.num_edges = 0;
        }

        // Free the temporary valid_edges array
        delete[] valid_edges;

        // Count remaining valid edges
        remaining_edges += node.num_edges;
    }

    std::cout << "Erased edges: " << erased_edges << std::endl;
    std::cout << "Remaining edges: " << remaining_edges << std::endl;

    return erased_edges > 0;
}

std::vector<float> generate_spring_constants(float start_value, int steps) {
    std::vector<float> spring_constants;
    float current_value = start_value;
    float multiplier = std::pow(0.1f / (start_value - 1.0f), 1.0f / steps); // after steps should get value to 1.1
    std::cout << "Multiplier: " << multiplier << std::endl;

    for (int i = 0; i < steps; ++i) {
        spring_constants.push_back(current_value);

        // Alternate between above and below 1, gradually reducing the difference
        if (current_value > 1.0f) {
            // Reduce the multiplier slightly to get closer to 1
            current_value = 1.0f + multiplier * (current_value - 1.0f);
            current_value = 1.0f / current_value;
        }
        else {
            current_value = 1.0f / current_value;
            // Reduce the multiplier slightly to get closer to 1
            current_value = 1.0f + multiplier * (current_value - 1.0f);
        }
    }

    // Ensure the final value is exactly 1
    spring_constants.push_back(1.0f);

    return spring_constants;
}

std::vector<size_t> get_valid_indices(const std::vector<Node>& graph) {
    std::vector<size_t> valid_indices;
    size_t num_valid_nodes = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        // node is not deleted and not fixed, can be updated
        if (!graph[i].deleted) {
            valid_indices.push_back(i);
            num_valid_nodes++;
        }
    }
    std::cout << "Number of valid nodes: " << num_valid_nodes << std::endl;
    return valid_indices;
}

std::vector<size_t> get_valid_gt_indices(const std::vector<Node>& graph) {
    std::vector<size_t> valid_gt_indices;
    size_t num_valid_nodes = 0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (!graph[i].deleted && graph[i].gt) {
            valid_gt_indices.push_back(i);
            num_valid_nodes++;
        }
    }
    std::cout << "Number of valid gt nodes: " << num_valid_nodes << std::endl;
    return valid_gt_indices;
}

// This is an example of a solve function that takes the graph and parameters as input
void solve(std::vector<Node>& graph, argparse::ArgumentParser* program, int num_iterations = 10000) {
    // Default values for parameters
    int estimated_windings = 0;
    float spring_constant = 2.0f;
    float o = 2.0f;
    float iterations_factor = 2.0f;
    float o_factor = 0.25f;
    float spring_factor = 6.0f;
    int steps = 5;
    bool auto_mode = false;
    bool video_mode = false;

    // Parse the arguments
    try {
        estimated_windings = program->get<int>("--estimated_windings");
        o = program->get<float>("--o");
        spring_constant = program->get<float>("--spring_constant");
        steps = program->get<int>("--steps");
        iterations_factor = program->get<float>("--iterations_factor");
        o_factor = program->get<float>("--o_factor");
        spring_factor = program->get<float>("--spring_factor");
        auto_mode = program->get<bool>("--auto");
        video_mode = program->get<bool>("--video");
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return;
    }

    // Print the parameters
    std::cout << "Estimated Windings: " << estimated_windings << std::endl;
    std::cout << "Number of Iterations: " << num_iterations << std::endl;
    std::cout << "O: " << o << std::endl;
    std::cout << "Spring Constant: " << spring_constant << std::endl;
    std::cout << "Steps: " << steps << std::endl;
    std::cout << "Iterations Factor: " << iterations_factor << std::endl;
    std::cout << "O Factor: " << o_factor << std::endl;
    std::cout << "Spring Factor: " << spring_factor << std::endl;
    std::cout << "Auto Mode: " << (auto_mode ? "Enabled" : "Disabled") << std::endl;

    // Path to the histogram directory
    std::string histogram_dir = "histogram";

    // Delete the existing histogram directory if it exists
    if (fs::exists(histogram_dir)) {
        fs::remove_all(histogram_dir);
    }

    // Create a new histogram directory
    fs::create_directory(histogram_dir);

    // Generate spring constants starting from 5.0 with 12 steps
    std::vector<float> spring_constants = generate_spring_constants(spring_constant, steps);

    // Calculate the number of digits needed for padding
    int max_index_digits = static_cast<int>(std::log10(spring_constants.size())) + 1;
    int max_iter_digits = static_cast<int>(std::log10(num_iterations - 1)) + 1;

    // store only the valid indices to speed up the loop
    std::vector<size_t> valid_indices;
    std::vector<size_t> valid_gt_indices;

    float invalid_edge_threshold = 3.5f;

    int edges_deletion_round = 0;
    while (true) {
        // store only the valid indices to speed up the loop
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph);
        // Do 2 rounds of edge deletion
        if (edges_deletion_round > 8 || invalid_edge_threshold <= 0.05) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations = num_iterations * 3;
            spring_constant = 1.0f;

            break;
        }
        solve_gpu_session(graph, edges_deletion_round, video_mode, max_index_digits, max_iter_digits, num_iterations, o, spring_factor, steps, spring_constants, valid_indices, iterations_factor, o_factor, estimated_windings, histogram_dir);
        // After first edge deletion round remove the invalid edges
        if (edges_deletion_round >= 0) {
            // Remove edges with too much difference between f_star and k
            remove_invalid_edges(graph, invalid_edge_threshold);
        }
        find_largest_connected_component(graph);
        // Update the valid indices
        valid_indices = get_valid_indices(graph);
        valid_gt_indices = get_valid_gt_indices(graph);
        
        // Reduce the threshold by 20% each time
        invalid_edge_threshold *= 0.7f;
        invalid_edge_threshold -= 0.1f;
        if (invalid_edge_threshold < 0.30) {
            invalid_edge_threshold = 0.30;
        }
        std::cout << "Reducing invalid edges threshold to: " << invalid_edge_threshold << std::endl;
        // Assign winding angles again after removing invalid edges
        float scale = calculate_scale(graph, estimated_windings);
        
        // // Detect folds
        // if (edges_deletion_round == 0) {
        //     // Solve the fold detection
        //     solve_fold(graph, program, get_valid_indices(graph), 10000);
        // }

        // Assign winding angles again after removing invalid edges
        assign_winding_angles(graph, scale);
        
        // Print the error statistics
        auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
        auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
        std::cout << "After assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;

        // Save the graph back to a binary file
        save_graph_to_binary("temp_output_graph.bin", graph);

        edges_deletion_round++;
    }
    // Assign winding angles to the graph
    float scale = calculate_scale(graph, estimated_windings);
    assign_winding_angles(graph, scale);

    // Print the error statistics
    auto [mean, min, q1, median, q3, max] = computeErrorStats(graph, valid_gt_indices);
    auto [mean_local, min_local, q1_local, median_local, q3_local, max_local] = computeLocalizedError(graph, valid_gt_indices, 100, 187); // 187 patches breath = radius of 30cm local area covered. 
    std::cout << "After final assigning winding angles with Prim MST. Mean Error to GT: " << mean << ", Min: " << min << ", Q1: " << q1 << ", Median: " << median << ", Q3: " << q3 << ", Max: " << max << " | Localized Error: " << mean_local << ", Min: " << min_local << ", Q1: " << q1_local << ", Median: " << median_local << ", Q3: " << q3_local << ", Max: " << max_local << std::endl;

    if (video_mode) {
        // Calculate final histogram after all iterations
        calculate_histogram(graph, "final_histogram.png");
        // After generating all histograms, create a final video from the images
        create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
    }
}

void invert_winding_direction_graph(std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];
            if (edge.same_block) {
                float turns = edge.k / 360;
                turns = std::round(turns);
                if (std::abs(turns) > 1 || std::abs(turns) == 0) {
                    std::cout << "Inverting winding direction failed, turns: " << turns << std::endl;
                }
                edge.k = edge.k - (2.0f * 360.0f * turns);
            }
        }
    }
}

void auto_winding_direction(std::vector<Node>& graph, argparse::ArgumentParser* program) {
    std::cout << "Auto Winding Direction" << std::endl;
    
    // Make a copy of the graph
    std::vector<Node> auto_graph = graph;
    // min max z values of graph
    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::min();
    for (const auto& node : auto_graph) {
        if (node.deleted) {
            continue;
        }
        if (node.z < z_min) {
            z_min = node.z;
        }
        if (node.z > z_max) {
            z_max = node.z;
        }
    }
    float middle_z = (z_min + z_max) / 2.0f;
    // set z range: middle +- 250
    set_z_range_graph(auto_graph, middle_z - 250.0f, middle_z + 250.0f); // speedup the winding direction computation
    
    std::vector<Node> auto_graph_other_block = auto_graph;
    invert_winding_direction_graph(auto_graph_other_block); // build inverted other block graph

    // solve
    int auto_num_iterations = program->get<int>("--auto_num_iterations");
    if (auto_num_iterations == -1) {
        auto_num_iterations = program->get<int>("--num_iterations");
    }
    solve(auto_graph_other_block, program, auto_num_iterations);
    solve(auto_graph, program, auto_num_iterations);

    // Exact matching score
    float exact_score = exact_matching_score(auto_graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;
    float exact_score_other_block = exact_matching_score(auto_graph_other_block);
    std::cout << "Exact Matching Score Other Block: " << exact_score_other_block << std::endl;

    // Remove all same_block edges for further comparison
    for (auto& node : auto_graph_other_block) {
        int valid_edge_count = 0;
        
        // Count the number of valid edges (those not marked as same_block)
        for (int i = 0; i < node.num_edges; ++i) {
            if (!node.edges[i].same_block) {
                ++valid_edge_count;
            }
        }

        // If there are valid edges, allocate a new array for valid edges
        if (valid_edge_count > 0) {
            Edge* valid_edges = new Edge[valid_edge_count];
            int index = 0;

            // Copy valid edges to the new array
            for (int i = 0; i < node.num_edges; ++i) {
                if (!node.edges[i].same_block) {
                    valid_edges[index++] = node.edges[i];
                }
            }

            // Free the old edges array
            delete[] node.edges;

            // Assign the new array to the node
            node.edges = valid_edges;
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, free the old edges array and set edges to nullptr
            delete[] node.edges;
            node.edges = nullptr;
            node.num_edges = 0;
        }
    }

    for (auto& node : auto_graph) {
        int valid_edge_count = 0;
        
        // Count the number of valid edges (those not marked as same_block)
        for (int i = 0; i < node.num_edges; ++i) {
            if (!node.edges[i].same_block) {
                ++valid_edge_count;
            }
        }

        // If there are valid edges, allocate a new array for valid edges
        if (valid_edge_count > 0) {
            Edge* valid_edges = new Edge[valid_edge_count];
            int index = 0;

            // Copy valid edges to the new array
            for (int i = 0; i < node.num_edges; ++i) {
                if (!node.edges[i].same_block) {
                    valid_edges[index++] = node.edges[i];
                }
            }

            // Free the old edges array
            delete[] node.edges;

            // Assign the new array to the node
            node.edges = valid_edges;
            node.num_edges = valid_edge_count;
        } else {
            // If no valid edges remain, free the old edges array and set edges to nullptr
            delete[] node.edges;
            node.edges = nullptr;
            node.num_edges = 0;
        }
    }

    // Calculate exact matching score for both graphs
    float exact_score2 = exact_matching_score(auto_graph);
    std::cout << "Exact Matching Score (no same block edges): " << exact_score2 << std::endl;
    float exact_score_other_block2 = exact_matching_score(auto_graph_other_block);
    std::cout << "Exact Matching Score Other Block (no same block edges): " << exact_score_other_block2 << std::endl;

    if (exact_score_other_block2 > exact_score2) {
        std::cout << "Inverting the winding direction" << std::endl;
        invert_winding_direction_graph(graph);
    }
    else {
        std::cout << "Standard winding direction has highest score. Not inverting the winding direction." << std::endl;
    }
}

void construct_ground_truth_graph(std::vector<Node>& graph) {
    std::cout << "Constructing Ground Truth Graph" << std::endl;

    // Delete node withouth ground truth
    for (size_t i = 0; i < graph.size(); ++i) {
        graph[i].deleted = !graph[i].gt;
        // assign gt to f_star
        graph[i].f_star = graph[i].gt_f_star;
    }
}

void fix_gt_parts(std::vector<Node>& graph, std::vector<float> fix_lines_z, int fix_windings = 0, float edge_good_certainty = 1.0f, bool fix_all = false) {
    // Update the lines coordinate system to mask3d system
    for (size_t i = 0; i < fix_lines_z.size(); ++i) {
        fix_lines_z[i] = (fix_lines_z[i] + 500) / 4.0f;
    }

    // Fix lines: if the z value of the node is in the fix_lines_z +- 25 and ground truth (gt) is available, set the gt to f_star and fix it
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted || !graph[i].gt) {
            continue;
        }
        for (size_t j = 0; j < fix_lines_z.size(); ++j) {
            if (graph[i].z > fix_lines_z[j] - 25.0f && graph[i].z < fix_lines_z[j] + 25.0f) {
                graph[i].fixed = true;
            }
        }
    }

    // Get the min and max f_star using ground truth values
    float f_min = min_f_star(graph, true);
    float f_max = max_f_star(graph, true);

    // Fix windings based on the number of windings to fix
    float start_winding = f_min;
    float end_winding = f_max;
    
    if (fix_windings < 0) {
        start_winding = f_max - 360.0f * std::abs(fix_windings);
    } else if (fix_windings > 0) {
        end_winding = f_min + 360.0f * std::abs(fix_windings);
    }

    if (fix_windings != 0) {
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].deleted || !graph[i].gt) {
                continue;
            }
            if (graph[i].gt_f_star >= start_winding && graph[i].gt_f_star <= end_winding) {
                graph[i].fixed = true;
            }
        }
    }

    // Fix all ground truth nodes if the flag is set
    if (fix_all) {
        for (size_t i = 0; i < graph.size(); ++i) {
            if (graph[i].deleted || !graph[i].gt) {
                continue;
            }
            graph[i].fixed = true;
        }
    }

    // Now fix good edges and delete bad ones
    int good_edges = 0;
    int bad_edges = 0;

    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }

        for (int j = 0; j < graph[i].num_edges; ++j) {
            Edge& edge = graph[i].edges[j];

            if (graph[edge.target_node].deleted) {
                continue;
            }

            // If both the current node and the target node are fixed and have ground truth, evaluate the edge
            if (graph[i].fixed && graph[edge.target_node].fixed && graph[i].gt && graph[edge.target_node].gt) {
                // Check if edge has the correct k value
                float diff = graph[edge.target_node].gt_f_star - graph[i].gt_f_star;
                float k = edge.k;

                // Check if the edge k value is close to the ground truth difference
                if (std::abs(k - diff) < 0.1) {
                    // Good edge: set high certainty
                    edge.certainty = edge_good_certainty;
                    good_edges++;
                } else {
                    // Bad edge: set certainty to zero
                    edge.certainty = 0.0f;
                    bad_edges++;
                }

                // Fix the edge
                edge.fixed = true;
            }
        }
    }

    std::cout << "Fixed " << good_edges << " good edges and deleted " << bad_edges << " bad edges." << std::endl;
}

int main(int argc, char** argv) {
    // Parse the input graph file from arguments using argparse
    argparse::ArgumentParser program("Graph Solver");

    // Default values for parameters
    int estimated_windings = 0;
    int num_iterations = 10000;
    int auto_num_iterations = -1;
    float spring_constant = 2.0f;
    float o = 2.0f;
    float iterations_factor = 2.0f;
    float o_factor = 0.25f;
    float spring_factor = 6.0f;
    int steps = 5;
    int z_min = -2147483648;
    int z_max = 2147483647;
    float same_winding_factor = 1.0f;

    // Add command-line arguments for graph input and output
    program.add_argument("--input_graph")
        .help("Input graph binary file")
        .default_value(std::string("graph.bin"));

    program.add_argument("--output_graph")
        .help("Output graph binary file")
        .default_value(std::string("output_graph.bin"));

    // Add command-line arguments
    program.add_argument("--estimated_windings")
        .help("Estimated windings (int)")
        .default_value(estimated_windings)
        .scan<'i', int>();

    program.add_argument("--num_iterations")
        .help("Number of iterations (int)")
        .default_value(num_iterations)
        .scan<'i', int>();

    program.add_argument("--o")
        .help("O parameter (float)")
        .default_value(o)
        .scan<'g', float>();

    program.add_argument("--spring_constant")
        .help("Spring constant (float)")
        .default_value(spring_constant)
        .scan<'g', float>();

    program.add_argument("--steps")
        .help("Steps (int)")
        .default_value(steps)
        .scan<'i', int>();

    program.add_argument("--iterations_factor")
        .help("Iterations factor (float)")
        .default_value(iterations_factor)
        .scan<'g', float>();

    program.add_argument("--o_factor")
        .help("O factor (float)")
        .default_value(o_factor)
        .scan<'g', float>();

    program.add_argument("--spring_factor")
        .help("Spring factor (float)")
        .default_value(spring_factor)
        .scan<'g', float>();

    program.add_argument("--same_winding_factor")
        .help("Same winding factor (float)")
        .default_value(same_winding_factor)
        .scan<'g', float>();

    program.add_argument("--z_min")
        .help("Z range (int)")
        .default_value(z_min)
        .scan<'i', int>();

    program.add_argument("--z_max")
        .help("Z range (int)")
        .default_value(z_max)
        .scan<'i', int>();

    // Add the boolean flag --auto
    program.add_argument("--auto")
        .help("Enable automatic mode")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Add the auto number of iterations
    program.add_argument("--auto_num_iterations")
        .help("Number of iterations for auto mode (int)")
        .default_value(auto_num_iterations)
        .scan<'i', int>();

    // Add the boolean flag --video
    program.add_argument("--video")
        .help("Enable video creation")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Add boolean flag --gt_graph
    program.add_argument("--gt_graph")
        .help("Enable ground truth graph construction")
        .default_value(false)   // Set default to false
        .implicit_value(true);  // If present, set to true

    // Multithreading number threads
    program.add_argument("--threads")
        .help("Number of threads (int)")
        .default_value(-1)
        .scan<'i', int>();

    // Flag fix_same_block_edges
    program.add_argument("--fix_same_block_edges")
        .help("Fix same block edges")
        .default_value(false)
        .implicit_value(true);

    // Flag to invert graph
    program.add_argument("--invert")
        .help("Invert the winding direction of the graph")
        .default_value(false)
        .implicit_value(true);

    // Push out factor of the fold detection
    program.add_argument("--push_out_factor")
        .help("Push out factor of the fold detection (float)")
        .default_value(1.01f)
        .scan<'g', float>();

    // Flag fix all gt for solver
    program.add_argument("--fix_all_gt")
        .help("Fix all gt nodes")
        .default_value(false)
        .implicit_value(true);

    try {
        program.parse_args(argc, argv);
        same_winding_factor = program.get<float>("--same_winding_factor");
    } catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    // Set number threads
    if (program.get<int>("--threads") > 0) {
        omp_set_num_threads(program.get<int>("--threads"));
    }

    // Load the graph from the input binary file
    std::string input_graph_file = program.get<std::string>("--input_graph");
    auto [graph, max_certainty] = load_graph_from_binary(input_graph_file, true, (static_cast<float>(program.get<int>("--z_min")) + 500) / 4, (static_cast<float>(program.get<int>("--z_max")) + 500) / 4, same_winding_factor, program.get<bool>("--fix_same_block_edges"));

    // invert graph
    if (program.get<bool>("--invert")) {
        std::cout << "Inverting the winding direction of the graph" << std::endl;
        invert_winding_direction_graph(graph);
    }

    // Calculate the exact matching loss
    float exact_score = exact_matching_score(graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;

    // Calculate the approximate matching loss
    float approx_loss = approximate_matching_loss(graph, 1.0f);
    std::cout << "Approximate Matching Loss: " << approx_loss << std::endl;

    // Calculate and display the histogram of f_star values
    // calculate_histogram(graph);

    // Check if the ground truth graph construction is enabled
    if (program.get<bool>("--gt_graph")) {
        // Construct the ground truth graph
        construct_ground_truth_graph(graph);
    }
    else {
        // Solve the problem using a solve function
        fix_gt_parts(graph, {6000}, -10, 10.0f * max_certainty, program.get<bool>("--fix_all_gt"));
        num_iterations = program.get<int>("--num_iterations");
        solve(graph, &program, num_iterations);
    }

    // print the min and max f_star values
    std::cout << "Min f_star: " << min_f_star(graph) << std::endl;
    std::cout << "Max f_star: " << max_f_star(graph) << std::endl;

    // Save the graph back to a binary file
    std::string output_graph_file = program.get<std::string>("--output_graph");
    save_graph_to_binary(output_graph_file, graph);

    return 0;
}

// Example command to run the program: ./build/graph_problem --input_graph graph.bin --output_graph output_graph.bin --auto --auto_num_iterations 2000 --video --z_min 5000 --z_max 7000 --num_iterations 5000 --estimated_windings 160 --steps 3 --spring_constant 1.2
