/*
Julian Schilliger 2024 ThaumatoAnakalyptor
*/

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

namespace fs = std::filesystem;

struct Edge {
    unsigned int target_node;
    float certainty;
    float certainty_factor = 1.0f;
    float certainty_factored;
    float k;
    bool same_block;
};

struct Node {
    float z;
    float f_init;
    float f_tilde;
    float f_star;
    bool gt;
    float gt_f_star;
    bool deleted = false;
    std::vector<Edge> edges;
};

std::vector<Node> load_graph_from_binary(const std::string &file_name, bool clip_z = false, float z_min = 0.0f, float z_max = 0.0f, float same_winding_factor = 1.0f) {
    std::vector<Node> graph;
    std::ifstream infile(file_name, std::ios::binary);

    if (!infile.is_open()) {
        std::cerr << "Error opening file." << std::endl;
        return graph;
    }

    // Read the number of nodes
    unsigned int num_nodes;
    infile.read(reinterpret_cast<char*>(&num_nodes), sizeof(unsigned int));
    std::cout << "Number of nodes in graph: " << num_nodes << std::endl;

    // Prepare the graph with empty nodes
    graph.resize(num_nodes);

    // Read each node's winding angle and store it as int16_t
    for (unsigned int i = 0; i < num_nodes; ++i) {
        infile.read(reinterpret_cast<char*>(&graph[i].z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        infile.read(reinterpret_cast<char*>(&graph[i].gt), sizeof(bool));
        infile.read(reinterpret_cast<char*>(&graph[i].gt_f_star), sizeof(float));
        graph[i].f_tilde = graph[i].f_init;
        graph[i].f_star = graph[i].f_init;
    }
    std::cout << "Nodes loaded successfully." << std::endl;
    // Read the adjacency list
    int count_same_block_edges = 0;
    for (unsigned int i = 0; i < num_nodes; ++i) {
        unsigned int node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(unsigned int));

        unsigned int num_edges;
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(unsigned int));

        for (unsigned int j = 0; j < num_edges; ++j) {
            Edge edge;
            infile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(unsigned int));
            infile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(float));
            // clip certainty
            // if (edge.certainty < 0.001f) {
            //     std::cout << "Certainty below 0.001: " << edge.certainty << std::endl;
            //     edge.certainty = 0.001f;
            // }
            // add certainty factored
            edge.certainty_factored = edge.certainty;
            infile.read(reinterpret_cast<char*>(&edge.k), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(bool));
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
            if (edge.same_block) { // no same subvolume edges
                edge.certainty *= same_winding_factor;
                edge.certainty_factored *= same_winding_factor;
                count_same_block_edges++;
                // continue;
                if (std::abs(edge.k) > 450) {
                    std::cout << "Edge with k > 450: " << edge.k << std::endl;
                }
                if (std::abs(edge.k) < 180) {
                    std::cout << "Edge with k < 180: " << edge.k << std::endl;
                }
            }
            graph[node_id].edges.push_back(edge);
        }
    }

    std::cout << "Same block edges: " << count_same_block_edges << std::endl;

    std::cout << "Graph loaded successfully." << std::endl;

    // Find largest certainty (scaled back to float for display)
    float max_certainty = 0;
    for (const auto& node : graph) {
        for (const auto& edge : node.edges) {
            if (edge.certainty > max_certainty) {
                max_certainty = edge.certainty;
            }
        }
    }

    std::cout << "Max Certainty: " << max_certainty << std::endl;

    infile.close();
    return graph;
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

        for (const auto& edge : graph[current].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
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

void dfs_assign_f_star(int node_index, std::vector<Node>& graph, std::vector<bool>& visited) {
    std::stack<int> stack;
    stack.push(node_index);
    visited[node_index] = true;

    // Initialize f_star with f_init for the starting node
    graph[node_index].f_star = graph[node_index].f_init;

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        for (auto& edge : graph[current].edges) {
            if (!visited[edge.target_node] && !graph[edge.target_node].deleted) {
                visited[edge.target_node] = true;

                // Calculate the f_star for the target node
                graph[edge.target_node].f_star = closest_valid_winding_angle(graph[current].f_init , graph[current].f_tilde + edge.k);

                stack.push(edge.target_node);
            }
        }
    }
}

void assign_winding_angles_dfs(std::vector<Node>& graph) {
    size_t num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);

    // Find a non-deleted node in the largest connected component to start the DFS
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

    // Perform DFS to assign f_star values
    dfs_assign_f_star(start_node, graph, visited);
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

        for (const auto& edge : graph[u].edges) {
            if (graph[u].deleted) {
                continue;
            }
            size_t v = edge.target_node;
            if (graph[v].deleted) {
                continue;
            }
            // float k_delta = std::abs((graph[v].f_tilde - graph[u].f_tilde) - edge.k) / std::abs(edge.k); // difference between BP solution and estimated k from the graph
            // float k_delta = std::abs((graph[v].f_tilde - graph[u].f_tilde) - edge.k); // difference between BP solution and estimated k from the graph
            float k_delta = std::abs(scale * (graph[v].f_tilde - graph[u].f_tilde) - edge.k); // difference between BP solution and estimated k from the graph
            // float k_delta = std::abs(edge.certainty_factored); // adpated certainty of the edge

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

    // Find for each node the children
    std::vector<std::vector<size_t>> children(num_nodes);
    std::vector<std::vector<float>> children_k_values(num_nodes);
    
    for (size_t i = 0; i < num_nodes; ++i) {
        if (valid[i]) {
            children[parent[i]].push_back(i);
            children_k_values[parent[i]].push_back(k_values[i]);
        }
    }

    // Traverse the MST in a DFS manner to assign f_star values
    std::stack<size_t> stack;
    stack.push(start_node);

    // Sanity Check
    while (!stack.empty()) {
        size_t current = stack.top();
        stack.pop();

        if (graph[current].deleted) {
            // graph[current].f_star = 0;
            // graph[current].f_tilde = 0;
            continue;
        }

        for (size_t i = 0; i < children[current].size(); ++i) {
            size_t child = children[current][i];
            if (graph[child].deleted) {
                // graph[child].f_star = 0;
                // graph[child].f_tilde = 0;
                continue;
            }
            float k = children_k_values[current][i];
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + k);
            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[child].f_star);
            
            graph[child].f_tilde = graph[child].f_star;
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

void update_weights(std::vector<Node>& graph, float scale) {
    for (auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        std::vector<float> errors(node.edges.size());
        float weighted_mean_error = 0.0f;
        float mean_certainty = 0.0f;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            if (graph[node.edges[i].target_node].deleted) {
                continue;
            }
            // difference between BP solution and estimated k from the graph
            errors[i] = std::abs(scale * (graph[node.edges[i].target_node].f_tilde - node.f_tilde) - node.edges[i].k);
            weighted_mean_error += errors[i] * node.edges[i].certainty_factored;
            mean_certainty += node.edges[i].certainty_factored;
        }
        if (mean_certainty == 0.0f) {
            continue;
        }
        weighted_mean_error /= mean_certainty;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            // asolute distance from the mean error
            errors[i] = std::abs(errors[i] - weighted_mean_error);
        }
        float factors_mean = 0.0f;
        size_t factors_count = 0;
        for (size_t i = 0; i < node.edges.size(); ++i) {
            if (graph[node.edges[i].target_node].deleted) {
                continue;
            }
            // certainty factor based on the error
            node.edges[i].certainty_factor = 0.8f * node.edges[i].certainty_factor + 0.2f * certainty_factor(errors[i]);
            factors_mean += node.edges[i].certainty_factor;
            factors_count++;
        }
        factors_mean /= factors_count;
        // Normalize the factors
        for (auto& edge : node.edges) {
            edge.certainty_factor = edge.certainty_factor / factors_mean;
            edge.certainty_factored = edge.certainty * edge.certainty_factor;
        }
    }
}

float min_f_star(const std::vector<Node>& graph) {
    float min_f = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.f_star < min_f) {
            min_f = node.f_star;
        }
    }

    return min_f;
}

float max_f_star(const std::vector<Node>& graph) {
    float max_f = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.deleted) {
            continue;
        }
        if (node.f_star > max_f) {
            max_f = node.f_star;
        }
    }

    return max_f;
}

float calculate_scale(const std::vector<Node>& graph, int estimated_windings) {
    if (estimated_windings <= 0) {
        return 1.0f;
    }
    float min_f = min_f_star(graph);
    float max_f = max_f_star(graph);
    return std::abs((360.0f * estimated_windings) / (max_f - min_f));
}

float exact_matching_score(std::vector<Node>& graph) {
    // Assing closest valid winding angle to f_star based on f_init
    std::vector<Node> graph_copy = graph;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        graph_copy[i].f_star = closest_valid_winding_angle(graph[i].f_init, graph[i].f_star);
    }

    float score = 0.0;
    for (size_t i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        for (const auto& edge : graph_copy[i].edges) {
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
    float loss = 0.0;

    for (const auto& node : graph) {
        for (const auto& edge : node.edges) {
            float diff = graph[edge.target_node].f_star - node.f_star;
            float l = diff - edge.k;
            loss += edge.certainty * std::exp(-a * std::abs(l));
        }
    }

    return loss;
}

float computeMeanError(const std::vector<Node>& graph, const std::vector<size_t>& valid_gt_indices, int n_pairs = 10'000) {
    if (valid_gt_indices.size() < 2) {
        return 0.0f; // Not enough valid ground truth nodes to form pairs
    }

    float total_error = 0.0f;
    int pair_count = 0;
    
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
        float dist1 = std::abs(node_i.gt_f_star - node_j.gt_f_star);

        // Compute the distance2 (computed f_star distances)
        float dist2 = std::abs(node_i.f_star - node_j.f_star);

        // Compute the absolute error
        float error = std::abs(dist1 - dist2);

        // Accumulate the error
        total_error += error;
        ++pair_count;
    }

    // If no valid pairs are found, return 0 to avoid division by zero
    if (pair_count == 0) {
        return 0.0f;
    }

    // Return the mean error
    return total_error / pair_count;
}

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 1000) {
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

    // Create a blank image for the histogram
    int hist_w = num_buckets;  // width of the histogram image matches the number of buckets
    int hist_h = 800;  // height of the histogram image, increased to 1000 pixels
    int bin_w = std::max(1, hist_w / num_buckets);  // Ensure bin width is at least 1 pixel

    cv::Mat hist_image(hist_h, hist_w + 100, CV_8UC3, cv::Scalar(255, 255, 255));  // Extra space for labels

    // Normalize the histogram to fit in the image
    int max_value = *std::max_element(histogram.begin(), histogram.end());
    for (int i = 0; i < num_buckets; ++i) {
        histogram[i] = (histogram[i] * (hist_h - 50)) / max_value;  // Leaving some space at the top for labels
    }

    // Draw the histogram
    for (int i = 0; i < num_buckets; ++i) {
        cv::rectangle(hist_image, 
                      cv::Point(i * bin_w, hist_h - histogram[i] - 50),  // Adjusted to leave space for labels
                      cv::Point((i + 1) * bin_w, hist_h - 50),  // Adjusted to leave space for labels
                      cv::Scalar(0, 0, 0), 
                      cv::FILLED);
    }

    // Add x-axis labels
    std::string min_label = "Min: " + std::to_string(min_f);
    std::string max_label = "Max: " + std::to_string(max_f);
    cv::putText(hist_image, min_label, cv::Point(10, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    cv::putText(hist_image, max_label, cv::Point(hist_w - 200, hist_h - 10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);

    // Save the histogram image to a file if string not empty
    if (!filename.empty()) {
        cv::imwrite(filename, hist_image);
    }

    // Display the histogram
    cv::imshow("Histogram of f_star values", hist_image);
    cv::waitKey(1);
}

void create_video_from_histograms(const std::string& directory, const std::string& output_file, int fps = 10) {
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
        if (node.deleted) {
            continue;
        }

        int edges_before = node.edges.size();
        
        // Use the correct current_node reference
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [&](const Edge& edge) {
            return !is_edge_valid(graph, edge, node, threshold);
        }), node.edges.end());
        
        erased_edges += edges_before - node.edges.size();
        
        for (const auto& edge : node.edges) {
            if (!graph[edge.target_node].deleted) {
                remaining_edges++;
            }
        }
    }

    std::cout << "Erased edges: " << erased_edges << std::endl;
    std::cout << "Remaining edges: " << remaining_edges << std::endl;
    
    return erased_edges > 0;
}

void update_nodes(std::vector<Node>& graph, float o, float spring_constant, std::vector<size_t> valid_indices) {
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        float sum_w_f_tilde_k = 0.0f;
        float sum_w = 0.0f;

        for (const auto& edge : graph[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            size_t neighbor_node = edge.target_node;
            // sum_w_f_tilde_k += edge.certainty_factored * (graph[neighbor_node].f_tilde - spring_constant * edge.k);
            // sum_w += edge.certainty_factored;
            sum_w_f_tilde_k += edge.certainty * (graph[neighbor_node].f_tilde - spring_constant * edge.k);
            sum_w += edge.certainty;
        }

        // Calculate the new f_star for node i
        graph[i].f_star = (sum_w_f_tilde_k + o * graph[i].f_tilde) / (sum_w + o);
    }

    // Update f_tilde with the newly computed f_star values
    #pragma omp parallel for
    for (size_t j = 0; j < valid_indices.size(); ++j) {
        size_t i = valid_indices[j];
        if (graph[i].deleted) {
            continue;
        }
        graph[i].f_tilde = graph[i].f_star;
    }
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
        if (edges_deletion_round > 40 || invalid_edge_threshold <= 0.05) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations = num_iterations * 3;
            spring_constant = 1.0f;

            break;
        }
        // Solve for each spring constant
        for (int64_t i = -1; i < steps+1; ++i) {
            // Skip the first iterations if the warmup is already done
            if (edges_deletion_round > 1 && i < steps) {
                continue;
            }
            int num_iterations_iteration = num_iterations;
            float o_iteration = o;
            float spring_constant_iteration = i == -1 ? spring_constants[0] : spring_constants[i];
            if (i == -1 && edges_deletion_round == 0) {
                // Use a warmup iteration with 10x the spring constant
                num_iterations_iteration *= iterations_factor;
                o_iteration = o * o_factor;
                spring_constant_iteration = spring_factor;
            }
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
            for (int iter = 0; first_estimated_iteration ? true : iter < num_iterations_iteration; ++iter) {
                update_nodes(graph, o_iteration, spring_constant_iteration, valid_indices);

                if (iter % 100 == 0) {
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

                    // Calculate new weights factors
                    float scale = calculate_scale(graph, estimated_windings);
                    update_weights(graph, scale);

                    // Print
                    float mean_error = computeMeanError(graph, valid_gt_indices);
                    std::cout << "\rIteration: " << iter << " Mean Error to GT: " << mean_error << std::flush;  // Updates the same line

                    // escape if estimated windings reached
                    if (first_estimated_iteration) {
                        float min_f = min_f_star(graph);
                        float max_f = max_f_star(graph);
                        if (max_f - min_f > 1.1f * 360.0f * estimated_windings) {
                            break;
                        }
                    }
                }
            }
            // endline
            std::cout << std::endl;

            // After generating histograms, create a video from the images
            if (video_mode) {
                create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
            }
        }
        // After first edge deletion round remove the invalid edges
        if (edges_deletion_round >= 0) {
            // Remove edges with too much difference between f_star and k
            remove_invalid_edges(graph, invalid_edge_threshold);
        }
        find_largest_connected_component(graph);
        // Reduce the threshold by 20% each time
        invalid_edge_threshold *= 0.7f;
        invalid_edge_threshold -= 0.1f;
        if (invalid_edge_threshold < 0.05) {
            invalid_edge_threshold = 0.05;
        }
        std::cout << "Reducing invalid edges threshold to: " << invalid_edge_threshold << std::endl;

        // Assign winding angles again after removing invalid edges
        float scale = calculate_scale(graph, estimated_windings);
        assign_winding_angles(graph, scale);

        // Save the graph back to a binary file
        save_graph_to_binary("temp_output_graph.bin", graph);
        
        edges_deletion_round++;
    }
    // Assign winding angles to the graph
    float scale = calculate_scale(graph, estimated_windings);
    assign_winding_angles(graph, scale);

    if (video_mode) {
        // Calculate final histogram after all iterations
        calculate_histogram(graph, "final_histogram.png");
        // After generating all histograms, create a final video from the images
        create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
    }
}

void invert_winding_direction_graph(std::vector<Node>& graph) {
    for (size_t i = 0; i < graph.size(); ++i) {
        for (auto& edge : graph[i].edges) {
            if (edge.same_block) {
                float turns = edge.k / 360;
                turns = std::round(turns);
                if (std::abs(turns) > 1 || std::abs(turns) == 0) {
                    std::cout << "Inverting winding direction failed " << turns << std::endl;
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

    // delete all same_block edges
    for (auto& node : auto_graph_other_block) {
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [](const Edge& edge) {
            return edge.same_block;
        }), node.edges.end());
    }
    for (auto& node : auto_graph) {
        node.edges.erase(std::remove_if(node.edges.begin(), node.edges.end(), [](const Edge& edge) {
            return edge.same_block;
        }), node.edges.end());
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
    std::vector<Node> graph = load_graph_from_binary(input_graph_file, true, (static_cast<float>(program.get<int>("--z_min")) + 500) / 4, (static_cast<float>(program.get<int>("--z_max")) + 500) / 4, same_winding_factor);

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
        // Automatically determing winding direction
        if (program.get<bool>("--auto")) {
            auto_winding_direction(graph, &program);
        }
        // Solve the problem using a solve function
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
