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

namespace fs = std::filesystem;

struct Edge {
    unsigned int target_node;
    float certainty;
    float k;
    bool same_block;
};

struct Node {
    float f_init;
    float f_tilde;
    float f_star;
    bool deleted = false;
    std::vector<Edge> edges;
};

std::vector<Node> load_graph_from_binary(const std::string &file_name) {
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
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        graph[i].f_tilde = graph[i].f_init;
        graph[i].f_star = graph[i].f_init;
    }

    // Read the adjacency list
    for (unsigned int i = 0; i < num_nodes; ++i) {
        unsigned int node_id;
        infile.read(reinterpret_cast<char*>(&node_id), sizeof(unsigned int));

        unsigned int num_edges;
        infile.read(reinterpret_cast<char*>(&num_edges), sizeof(unsigned int));

        for (unsigned int j = 0; j < num_edges; ++j) {
            Edge edge;
            infile.read(reinterpret_cast<char*>(&edge.target_node), sizeof(unsigned int));

            infile.read(reinterpret_cast<char*>(&edge.certainty), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.k), sizeof(float));
            infile.read(reinterpret_cast<char*>(&edge.same_block), sizeof(bool));

            graph[node_id].edges.push_back(edge);
        }
    }

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

float closest_valid_winding_angle(float f_init, float f_target) {
    int x = static_cast<int>(std::round((f_target - f_init) / 360.0f));
    return f_init + x * 360.0f;
}

void dfs(int node_index, const std::vector<Node>& graph, std::vector<bool>& visited, std::vector<int>& component) {
    std::stack<int> stack;
    stack.push(node_index);
    visited[node_index] = true;

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();
        component.push_back(current);

        for (const auto& edge : graph[current].edges) {
            if (!visited[edge.target_node]) {
                visited[edge.target_node] = true;
                stack.push(edge.target_node);
            }
        }
    }
}

void find_largest_connected_component(std::vector<Node>& graph) {
    int num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);
    std::vector<int> largest_component;

    for (int i = 0; i < num_nodes; ++i) {
        if (!visited[i] && !graph[i].deleted) {
            std::vector<int> current_component;
            dfs(i, graph, visited, current_component);

            if (current_component.size() > largest_component.size()) {
                largest_component = current_component;
            }
        }
    }

    // Flag nodes not in the largest connected component as deleted
    std::vector<bool> in_largest_component(num_nodes, false);
    for (int node_index : largest_component) {
        in_largest_component[node_index] = true;
    }

    int remaining_nodes = 0;
    for (int i = 0; i < num_nodes; ++i) {
        if (!in_largest_component[i]) {
            graph[i].deleted = true;
        }
        if (!graph[i].deleted) {
            remaining_nodes++;
        }
    }
    std::cout << "Remaining nodes: " << remaining_nodes << " out of " << num_nodes << std::endl;
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
    int num_nodes = graph.size();
    std::vector<bool> visited(num_nodes, false);

    // Find a non-deleted node in the largest connected component to start the DFS
    int start_node = -1;
    for (int i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            break;
        }
    }

    if (start_node == -1) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
        return;
    }

    // Perform DFS to assign f_star values
    dfs_assign_f_star(start_node, graph, visited);
}

using EdgeWithCertainty = std::pair<float, int>;  // {certainty, target_node}

void prim_mst_assign_f_star(int start_node, std::vector<Node>& graph) {
    size_t num_nodes = graph.size();
    std::vector<bool> in_mst(num_nodes, false);
    std::vector<float> min_k_delta(num_nodes, std::numeric_limits<float>::max());
    std::vector<int> parent(num_nodes, -1);

    // Priority queue to pick the edge with the minimum k delta
    std::priority_queue<EdgeWithCertainty, std::vector<EdgeWithCertainty>, std::greater<EdgeWithCertainty>> pq;

    pq.push({0.0f, start_node});
    min_k_delta[start_node] = 0.0f;

    while (!pq.empty()) {
        int u = pq.top().second;
        pq.pop();

        if (in_mst[u]) continue;
        in_mst[u] = true;

        for (const auto& edge : graph[u].edges) {
            int v = edge.target_node;
            float k_delta = std::abs((graph[v].f_tilde - graph[u].f_tilde) - edge.k) / std::abs(edge.k); // difference between BP solution and estimated k from the graph

            if (!in_mst[v] && k_delta < min_k_delta[v]) {
                min_k_delta[v] = k_delta;
                pq.push({k_delta, v});
                parent[v] = u;
            }
        }
    }

    // Set f_star for the root node (start_node)
    graph[start_node].f_star = graph[start_node].f_init;

    // Find for each node the children
    std::vector<std::vector<unsigned int>> children(num_nodes);
    for (unsigned int i = 0; i < num_nodes; ++i) {
        if (parent[i] != -1) {
            children[parent[i]].push_back(i);
        }
    }

    // Traverse the MST in a DFS manner to assign f_star values
    std::stack<int> stack;
    stack.push(start_node);

    while (!stack.empty()) {
        int current = stack.top();
        stack.pop();

        for (unsigned int child : children[current]) {
            // Find edge from parent to child
            Edge edge;
            for (const auto& e : graph[current].edges) {
                if (e.target_node == child) {
                    edge = e;
                    break;
                }
                // verbose if edge not found at last child
                if (e.target_node == graph[current].edges.back().target_node) {
                    std::cerr << "Edge not found for parent: " << current << " and child: " << child << std::endl;
                }
            }

            graph[child].f_star = closest_valid_winding_angle(graph[child].f_init, graph[current].f_tilde + edge.k);
            stack.push(child);
        }
    }
}

void assign_winding_angles(std::vector<Node>& graph) {
    int num_nodes = graph.size();
    
    // Find a non-deleted node in the largest connected component to start the MST
    int start_node = -1;
    for (int i = 0; i < num_nodes; ++i) {
        if (!graph[i].deleted) {
            start_node = i;
            break;
        }
    }

    if (start_node == -1) {
        std::cerr << "No non-deleted nodes found in the graph." << std::endl;
        return;
    }

    // Perform MST to assign f_star values
    prim_mst_assign_f_star(start_node, graph);
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

float exact_matching_score(const std::vector<Node>& graph) {
    // Assing closest valid winding angle to f_star based on f_init
    std::vector<Node> graph_copy = graph;
    for (auto& node : graph_copy) {
        node.f_star = closest_valid_winding_angle(node.f_init, node.f_star);
    }

    float score = 0.0;
    for (const auto& node : graph_copy) {
        for (const auto& edge : node.edges) {
            float diff = graph_copy[edge.target_node].f_star - node.f_star;
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

void calculate_histogram(const std::vector<Node>& graph, const std::string& filename = std::string(), int num_buckets = 512) {
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
    int hist_h = 1000;  // height of the histogram image, increased to 1000 pixels
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
    return std::abs(diff - edge.k) < 360 * threshold;
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

void update_nodes(std::vector<Node>& graph, float o, float spring_constant) {
    #pragma omp parallel for
    for (unsigned int i = 0; i < graph.size(); ++i) {
        if (graph[i].deleted) {
            continue;
        }
        float sum_w_f_tilde_k = 0.0f;
        float sum_w = 0.0f;

        for (const auto& edge : graph[i].edges) {
            if (graph[edge.target_node].deleted) {
                continue;
            }
            unsigned int neighbor_node = edge.target_node;
            sum_w_f_tilde_k += edge.certainty * (graph[neighbor_node].f_tilde - spring_constant * edge.k);
            sum_w += edge.certainty;
        }

        // Calculate the new f_star for node i
        graph[i].f_star = (sum_w_f_tilde_k + o * graph[i].f_tilde) / (sum_w + o);
    }

    // Update f_tilde with the newly computed f_star values
    #pragma omp parallel for
    for (unsigned int i = 0; i < graph.size(); ++i) {
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

// This is an example of a solve function that takes the graph and parameters as input
void solve(std::vector<Node>& graph, int argc, char** argv) {
    // Default values for parameters
    int num_iterations = 10000;
    float spring_constant = 2.0f;
    float o = 2.0f;
    float iterations_factor = 2.0f;
    float o_factor = 0.25f;
    float spring_factor = 3.0f;
    int steps = 5;

    // Override default values with command-line arguments if provided
    if (argc > 1) {
        num_iterations = std::atoi(argv[1]); // Convert the first argument to int
    }
    if (argc > 2) {
        o = std::atof(argv[2]); // Convert the second argument to float
    }
    if (argc > 3) {
        spring_constant = std::atof(argv[3]); // Convert the third argument to float
    }
    if (argc > 4) {
        steps = std::atoi(argv[4]); // Convert the fourth argument to int
    }
    if (argc > 5) {
        iterations_factor = std::atof(argv[5]); // Convert the fifth argument to float
    }
    if (argc > 6) {
        o_factor = std::atof(argv[6]); // Convert the sixth argument to float
    }
    if (argc > 7) {
        spring_factor = std::atof(argv[7]); // Convert the seventh argument to float
    }

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

    float invalid_edge_threshold = 1.1f;

    int edges_deletion_round = 0;
    while (true) {
        // Do 2 rounds of edge deletion
        if (edges_deletion_round > 1) {
            // Do last of updates with 3x times iterations and spring constant 1.0
            num_iterations = num_iterations * 3;
            spring_constant = 1.0f;

            break;
        }
        // Solve for each spring constant
        for (int64_t i = -1; i < steps+1; ++i) {
            int num_iterations_iteration = num_iterations;
            float o_iteration = o;
            float spring_constant_iteration = i == -1 ? spring_constants[0] : spring_constants[i];
            if (i == -1 && edges_deletion_round == 0) {
                // Use a warmup iteration with 10x the spring constant
                num_iterations_iteration *= iterations_factor;
                o_iteration = o * o_factor;
                spring_constant_iteration *= spring_factor;
            }
            else if (i == -1) {
                // Skip the warmup iteration for subsequent rounds
                continue;
            }
            else if (i == steps && edges_deletion_round == 1) {
                // Do last of updates with 3x times iterations and spring constant 1.0
                num_iterations_iteration *= 3.0f;
                spring_constant_iteration = 1.0f;
            }
            std::cout << "Spring Constant " << i << ": " << std::setprecision(10) << spring_constant_iteration << std::endl;
            for (int iter = 0; iter < num_iterations_iteration; ++iter) {
                update_nodes(graph, o_iteration, spring_constant_iteration);

                if (iter % 100 == 0) {
                    std::cout << "Iteration: " << iter << std::endl;

                    // Generate filename with zero padding
                    std::ostringstream filename;
                    filename << "histogram/histogram_" 
                            << std::setw(2) << std::setfill('0') << edges_deletion_round << "_"
                            << std::setw(max_index_digits) << std::setfill('0') << i+1 << "_"
                            << std::setw(max_iter_digits) << std::setfill('0') << iter << ".png";
                    // Calculate and display the histogram of f_star values
                    calculate_histogram(graph, filename.str());
                }
            }
            // After generating histograms, create a video from the images
            create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
        }
        // After first edge deletion round remove the invalid edges
        if (edges_deletion_round == 0) {
            // Remove edges with too much difference between f_star and k
            remove_invalid_edges(graph, invalid_edge_threshold);
        }
        find_largest_connected_component(graph);
        // Reduce the threshold by 20% each time
        invalid_edge_threshold *= 0.8f;
        invalid_edge_threshold -= 0.1f;
        if (invalid_edge_threshold < 0.05) {
            invalid_edge_threshold = 0.05;
        }
        std::cout << "Reducing invalid edges threshold to: " << invalid_edge_threshold << std::endl;

        // Assign winding angles again after removing invalid edges
        assign_winding_angles(graph);

        // Save the graph back to a binary file
        save_graph_to_binary("temp_output_graph.bin", graph);
        
        edges_deletion_round++;
    }
    // Calculate final histogram after all iterations
    calculate_histogram(graph, "final_histogram.png");
    // After generating all histograms, create a final video from the images
    create_video_from_histograms(histogram_dir, "winding_angle_histogram.avi", 10);
}

int main(int argc, char** argv) {
    std::string file_name = "graph.bin";
    std::vector<Node> graph = load_graph_from_binary(file_name);

    // Calculate the exact matching loss
    float exact_score = exact_matching_score(graph);
    std::cout << "Exact Matching Score: " << exact_score << std::endl;

    // Calculate the approximate matching loss
    float approx_loss = approximate_matching_loss(graph, 1.0f);
    std::cout << "Approximate Matching Loss: " << approx_loss << std::endl;

    // Calculate and display the histogram of f_star values
    calculate_histogram(graph);

    // Solve the problem using a solve function
    solve(graph, argc, argv);

    // print the min and max f_star values
    std::cout << "Min f_star: " << min_f_star(graph) << std::endl;
    std::cout << "Max f_star: " << max_f_star(graph) << std::endl;

    // Assign winding angles to the graph
    assign_winding_angles(graph);

    // Save the graph back to a binary file
    save_graph_to_binary("output_graph.bin", graph);

    return 0;
}
