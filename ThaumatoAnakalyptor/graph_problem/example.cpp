#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <cmath>
#include <limits>

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

    // Prepare the graph with empty nodes
    graph.resize(num_nodes);

    // Read each node's winding angle
    for (unsigned int i = 0; i < num_nodes; ++i) {
        infile.read(reinterpret_cast<char*>(&graph[i].f_init), sizeof(float));
        // Save to f_tilde and f_star as well
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

    infile.close();
    return graph;
}

float min_f_star(const std::vector<Node>& graph) {
    float min_f = std::numeric_limits<float>::max();

    for (const auto& node : graph) {
        if (node.f_star < min_f) {
            min_f = node.f_star;
        }
    }

    return min_f;
}

float max_f_star(const std::vector<Node>& graph) {
    float max_f = std::numeric_limits<float>::min();

    for (const auto& node : graph) {
        if (node.f_star > max_f) {
            max_f = node.f_star;
        }
    }

    return max_f;
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

int main() {
    std::string file_name = "graph.bin";
    std::vector<Node> graph = load_graph_from_binary(file_name);

    // Calculate and display the histogram of f_star values
    calculate_histogram(graph, "histogram.png");

    return 0;
}
