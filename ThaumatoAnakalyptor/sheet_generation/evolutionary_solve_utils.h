// EvolutionaryAlgorithm.h

#ifndef EVOLUTIONARY_ALGORITHM_H
#define EVOLUTIONARY_ALGORITHM_H

#include <vector>
#include <thread>
#include <iostream>
#include <mutex>
#include <functional>
#include <random>
#include <memory>
#include <iomanip> // For std::setw and std::setfill

class WeightedUF {
private:
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> size;
    std::unordered_map<int, int> weight; // Weight to the parent
    std::unordered_map<int, bool> burned; // Weight to the parent

public:
    // Constructor: Initialize union-find structure with known node count.
    WeightedUF() {
        // Initial setup is unnecessary here because maps will default-initialize elements.
    }

    // Find the root of node p, and compress the path
    int find(int p, int& path_weight) {
        if (parent.find(p) == parent.end()) return p; // If p isn't in the map, it has no parent.

        int root = p;
        path_weight = weight[root];

        // Find the actual root
        while (root != parent[root]) {
            root = parent[root];
            path_weight += weight[root];  // Accumulate weights
        }
        int total_weight = path_weight;

        // // Path compression
        while (p != root) {
            int next = parent[p];
            parent[p] = root;
            burned[root] = burned[root] || burned[p];  // Entire component is burned if any part is burned
            int total_weight_ = total_weight - weight[p];  // Update total weight delayed
            weight[p] = total_weight;  // Set the new weight to reflect total path weight
            total_weight = total_weight_;  // Update total weight for next iteration
            p = next;
        }

        return root;
    }

    bool merge(int x, int y, int k) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x; // Initialize if not already present
            weight[x] = 0; // Initial weight to self is 0
            burned[x] = false; // Initial burn status is false
            size[x] = 1; // Initial size is 1
        }
        if (parent.find(y) == parent.end()) {
            parent[y] = y;
            weight[y] = 0;
            burned[x] = false; // Initial burn status is false
            size[y] = 1;
        }

        int weightX = 0, weightY = 0;
        int rootX = find(x, weightX);
        int rootY = find(y, weightY);
        if (rootX == rootY) {
            // Check if k is valid
            int connection_weight = weightX - weightY;
            return connection_weight == k;
        }

        // Merge by size and update weights
        if (size[rootX] < size[rootY]) {
            parent[rootX] = rootY;
            weight[rootX] = weightY + k - weightX;  // Correctly maintain the weight difference
            burned[rootY] = burned[rootX] || burned[rootY];  // Entire component is burned if any part is burned
            size[rootY] += size[rootX];
            size.erase(rootX); // Delete the old size
        } else {
            parent[rootY] = rootX;
            weight[rootY] = weightX - k - weightY;  // Ensure symmetry in weight handling
            burned[rootX] = burned[rootX] || burned[rootY];  // Entire component is burned if any part is burned
            size[rootX] += size[rootY];
            size.erase(rootY); // Delete the old size
        }
        return true;
    }

    // Check if x and y are connected, and optionally retrieve the connection weight
    bool connected(int x, int y, int& connection_weight) {
        int weightX = 0, weightY = 0;
        int rootX = find(x, weightX);
        int rootY = find(y, weightY);
        connection_weight = weightX - weightY;
        return rootX == rootY;
    }

    void burn(int x) {
        int weightX = 0;
        int rootX = find(x, weightX);
        burned[rootX] = true;
    }

    std::tuple<int, int, double>extract_largest_unburned_component() {
        double total_unburned_size = 0;
        int max_size = 0;
        int max_root = -1;
        for (auto& [root, size_root] : size) {
            if (!burned[root]) {
                total_unburned_size += size_root * std::log(size_root);
                if (size_root > max_size) {
                    max_size = size_root;
                    max_root = root;
                }
            }
        }
        return {max_root, max_size, total_unburned_size};
    }

    bool edge_part_of_component(int component_root, int node1, int node2, int k) {
        int weightX = 0, weightY = 0;
        int rootX = find(node1, weightX);
        int rootY = find(node2, weightY);
        bool both_root_component = rootX == component_root && rootY == component_root;
        bool valid_edge = weightX - weightY == k;
        return both_root_component && valid_edge;
    }

    int get_size(int root) {
        if (size.find(root) == size.end()) {
            return 0;
        }
        return size[root];
    }
};

// Function to check if the given edge between node1 and node2 is valid based on 'k'
bool check_valid(int connection_weight, int k) {
    return connection_weight == k;
}

bool check_very_bad_direction(int connection_weight, int k) {
    bool difference_greater_1 = std::abs(connection_weight - k) > 2;
    return difference_greater_1;
}

// Function to merge two components
bool merge_components(WeightedUF &uf, int node1, int node2, int k) {
    return uf.merge(node1, node2, k);
}

// Function to add a node to an existing component
void add_node_to_component(WeightedUF &uf, int node1, int node2, int k) {
    // This function is conceptual because union-find inherently manages this
    // Instead, we directly merge node1 and node2 with the given k
    uf.merge(node1, node2, k);
}

// Define a hash function for the tuple
struct hash_tuple {
    template <class T>
    std::size_t operator()(const T& tuple) const {
        auto hash1 = std::hash<int>{}(std::get<0>(tuple));
        auto hash2 = std::hash<int>{}(std::get<1>(tuple));
        auto hash3 = std::hash<int>{}(std::get<2>(tuple));
        auto hash4 = std::hash<int>{}(std::get<3>(tuple));

        return hash1 ^ hash2 ^ hash3 ^ hash4;  // Combine the hash values
    }
};

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual__(int length_individual, float* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, double valid_edges_factor, int legth_initial_component, int* initial_component, bool build_valid_edges) {
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    // Add the initial components to the graph
    for (int i = 0; i < legth_initial_component; i++) {
        int node1 = -1;
        int node2 = initial_component[i*2];
        int k = initial_component[i*2 + 1];
        add_node_to_component(uf, node1, node2, k);
    }

    std::vector<int> sorted_indices(graph_raw_length);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    double invalid_edges_count = 1;
    double invalid_by_k_direction = 0;
    int* valid_edges;
    float max_building_edge_gene = individual[length_individual-2];
    float max_valid_edge_gene = individual[length_individual-1];
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[graph_raw_length];
    }
    else {
        valid_edges = new int[1];
    }
    int max_valid_i = (int) (valid_edges_factor * graph_raw_length);
    for (int i=0; i < graph_raw_length; i++) {
        int index = sorted_indices[i];
        int node1 = graph_raw[4*index];
        int node2 = graph_raw[4*index + 1];
        int k = graph_raw[4*index + 2];
        int certainty = graph_raw[4*index + 3];

        double k_factor = (k == 0) ? factor_0 : factor_not_0;
        double score_edge = k_factor * ((double)certainty);
        
        // if (min_i_invalid <= i && max_valid_edge_gene < individual[index]) { // if the gene is less than the max valid edge gene, we can skip this edge, the individual unselected it
        //     continue;
        // }
        int connection_weight;
        bool connected_nodes = uf.connected(node1, node2, connection_weight);
        if (!connected_nodes) { // if not connected we can unconditionally add the edge
            if (i < max_valid_i) { // if the gene is smaller than the max building edge gene, we can add the edge
                add_node_to_component(uf, node1, node2, k);
                valid_edges_count += score_edge;
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
            }
            else {
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
            }
        } else {
            if (!check_valid(connection_weight, k)) {
                // invalid edge that was not marked as invalid by the individual (max_valid_edge_gene and min_i_invalid)
                // therefore BURN the component
                // uf.burn(node1);
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
                invalid_edges_count += score_edge; // Bad edge
                if (!check_very_bad_direction(connection_weight, k)) {
                    invalid_by_k_direction += score_edge; // Even worse edge (k in (-1, 1) and weight has other sign ...)
                }
            }
            else {
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
                valid_edges_count += score_edge;
            }
        }
    }

    double fitness_component = valid_edges_count - invalid_edges_count; // - invalid_by_k_direction;

    return {fitness_component, valid_edges};
}

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual(int length_individual, float* individual, int graph_raw_length, int* graph_raw, bool* same_block, int length_bad_edges, int* bad_edges, double factor_0, double factor_not_0, double factor_bad, int legth_initial_component, int* initial_component, bool build_valid_edges) {
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    // Add the initial components to the graph
    for (int i = 0; i < legth_initial_component; i++) {
        int node1 = -1;
        int node2 = initial_component[i*2];
        int k = initial_component[i*2 + 1];
        add_node_to_component(uf, node1, node2, k);
    }

    std::vector<int> sorted_indices(graph_raw_length);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    double invalid_edges_count = 1;
    double invalid_by_k_direction = 0;
    double invalid_by_bad_edges = 0;
    int* valid_edges;
    float max_building_edge_gene = individual[length_individual-2];
    float max_valid_edge_gene = individual[length_individual-1];
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[graph_raw_length];
    }
    else {
        valid_edges = new int[1];
    }
    int min_i_invalid = (int) (factor_bad * graph_raw_length);
    for (int i=0; i < graph_raw_length; i++) {
        int index = sorted_indices[i];
        int node1 = graph_raw[4*index];
        int node2 = graph_raw[4*index + 1];
        int k = graph_raw[4*index + 2];
        int certainty = graph_raw[4*index + 3];

        double k_factor = same_block[index] ? factor_not_0 : factor_0;
        double score_edge = k_factor * ((double)certainty);
        
        // if (min_i_invalid <= i && max_valid_edge_gene < individual[index]) { // if the gene is less than the max valid edge gene, we can skip this edge, the individual unselected it
        //     continue;
        // }
        int connection_weight;
        bool connected_nodes = uf.connected(node1, node2, connection_weight);
        if (!connected_nodes) { // if not connected we can unconditionally add the edge
            if (max_building_edge_gene >= individual[index]) { // if the gene is smaller than the max building edge gene, we can add the edge
                add_node_to_component(uf, node1, node2, k);
                valid_edges_count += score_edge;
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
            }
            else {
                valid_edges_count += 0.25 * score_edge;
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
            }
        } else {
            if (!check_valid(connection_weight, k)) {
                // invalid edge that was not marked as invalid by the individual (max_valid_edge_gene and min_i_invalid)
                // therefore BURN the component
                uf.burn(node1);
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
                if (max_building_edge_gene >= individual[index]) { // Only count invalid edges that were not unselected by the individual
                    invalid_edges_count += score_edge;
                }
                // if (check_very_bad_direction(connection_weight, k)) {
                //     invalid_by_k_direction += score_edge; // Even worse edge (k and weight has dif > 1 ...)
                // }
            }
            else {
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
                valid_edges_count += score_edge;
            }
        }
    }

    for (int i=0; i<length_bad_edges; i++) {
        int node1 = bad_edges[4*i];
        int node2 = bad_edges[4*i + 1];
        int k = bad_edges[4*i + 2];
        int certainty = bad_edges[4*i + 3];

        double score_edge = factor_bad * ((double)certainty);
        
        int connection_weight;
        bool connected_nodes = uf.connected(node1, node2, connection_weight);
        if (connected_nodes) { // Check if the nodes of the same "block" are connected
            if (check_valid(connection_weight, k)) { // Bad edge is valid, penalize the fitness
                invalid_by_bad_edges += score_edge;
            }
        }
    }

    double fitness_component = valid_edges_count - invalid_edges_count - 2*invalid_by_k_direction - invalid_by_bad_edges;

    return {fitness_component, valid_edges};
}

// Main function to build the graph from given inputs
// build_graph_from_individual_partially_working
std::pair<double, int*> build_graph_from_individual_partially_working(int length_individual, float* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, double max_invalid_edges_factor, int legth_initial_component, int* initial_component, bool build_valid_edges) {
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    // Add the initial components to the graph
    for (int i = 0; i < legth_initial_component; i++) {
        int node1 = -1;
        int node2 = initial_component[i*2];
        int k = initial_component[i*2 + 1];
        add_node_to_component(uf, node1, node2, k);
    }

    std::vector<int> sorted_indices(graph_raw_length);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    double invalid_edges_count = 1;
    double invalid_by_k_direction = 0;
    int* valid_edges;
    float max_building_edge_gene = individual[length_individual-2];
    float max_valid_edge_gene = individual[length_individual-1];
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[graph_raw_length];
    }
    else {
        valid_edges = new int[1];
    }
    int min_i_invalid = (int) (max_invalid_edges_factor * graph_raw_length);
    for (int i=0; i < graph_raw_length; i++) {
        int index = sorted_indices[i];
        int node1 = graph_raw[4*index];
        int node2 = graph_raw[4*index + 1];
        int k = graph_raw[4*index + 2];
        int certainty = graph_raw[4*index + 3];

        double k_factor = (k == 0) ? factor_0 : factor_not_0;
        double score_edge = k_factor * ((double)certainty);
        
        if (min_i_invalid <= i && max_valid_edge_gene < individual[index]) { // if the gene is less than the max valid edge gene, we can skip this edge, the individual unselected it
            if (build_valid_edges){
                    valid_edges[index] = 0;
            }
            continue;
        }
        int connection_weight;
        bool connected_nodes = uf.connected(node1, node2, connection_weight);
        if (!connected_nodes) { // if not connected we can unconditionally add the edge
            if (max_building_edge_gene >= individual[index]) { // if the gene is smaller than the max building edge gene, we can add the edge
                add_node_to_component(uf, node1, node2, k);
                valid_edges_count += score_edge;
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
            }
            else {
                valid_edges_count += 0.5 * score_edge;
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
            }
        } else {
            if (!check_valid(connection_weight, k)) {
                // invalid edge that was not marked as invalid by the individual (max_valid_edge_gene and min_i_invalid)
                // therefore BURN the component
                if (max_building_edge_gene >= individual[index]) { // Only count invalid edges that were not unselected by the individual
                    uf.burn(node1);
                    invalid_edges_count += score_edge;
                }
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
            }
            else {
                if (build_valid_edges){
                        valid_edges[index] = 1;
                }
                valid_edges_count += score_edge;
            }
        }
    }

    double fitness_component = valid_edges_count - invalid_edges_count;
    fitness_component = std::max(0.0d, fitness_component); // make sure the fitness is not negative

    // double fitness_component = valid_edges_count / invalid_edges_count;

    // extract the largest unburned component
    std::tuple<int, int, double> largest_component = uf.extract_largest_unburned_component();
    int max_component_root = std::get<0>(largest_component);
    int max_component_size = std::get<1>(largest_component);
    double total_unburned_size = std::get<2>(largest_component);

    fitness_component += max_component_size*max_component_size + total_unburned_size;
    // fitness_component += max_component_size * total_unburned_size;
    if (build_valid_edges) {
        std::cout << "Valid edges count: " << (int)valid_edges_count << std::endl;
        std::cout << "Invalid edges count: " << (int)invalid_edges_count << std::endl;
        std::cout << "Max component size: " << max_component_size << std::endl;
        std::cout << "Fitness: " << fitness_component << std::endl;
    }

    // if (build_valid_edges) {
    //     std::cout << "Max component size: " << max_component_size << std::endl;
    //     for (int i=0; i < graph_raw_length; i++) {
    //         int index = sorted_indices[i];
    //         int node1 = graph_raw[4*index];
    //         int node2 = graph_raw[4*index + 1];
    //         int k = graph_raw[4*index + 2];

    //         bool valid_edge = uf.edge_part_of_component(max_component_root, node1, node2, k);
    //         if (valid_edge) {
    //             valid_edges[index] = 1;
    //         } else {
    //             valid_edges[index] = 0;
    //         }
    //     }
    // }

    return {fitness_component, valid_edges};
}

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual_patch(int length_individual, float* individual, int graph_raw_length, int* graph_raw, bool* same_block, int length_bad_edges, int* bad_edges, double factor_0, double factor_not_0, double factor_bad, bool build_valid_edges) {
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    std::unordered_map<std::tuple<int, int, int, int>, int, hash_tuple> visited_subvolumes;

    std::vector<int> sorted_indices(graph_raw_length);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    int* valid_edges;
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[graph_raw_length];
    }
    else {
        valid_edges = new int[1];
    }

    for (int i=0; i < graph_raw_length; i++) {
        int index = sorted_indices[i];
        int node1 = graph_raw[12*index];
        int node2 = graph_raw[12*index + 1];
        int k = graph_raw[12*index + 2];
        int certainty = graph_raw[12*index + 3];
        int node1_subvolume_0 = graph_raw[12*index + 4];
        int node1_subvolume_1 = graph_raw[12*index + 5];
        int node1_subvolume_2 = graph_raw[12*index + 6];
        int node2_subvolume_0 = graph_raw[12*index + 7];
        int node2_subvolume_1 = graph_raw[12*index + 8];
        int node2_subvolume_2 = graph_raw[12*index + 9];
        int assigned_k1 = graph_raw[12*index + 10];
        int assigned_k2 = graph_raw[12*index + 11];

        if (certainty <= 0) {
            std::cout << "Invalid certainty value: " << certainty << std::endl;
        }

        auto node1_subvolume = std::make_tuple(node1_subvolume_0, node1_subvolume_1, node1_subvolume_2, assigned_k1);
        auto node2_subvolume = std::make_tuple(node2_subvolume_0, node2_subvolume_1, node2_subvolume_2, assigned_k2);
        // Check for visited subvolume vs found components
        if (visited_subvolumes.find(node1_subvolume) != visited_subvolumes.end()) {
            if (visited_subvolumes[node1_subvolume] != node1) {
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
                continue;
            }
        }
        else if (visited_subvolumes.find(node2_subvolume) != visited_subvolumes.end()) {
            if (visited_subvolumes[node2_subvolume] != node2) {
                if (build_valid_edges){
                        valid_edges[index] = 0;
                }
                continue;
            }
        }

        double k_factor = same_block[index] ? factor_not_0 : factor_0;
        double score_edge = k_factor * ((double)certainty);

        int connection_weight1;
        
        valid_edges_count += score_edge;
        if (!(uf.connected(node1, node2, connection_weight1))) {
            // std::cout << "Merging components: " << node1 << " " << node2 << " " << k << std::endl;
            add_node_to_component(uf, node1, node2, k);
            
            if (build_valid_edges){
                valid_edges[index] = 1;
            }
            // Add the subvolumes to the visited set
            visited_subvolumes[node1_subvolume] = node1;
            visited_subvolumes[node2_subvolume] = node2;
        } else {
            int connection_weight2;
            uf.connected(node2, node1, connection_weight2);
            if (connection_weight1 != -connection_weight2) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " " << connection_weight2 << std::endl;
            }
            if (!check_valid(connection_weight1, k)) {
                valid_edges_count -= 2*score_edge; // Invalid edge, subtract its score
                if (build_valid_edges){
                    valid_edges[index] = 0;
                }
            }
            else {
                if (build_valid_edges){
                    valid_edges[index] = 1;
                }
                if (connection_weight1 != k && connection_weight2 != -k) {
                    std::cout << "Invalid connection weight: " << connection_weight1 << " k " << k << std::endl;
                }
                // Add the subvolumes to the visited set
                visited_subvolumes[node1_subvolume] = node1;
                visited_subvolumes[node2_subvolume] = node2;
            }    
        }
    }
    // std::cout << "Valid edges count: " << (int)valid_edges_count << std::endl;
    return {valid_edges_count, valid_edges};
}

class Individual {
public:
    float* genes;
    float* mutation_chance; // Array of mutation chances per gene
    float* crossover_chance; // Array of crossover chances per gene
    int* gene_direction; // Array of gene directions
    bool* fixed_genes; // Array of fixed genes
    double fitness;
    int genes_length;

    Individual(int size) : fitness(0) {
        genes = new float[size];
        genes_length = size;
        mutation_chance = new float[size];
        crossover_chance = new float[size];
        gene_direction = new int[size];
        fixed_genes = new bool[size];
        // Initialize mutation chances and modulo values
        for (int i = 0; i < size; ++i) {
            genes[i] =  (rand() % 100) / 100.0; // random in 0, 1 float
            mutation_chance[i] = 0.001;  // Default mutation chance, can be adjusted
            crossover_chance[i] = 0.1;  // Default crossover chance, can be adjusted
            gene_direction[i] = 0; // Default direction is 1
            fixed_genes[i] = false;
        }
    }
};

class EvolutionaryAlgorithm {
private:
    std::vector<Individual> pool;
    std::vector<Individual*> population;
    std::vector<Individual*> new_population;
    std::vector<double> genes_performance;
    int population_size;
    const double crossover_rate = 0.01;
    const double mutation_rate = 0.1;
    const double fix_percentage = 0.05;
    const double max_fix_percentage = 0.80;
    const int fix_step = 50; // Fix the best genes every 20 epochs. INFORMATION: the larger the problem length (graph length), the more fix steps are needed. try 20, 50, 100 at least and see how it converges.
    int tournament_size = 5;
    std::function<double(const Individual&, int*, bool*, int, int*, int, double, double, double, int, int*)> evaluate_function;
    int* graph;
    bool* same_block;
    int length_bad_edges;
    int* bad_edges;
    int graph_length;
    double factor_0;
    double factor_not_0;
    double factor_bad;
    int legth_initial_component;
    int* initial_component;
    int num_threads;

    std::vector<std::default_random_engine> generators;
    std::uniform_real_distribution<float> distribution;

    Individual best_individual;

    void fix_genes(int nr_fixes) {
        if (nr_fixes * fix_percentage > max_fix_percentage) {
            return;
        }
        std::vector<int> sorted_indices(graph_length);
        std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
        std::sort(sorted_indices.begin(), sorted_indices.end(), 
                [this](int i1, int i2) { return this->genes_performance[i1] < this->genes_performance[i2]; });
        // Fix all the best genes, genes are ordered from best to worst ascendingly
        int num_fixed_genes = nr_fixes * fix_percentage * graph_length;
        for (int indv = 0; indv < population_size; ++indv) {
            for (int i = 0; i < num_fixed_genes; ++i) {
                int index = sorted_indices[i];
                population[indv]->fixed_genes[index] = true;
            }
            // Reset the genes
            for (int i = num_fixed_genes; i < graph_length - num_fixed_genes; ++i) {
                if (i >= graph_length) {
                    break;
                }
                int index = sorted_indices[i];
                population[indv]->genes[index] = (rand() % 100) / 100.0; // random in 0, 1 float
                population[indv]->mutation_chance[index] = 0.001;  // Default mutation chance, can be adjusted
                population[indv]->crossover_chance[index] = 0.1;  // Default crossover chance, can be adjusted
                population[indv]->gene_direction[index] = 0; // Default direction is 1
                population[indv]->fixed_genes[index] = false;
            }
            // Fix worst genes
            for (int i = graph_length - num_fixed_genes; i < graph_length; ++i) {
                int index = sorted_indices[i];
                population[indv]->fixed_genes[index] = true;
            }
            // // Reset the last genes
            // for (int i = graph_length; i < population[indv]->genes_length; ++i) {
            //     population[indv]->genes[i] = (rand() % 100) / 100.0; // random in 0, 1 float
            //     population[indv]->mutation_chance[i] = 0.001;  // Default mutation chance, can be adjusted
            //     population[indv]->crossover_chance[i] = 0.1;  // Default crossover chance, can be adjusted
            //     population[indv]->gene_direction[i] = 0; // Default direction is 1
            //     population[indv]->fixed_genes[i] = false;
            // }
        }
        for (int indv = 0; indv < population_size; ++indv) {
            for (int i = 0; i < num_fixed_genes; ++i) {
                int index = sorted_indices[i];
                new_population[indv]->fixed_genes[index] = true;
            }
            // Reset the genes
            for (int i = num_fixed_genes; i < graph_length - num_fixed_genes; ++i) {
                if (i >= graph_length) {
                    break;
                }
                int index = sorted_indices[i];
                new_population[indv]->genes[index] = (rand() % 100) / 100.0; // random in 0, 1 float
                new_population[indv]->mutation_chance[index] = 0.001;  // Default mutation chance, can be adjusted
                new_population[indv]->crossover_chance[index] = 0.1;  // Default crossover chance, can be adjusted
                new_population[indv]->gene_direction[index] = 0; // Default direction is 1
                new_population[indv]->fixed_genes[index] = false;
            }
            // Fix worst genes
            for (int i = graph_length - num_fixed_genes; i < graph_length; ++i) {
                int index = sorted_indices[i];
                population[indv]->fixed_genes[index] = true;
            }
            // // Reset the last genes
            // for (int i = graph_length; i < population[indv]->genes_length; ++i) {
            //     population[indv]->genes[i] = (rand() % 100) / 100.0; // random in 0, 1 float
            //     population[indv]->mutation_chance[i] = 0.001;  // Default mutation chance, can be adjusted
            //     population[indv]->crossover_chance[i] = 0.1;  // Default crossover chance, can be adjusted
            //     population[indv]->gene_direction[i] = 0; // Default direction is 1
            //     population[indv]->fixed_genes[i] = false;
            // }
        }
        // Reset the performance of the genes
        for (int i = 0; i < graph_length; ++i) {
            genes_performance[i] = 0.0;
        }
    }

    void track_best_performing_genes(int nr_fixes) {
        if (nr_fixes * fix_percentage >= max_fix_percentage) {
            return;
        }
        // Track the ranking of every gene in the population
        for (int i = 0; i < population_size; ++i) {
            for (int j = 0; j < graph_length; ++j) {
                genes_performance[j] += population[i]->genes[j];
            }
        }
    }

    std::tuple<double, double> evaluate(int epoch) {
        int num_threads = std::thread::hardware_concurrency();  // Get the number of threads supported by the hardware
        int chunk_size = std::ceil(population.size() / static_cast<double>(num_threads));
        std::vector<std::thread> threads;

        // random edges valid percentage for this generation
        double valid_edges_percentage = distribution(generators[0]) * 1.0;
        
        // Lambda to process a slice of the population
        auto process_chunk = [this, valid_edges_percentage](int start, int end) {
            for (int i = start; i < end && i < this->population.size(); ++i) {
                double fitness = this->evaluate_function(*this->population[i], this->graph, this->same_block, this->length_bad_edges, this->bad_edges, this->graph_length, this->factor_0, this->factor_not_0, this->factor_bad, this->legth_initial_component, this->initial_component);
                this->population[i]->fitness = fitness;
            }
        };

        // Creating and assigning threads to each chunk of the population
        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = start + chunk_size;
            threads.emplace_back(process_chunk, start, end);
        }

        // Join all threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Extract the best fitness to return
        Individual best_individual_generation = *population[0];
        double mean_fitness = 0;
        int pop_size = population.size();
        for (auto& ind : population) {
            mean_fitness += ind->fitness / pop_size;
            if (ind->fitness > best_individual_generation.fitness) best_individual_generation = *ind;
        }

        // get cross-generational comparable fitness score with fixed valid edges percentage
        double fitness = this->evaluate_function(best_individual_generation, this->graph, this->same_block, this->length_bad_edges, this->bad_edges, this->graph_length, this->factor_0, this->factor_not_0, factor_bad, this->legth_initial_component, this->initial_component);
        best_individual_generation.fitness = fitness;

        if (best_individual_generation.fitness > best_individual.fitness) {
            best_individual.fitness = best_individual_generation.fitness;
            for (int i = 0; i < best_individual.genes_length; ++i) {
                best_individual.genes[i] = best_individual_generation.genes[i];
                best_individual.mutation_chance[i] = best_individual_generation.mutation_chance[i];
                best_individual.crossover_chance[i] = best_individual_generation.crossover_chance[i];
                best_individual.gene_direction[i] = best_individual_generation.gene_direction[i];
                best_individual.fixed_genes[i] = best_individual_generation.fixed_genes[i];
            }
        }

        // Track the best performing genes
        track_best_performing_genes(epoch / fix_step);

        // Fix the best genes periodically
        if (epoch % fix_step == 0) {
            fix_genes(epoch / fix_step);
        }

        return {best_individual_generation.fitness, mean_fitness};
    }

    Individual* tournamentSelection(std::default_random_engine& generator) {
        std::vector<Individual*> tournament(tournament_size);
        for (int i = 0; i < tournament_size; ++i) {
            int index = distribution(generator) * population_size;
            tournament[i] = population[index];
        }

        Individual* fittest = tournament[0];
        for (auto& contender : tournament) {
            if (contender->fitness > fittest->fitness) fittest = contender;
        }
        return fittest;
    }

    void crossover(Individual& parent1, Individual& parent2, Individual& child, std::default_random_engine& generator) {
        for (int i = 0; i < parent1.genes_length; ++i) {
            float crossover_chance = distribution(generator);
            child.genes[i] = (crossover_rate < crossover_chance) ? parent1.genes[i] : parent2.genes[i];
        }
    }

    void mutate(Individual& individual, std::default_random_engine& generator) {
        for (int i = 0; i < individual.genes_length; ++i) {
            if (individual.fixed_genes[i]) { // Skip mutating fixed genes
                continue;
            }
            if (distribution(generator) < mutation_rate) {
                individual.genes[i] = distribution(generator);  // Apply mutation based on the chance
                float mutation_adaption = distribution(generator);
                if (mutation_adaption < 0.33) {
                    individual.mutation_chance[i] *= 0.9;  // Decrease mutation chance
                }
                else if (mutation_adaption < 0.66) {
                    individual.mutation_chance[i] *= 1.1;  // Increase mutation chance
                }
                float crossover_adaption = distribution(generator);
                if (crossover_adaption < 0.33) {
                    individual.crossover_chance[i] *= 0.9;  // Decrease crossover chance
                }
                else if (crossover_adaption < 0.66) {
                    individual.crossover_chance[i] *= 1.1;  // Increase crossover chance
                }
            }
        }
    }

    // void crossover(Individual& parent1, Individual& parent2, Individual& child, std::default_random_engine& generator) {
    //     if (distribution(generator) < 0.5) { // Crossover based on sorted indices
    //         std::vector<int> sorted_indices(graph_length);
    //         std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    //         std::sort(sorted_indices.begin(), sorted_indices.end(), 
    //                 [&parent1](int i1, int i2) { return parent1.genes[i1] < parent1.genes[i2]; });

    //         int crossover_point = distribution(generator) * graph_length;
    //         for (int i = 0; i < graph_length; ++i) {
    //             child.genes[i] = (i < crossover_point) ? parent1.genes[sorted_indices[i]] : parent2.genes[sorted_indices[i]];
    //             child.crossover_chance[i] = (i < crossover_point) ? parent1.crossover_chance[sorted_indices[i]] : parent2.crossover_chance[sorted_indices[i]];
    //             child.mutation_chance[i] = (i < crossover_point) ? parent1.mutation_chance[sorted_indices[i]] : parent2.mutation_chance[sorted_indices[i]];
    //             child.gene_direction[i] = (i < crossover_point) ? parent1.gene_direction[sorted_indices[i]] : parent2.gene_direction[sorted_indices[i]];
    //         }
    //         // take over the rest of the genes from the fittest parent
    //         for (int i = graph_length; i < parent1.genes_length; ++i) {
    //             child.genes[i] = parent1.genes[i];
    //             child.crossover_chance[i] = parent1.crossover_chance[i];
    //             child.mutation_chance[i] = parent1.mutation_chance[i];
    //             child.gene_direction[i] = parent1.gene_direction[i];
    //         }
    //     }
    //     else {
    //         for (int i = 0; i < parent1.genes_length; ++i) {
    //             float crossover_chance = distribution(generator);
    //             child.genes[i] = (crossover_rate < crossover_chance) ? parent1.genes[i] : parent2.genes[i];
    //             child.crossover_chance[i] = (crossover_rate < crossover_chance) ? parent1.crossover_chance[i] : parent2.crossover_chance[i];
    //             child.mutation_chance[i] = (crossover_rate < crossover_chance) ? parent1.mutation_chance[i] : parent2.mutation_chance[i];
    //             child.gene_direction[i] = (crossover_rate < crossover_chance) ? parent1.gene_direction[i] : parent2.gene_direction[i];
    //         }
    //     }
    // }

    // void mutate(Individual& individual, std::default_random_engine& generator) {
    //     for (int i = 0; i < individual.genes_length; ++i) {
    //         if (distribution(generator) < mutation_rate) {
    //             if (distribution(generator) < 0.75) {
    //                 if (distribution(generator) < 0.1) {
    //                     individual.gene_direction[i] = 2.0*distribution(generator) - 1.0;  // Randomize direction
    //                 }
    //                 else {
    //                     individual.gene_direction[i] *= 0.5;  // Decrease direction
    //                 }
    //             }
    //             else {
    //                 individual.gene_direction[i] = distribution(generator) - individual.genes[i];
    //             }
    //             individual.genes[i] = std::min(std::max(0.0f, individual.genes[i] + individual.gene_direction[i]), 1.0f);  // Apply mutation based on the chance

    //             float mutation_adaption = distribution(generator);
    //             if (mutation_adaption < 0.33) {
    //                 individual.mutation_chance[i] *= 0.9;  // Decrease mutation chance
    //             }
    //             else if (mutation_adaption < 0.66) {
    //                 individual.mutation_chance[i] *= 1.1;  // Increase mutation chance
    //             }
    //             else if (0.4999 < mutation_adaption && mutation_adaption < 0.5001) {
    //                 individual.mutation_chance[i] = distribution(generator);  // Randomize mutation chance
    //             }
    //             float crossover_adaption = distribution(generator);
    //             if (crossover_adaption < 0.33) {
    //                 individual.crossover_chance[i] *= 0.9;  // Decrease crossover chance
    //             }
    //             else if (crossover_adaption < 0.66) {
    //                 individual.crossover_chance[i] *= 1.1;  // Increase crossover chance
    //             }
    //             else if (0.4999 < crossover_adaption && crossover_adaption < 0.5001) {
    //                 individual.crossover_chance[i] = distribution(generator);  // Randomize crossover chance
    //             }
    //         }
    //     }
    // }

    void shift(Individual& individual, std::default_random_engine& generator) {
        // shift mutation where a random consecutive sorted chunk of the individual is shifted by a random amount
        if (distribution(generator) < individual.crossover_chance[graph_length+1]) {
            std::vector<int> sorted_indices(graph_length);
            std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
            std::sort(sorted_indices.begin(), sorted_indices.end(), 
                    [&individual](int i1, int i2) { return individual.genes[i1] < individual.genes[i2]; });

            // order of picking first end then start is important: -> different distribution than the inverse; "more mixing of the good ones with good ones"
            int shift_end = distribution(generator) * graph_length;
            int shift_start = distribution(generator) * shift_end;
            int shift_amount = distribution(generator) * (shift_end - shift_start);
            std::vector<float> shifted_values(shift_end - shift_start, 0);
            for (int i = shift_start; i < shift_end; ++i) {
                shifted_values[((i - shift_start) + shift_amount) % (shift_end - shift_start)] = individual.genes[sorted_indices[i]];
            }
            for (int i = shift_start; i < shift_end; ++i) {
                individual.genes[sorted_indices[i]] = shifted_values[i - shift_start];
            }
        }
    }

    void performSexualReproductionPair(int i, int end, std::default_random_engine& generator) {
        Individual* parent1 = tournamentSelection(generator);
        Individual* parent2 = tournamentSelection(generator);
        // Child 1
        crossover(*parent1, *parent2, *new_population[i], generator);
        mutate(*new_population[i], generator);
        // shift(*new_population[i], generator);
        // Child 2, check for bounds since the last chunk might not be full
        if (i + 1 < end) {
            crossover(*parent2, *parent1, *new_population[i + 1], generator);
            mutate(*new_population[i + 1], generator);
            // shift(*new_population[i + 1], generator);
        }
    }

    void performSexualReproduction(int start, int end, std::default_random_engine& generator) {
        for (int i = start; i < end; i += 2) {
            performSexualReproductionPair(i, end, generator);
        }
    }

public:
    EvolutionaryAlgorithm(int pop_size, int genes_length, std::function<double(const Individual&, int*, bool*, int, int*, int, double, double, double, int, int*)> eval_func,
                            int* graph, bool* same_block, int length_bad_edges, int* bad_edges, double factor_0, double factor_not_0, double factor_bad, int legth_initial_component, int* initial_component,
                            int num_thrds = std::thread::hardware_concurrency())
                            : population_size(pop_size), evaluate_function(eval_func), graph(graph), same_block(same_block), length_bad_edges(length_bad_edges), bad_edges(bad_edges), graph_length(genes_length),
                            factor_0(factor_0), factor_not_0(factor_not_0), factor_bad(factor_bad), legth_initial_component(legth_initial_component), initial_component(initial_component),
                            num_threads(num_thrds), distribution(0.0, 1.0), best_individual(genes_length+2) {
        pool.reserve(pop_size * 2);  // Preallocate memory for individuals
        for (int i = 0; i < pop_size * 2; ++i) {
            pool.emplace_back(genes_length+2);
        }
        for (int i = 0; i < pop_size; ++i) {
            population.push_back(&pool[i]);
            new_population.push_back(&pool[i + pop_size]);
        }
        // Tracking of individual genes performance, initialize to 0.0
        genes_performance.reserve(genes_length);
        for (int i = 0; i < genes_length; ++i) {
            genes_performance.push_back(0.0);
        }
        // Initialize generators for each thread
        std::random_device rd;
        auto seed_gen = [&rd]() {
            return rd() ^ (std::chrono::system_clock::now().time_since_epoch().count() +
                           (std::hash<std::thread::id>()(std::this_thread::get_id()) << 1));
        };
        generators.reserve(num_threads);
        for (int i = 0; i < num_threads; ++i) {
            generators.emplace_back(seed_gen());
        }
    }

    Individual run(int generations) {
        int chunk_size = std::ceil(static_cast<double>(population_size) / num_threads);
        
        for (int gen = 0; gen < generations; ++gen) {
            auto [best_fitness, mean_fitness] = evaluate(gen+1);
            std::cout << "\r" // Carriage return to move cursor to the beginning of the line
                << "Generation " << std::setw(5) << gen // setw and setfill to ensure line overwrite
                << " | Best Fitness: " << std::setw(12) << static_cast<long int>(best_fitness)
                << " | Mean Fitness: " << std::setw(12) << static_cast<long int>(mean_fitness)
                << " | Best Individual: " << std::setw(12) << static_cast<long int>(best_individual.fitness)
                << std::flush; // Flush to ensure output is written to the console
            std::vector<std::thread> threads;
            for (int i = 0; i < num_threads; ++i) {
                int start = i * chunk_size;
                int end = std::min(start + chunk_size, population_size);
                threads.emplace_back(&EvolutionaryAlgorithm::performSexualReproduction, this, start, end, std::ref(generators[i]));
            }
            for (auto& thread : threads) {
                thread.join();
            }
            std::swap(population, new_population);
        }
        std::cout << std::endl << "Evolution completed!" << std::endl;
        return best_individual;
    }
};

double evaluate_k_assignment(const Individual& individual, int* graph, bool* same_block, int length_bad_edges, int* bad_edges, int graph_length,  double factor_0, double factor_not_0, double factor_bad, int legth_initial_component, int* initial_component)
{
    auto result = build_graph_from_individual(individual.genes_length, individual.genes, graph_length, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, legth_initial_component, initial_component, false);
    return result.first;
}

std::tuple<double, int*, float*> evolution_solve_k_assignment(int population_size, int generations, int graph_length, int* graph, bool* same_block, int length_bad_edges, int* bad_edges, double factor_0, double factor_not_0, double factor_bad, int legth_initial_component, int* initial_component) {
    EvolutionaryAlgorithm ea(population_size, graph_length, evaluate_k_assignment, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, legth_initial_component, initial_component);
    auto best_individual = ea.run(generations);

    auto res = build_graph_from_individual(best_individual.genes_length, best_individual.genes, graph_length, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, legth_initial_component, initial_component, true);
    std::cout << "Best fitness: " << res.first << std::endl;
    return {res.first, res.second, best_individual.genes};
}

double evaluate_patches(const Individual& individual, int* graph, bool* same_block, int length_bad_edges, int* bad_edges, int graph_length,  double factor_0, double factor_not_0, double factor_bad, int legth_initial_component, int* initial_component)
{
    auto result = build_graph_from_individual_patch(individual.genes_length, individual.genes, graph_length, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, false);
    return result.first;
}

std::tuple<double, int*, float*> evolution_solve_patches(int population_size, int generations, int graph_length, int* graph, bool* same_block, int length_bad_edges, int* bad_edges, double factor_0, double factor_not_0, double factor_bad) {
    int legth_initial_component = 0;
    int* initial_component = nullptr;
    EvolutionaryAlgorithm ea(population_size, graph_length, evaluate_patches, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, legth_initial_component, initial_component);
    auto best_individual = ea.run(generations);

    auto res = build_graph_from_individual_patch(best_individual.genes_length, best_individual.genes, graph_length, graph, same_block, length_bad_edges, bad_edges, factor_0, factor_not_0, factor_bad, true);
    return {res.first, res.second, best_individual.genes};
}

#endif // EVOLUTIONARY_ALGORITHM_H