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

class WeightedUF {
private:
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> size;
    std::unordered_map<int, int> weight; // Weight to the parent

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
        // while (p != root) {
        //     int next = parent[p];
        //     parent[p] = root;
        //     int total_weight_ = total_weight - weight[p];  // Update total weight
        //     weight[p] = total_weight;  // Set the new weight to reflect total path weight
        //     total_weight = total_weight_;  // Update total weight for next iteration
        //     p = next;
        // }

        return root;
    }

    bool merge(int x, int y, int k) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x; // Initialize if not already present
            weight[x] = 0; // Initial weight to self is 0
            size[x] = 1; // Initial size is 1
        }
        if (parent.find(y) == parent.end()) {
            parent[y] = y;
            weight[y] = 0;
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
            size[rootY] += size[rootX];
        } else {
            parent[rootY] = rootX;
            weight[rootY] = weightX - k - weightY;  // Ensure symmetry in weight handling
            size[rootX] += size[rootY];
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
};

// Function to check if the given edge between node1 and node2 is valid based on 'k'
bool check_valid(WeightedUF &uf, int node1, int node2, int k) {
    int connection_weight;
    if (uf.connected(node1, node2, connection_weight)) {
        return connection_weight == k;
    }
    return false;
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
std::pair<double, int*> build_graph_from_individual(int length_individual, float* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, int legth_initial_component, int* initial_component, bool build_valid_edges) {
    // std::cout << " Factor 0: " << factor_0 << " Factor not 0: " << factor_not_0 << std::endl;
    
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    // Add the initial components to the graph
    for (int i = 0; i < legth_initial_component; i++) {
        int node1 = -1;
        int node2 = initial_component[i*2];
        int k = initial_component[i*2 + 1];
        add_node_to_component(uf, node1, node2, k);
    }

    std::vector<int> sorted_indices(length_individual);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    int* valid_edges;
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[length_individual];
    }
    else {
        valid_edges = new int[1];
    }

    for (int i=0; i < length_individual; i++) {
        int index = sorted_indices[i];
        int node1 = graph_raw[4*index];
        int node2 = graph_raw[4*index + 1];
        int k = graph_raw[4*index + 2];
        int certainty = graph_raw[4*index + 3];

        if (certainty <= 0) {
            std::cout << "Invalid certainty value: " << certainty << std::endl;
        }

        double k_factor = (k == 0) ? factor_0 : factor_not_0;
        double score_edge = k_factor * ((double)certainty);

        int connection_weight;
        
        valid_edges_count += score_edge;
        if (!(uf.connected(node1, node2, connection_weight))) {
            // std::cout << "Merging components: " << node1 << " " << node2 << " " << k << std::endl;
            add_node_to_component(uf, node1, node2, k);
            int connection_weight1;
            uf.connected(node1, node2, connection_weight1);
            int connection_weight2;
            uf.connected(node2, node1, connection_weight2);
            if (connection_weight1 != -connection_weight2) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " " << connection_weight2 << std::endl;
            }
            if (connection_weight1 != k) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " k1: " << k << std::endl;
            }
            if (connection_weight2 != -k) {
                std::cout << "Invalid connection weight: " << connection_weight2 << " k2: " << k << std::endl;
            }
            if (build_valid_edges){
                valid_edges[index] = 1;
            }
        } else {
            int connection_weight1;
            uf.connected(node1, node2, connection_weight1);
            int connection_weight2;
            uf.connected(node2, node1, connection_weight2);
            if (connection_weight1 != -connection_weight2) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " " << connection_weight2 << std::endl;
            }
            if (!check_valid(uf, node1, node2, k)) {
                valid_edges_count -= score_edge; // Invalid edge, subtract its score
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
            }    
        }
    }
    // std::cout << "Valid edges count: " << (int)valid_edges_count << std::endl;
    return {valid_edges_count, valid_edges};
}

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual_patch(int length_individual, float* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, bool build_valid_edges) {
    // std::cout << " Factor 0: " << factor_0 << " Factor not 0: " << factor_not_0 << std::endl;
    
    // Initialize the graph components with the maximum node id + 1
    WeightedUF uf;

    std::unordered_map<std::tuple<int, int, int, int>, int, hash_tuple> visited_subvolumes;

    std::vector<int> sorted_indices(length_individual);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), 
              [&individual](int i1, int i2) { return individual[i1] < individual[i2]; });

    double valid_edges_count = 0;
    int* valid_edges;
    // return an array containing 0/1 for each edge if selected or not
    if (build_valid_edges) {
        valid_edges = new int[length_individual];
    }
    else {
        valid_edges = new int[1];
    }

    for (int i=0; i < length_individual; i++) {
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

        double k_factor = (k == 0) ? factor_0 : factor_not_0;
        double score_edge = k_factor * ((double)certainty);

        int connection_weight;
        
        valid_edges_count += score_edge;
        if (!(uf.connected(node1, node2, connection_weight))) {
            // std::cout << "Merging components: " << node1 << " " << node2 << " " << k << std::endl;
            add_node_to_component(uf, node1, node2, k);
            int connection_weight1;
            uf.connected(node1, node2, connection_weight1);
            int connection_weight2;
            uf.connected(node2, node1, connection_weight2);
            if (connection_weight1 != -connection_weight2) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " " << connection_weight2 << std::endl;
            }
            if (connection_weight1 != k) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " k1: " << k << std::endl;
            }
            if (connection_weight2 != -k) {
                std::cout << "Invalid connection weight: " << connection_weight2 << " k2: " << k << std::endl;
            }
            if (build_valid_edges){
                valid_edges[index] = 1;
            }
            // Add the subvolumes to the visited set
            visited_subvolumes[node1_subvolume] = node1;
            visited_subvolumes[node2_subvolume] = node2;
        } else {
            int connection_weight1;
            uf.connected(node1, node2, connection_weight1);
            int connection_weight2;
            uf.connected(node2, node1, connection_weight2);
            if (connection_weight1 != -connection_weight2) {
                std::cout << "Invalid connection weight: " << connection_weight1 << " " << connection_weight2 << std::endl;
            }
            if (!check_valid(uf, node1, node2, k)) {
                valid_edges_count -= score_edge; // Invalid edge, subtract its score
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
    double fitness;
    int genes_length;

    Individual(int size) : fitness(0) {
        genes = new float[size];
        genes_length = size;
        mutation_chance = new float[size];
        crossover_chance = new float[size];
        // Initialize mutation chances and modulo values
        for (int i = 0; i < size; ++i) {
            genes[i] =  (rand() % 100) / 100.0; // random in 0, 1 float
            mutation_chance[i] = 0.001;  // Default mutation chance, can be adjusted
            crossover_chance[i] = 0.1;  // Default crossover chance, can be adjusted
        }
    }
};

class EvolutionaryAlgorithm {
private:
    std::vector<Individual> pool;
    std::vector<Individual*> population;
    std::vector<Individual*> new_population;
    int population_size;
    int genes_lengthgth;
    const double crossover_rate = 0.7;
    const double mutation_rate = 0.01;
    int tournament_size = 10;
    std::function<double(const Individual&, int*, int, double, double, int, int*)> evaluate_function;
    int* graph;
    int graph_length;
    double factor_0;
    double factor_not_0;
    int legth_initial_component;
    int* initial_component;
    int num_threads;

    std::vector<std::default_random_engine> generators;
    std::uniform_real_distribution<float> distribution;

    Individual best_individual;

    std::tuple<double, double> evaluate() {
        int num_threads = std::thread::hardware_concurrency();  // Get the number of threads supported by the hardware
        int chunk_size = std::ceil(population.size() / static_cast<double>(num_threads));
        std::vector<std::thread> threads;
        
        // Lambda to process a slice of the population
        auto process_chunk = [this](int start, int end) {
            for (int i = start; i < end && i < this->population.size(); ++i) {
                double fitness = this->evaluate_function(*this->population[i], this->graph, this->graph_length, this->factor_0, this->factor_not_0, this->legth_initial_component, this->initial_component);
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
        double best_fitness = population[0]->fitness;
        double mean_fitness = 0;
        for (auto& ind : population) {
            mean_fitness += ind->fitness;
            if (ind->fitness > best_fitness) best_fitness = ind->fitness;

            if (ind->fitness > best_individual.fitness) {
                best_individual.fitness = ind->fitness;
                for (int i = 0; i < genes_lengthgth; ++i) {
                    best_individual.genes[i] = ind->genes[i];
                }
            }
        }
        return {best_fitness, mean_fitness / population.size()};
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
        for (int i = 0; i < genes_lengthgth; ++i) {
            float crossover_chance = distribution(generator);
            child.genes[i] = (parent1.crossover_chance[i] < crossover_chance) ? parent1.genes[i] : parent2.genes[i];
        }
    }

    void mutate(Individual& individual, std::default_random_engine& generator) {
        for (int i = 0; i < individual.genes_length; ++i) {
            if (distribution(generator) < individual.mutation_chance[i]) {
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

    void performSexualReproductionPair(int i, int end, std::default_random_engine& generator) {
        Individual* parent1 = tournamentSelection(generator);
        Individual* parent2 = tournamentSelection(generator);
        // Child 1
        crossover(*parent1, *parent2, *new_population[i], generator);
        mutate(*new_population[i], generator);
        // Child 2, check for bounds since the last chunk might not be full
        if (i + 1 < end) {
            crossover(*parent2, *parent1, *new_population[i + 1], generator);
            mutate(*new_population[i + 1], generator);
        }
    }

    void performSexualReproduction(int start, int end, std::default_random_engine& generator) {
        for (int i = start; i < end; i += 2) {
            performSexualReproductionPair(i, end, generator);
        }
    }

public:
    EvolutionaryAlgorithm(int pop_size, int genes_length, std::function<double(const Individual&, int*, int, double, double, int, int*)> eval_func,
                            int* graph, double factor_0, double factor_not_0, int legth_initial_component, int* initial_component,
                            int num_thrds = std::thread::hardware_concurrency())
                            : population_size(pop_size), genes_lengthgth(genes_length), evaluate_function(eval_func), graph(graph), graph_length(genes_length),
                            factor_0(factor_0), factor_not_0(factor_not_0), legth_initial_component(legth_initial_component), initial_component(initial_component),
                            num_threads(num_thrds), distribution(0.0, 1.0), best_individual(genes_length) {
        pool.reserve(pop_size * 2);  // Preallocate memory for individuals
        for (int i = 0; i < pop_size * 2; ++i) {
            pool.emplace_back(genes_length);
        }
        for (int i = 0; i < pop_size; ++i) {
            population.push_back(&pool[i]);
            new_population.push_back(&pool[i + pop_size]);
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
            auto [best_fitness, mean_fitness] = evaluate();
            std::cout << "Generation " << gen << " Best Fitness: " << best_fitness << " Mean Fitness: " << mean_fitness << " Best Individual: " << best_individual.fitness << std::endl;

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
        return best_individual;
    }
};

double evaluate_k_assignment(const Individual& individual, int* graph, int graph_length,  double factor_0, double factor_not_0, int legth_initial_component, int* initial_component)
{
    auto result = build_graph_from_individual(graph_length, individual.genes, graph_length, graph, factor_0, factor_not_0, legth_initial_component, initial_component, false);
    return result.first;
}

std::tuple<double, int*, float*> evolution_solve_k_assignment(int population_size, int generations, int graph_length, int* graph, double factor_0, double factor_not_0, int legth_initial_component, int* initial_component) {
    EvolutionaryAlgorithm ea(population_size, graph_length, evaluate_k_assignment, graph, factor_0, factor_not_0, legth_initial_component, initial_component);
    auto best_individual = ea.run(generations);

    auto res = build_graph_from_individual(graph_length, best_individual.genes, graph_length, graph, factor_0, factor_not_0, legth_initial_component, initial_component, true);
    return {res.first, res.second, best_individual.genes};
}

double evaluate_patches(const Individual& individual, int* graph, int graph_length,  double factor_0, double factor_not_0, int legth_initial_component, int* initial_component)
{
    auto result = build_graph_from_individual_patch(graph_length, individual.genes, graph_length, graph, factor_0, factor_not_0, false);
    return result.first;
}

std::tuple<double, int*, float*> evolution_solve_patches(int population_size, int generations, int graph_length, int* graph, double factor_0, double factor_not_0) {
    int legth_initial_component = 0;
    int* initial_component = nullptr;
    EvolutionaryAlgorithm ea(population_size, graph_length, evaluate_patches, graph, factor_0, factor_not_0, legth_initial_component, initial_component);
    auto best_individual = ea.run(generations);

    auto res = build_graph_from_individual_patch(graph_length, best_individual.genes, graph_length, graph, factor_0, factor_not_0, true);
    return {res.first, res.second, best_individual.genes};
}

#endif // EVOLUTIONARY_ALGORITHM_H