#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <random>
#include <future>
#include <thread>
#include <chrono>
#include <queue>

// Global random number generator - initialized once for performance
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<int> dist_pick;
std::discrete_distribution<int> dist_frontier;
// std::mt19937 gen(31415629);

namespace py = pybind11;

class Config {
public:
    // Define all configuration parameters as public members
    float sampleRatioScore;
    bool display;
    bool printScores;
    float pickedScoresSimilarity;
    float finalScoreMax;
    float finalScoreMin;
    float scoreThreshold;
    bool fitSheet;
    int costThreshold;
    int costPercentile;
    int costPercentileThreshold;
    float costSheetDistanceThreshold;
    float rounddownBestScore;
    float costThresholdPrediction;
    float minPredictionThreshold;
    int nrPointsMin;
    int nrPointsMax;
    int minPatchPoints;
    std::vector<float> windingAngleRange;
    float multipleInstancesPerBatchFactor;
    float epsilon;
    int angleTolerance;
    int maxThreads;
    int minPointsWindingSwitch;
    int minWindingSwitchSheetDistance;
    int maxWindingSwitchSheetDistance;
    float windingSwitchSheetScoreFactor;
    int windingDirection;
    bool enableWindingSwitch;
    int surroundingPatchesSize;
    int maxSheetClipDistance;
    Eigen::Vector2f sheetZRange;
    Eigen::Vector2i sheetKRange;
    float volumeMinCertaintyTotalPercentage;
    float maxUmbilicusDifference;
    int walkAggregationThreshold;
    int walkAggregationMaxCurrent;
    int max_nr_walks;
    int max_unchanged_walks;
    bool continue_walks;
    int max_same_block_jump_range;
    int max_steps;
    int max_tries; 
    int min_steps;
    int min_end_steps;

    // Load configuration from a YAML file
    void load(const std::string& filename) {
        YAML::Node config = YAML::LoadFile(filename);
        sampleRatioScore = config["sample_ratio_score"].as<float>();
        display = config["display"].as<bool>();
        printScores = config["print_scores"].as<bool>();
        pickedScoresSimilarity = config["picked_scores_similarity"].as<float>();
        finalScoreMax = config["final_score_max"].as<float>();
        finalScoreMin = config["final_score_min"].as<float>();
        scoreThreshold = config["score_threshold"].as<float>();
        fitSheet = config["fit_sheet"].as<bool>();
        costThreshold = config["cost_threshold"].as<int>();
        costPercentile = config["cost_percentile"].as<int>();
        costPercentileThreshold = config["cost_percentile_threshold"].as<int>();
        costSheetDistanceThreshold = config["cost_sheet_distance_threshold"].as<float>();
        rounddownBestScore = config["rounddown_best_score"].as<float>();
        costThresholdPrediction = config["cost_threshold_prediction"].as<float>();
        minPredictionThreshold = config["min_prediction_threshold"].as<float>();
        nrPointsMin = config["nr_points_min"].as<float>();
        nrPointsMax = config["nr_points_max"].as<float>();
        minPatchPoints = config["min_patch_points"].as<float>();
        multipleInstancesPerBatchFactor = config["multiple_instances_per_batch_factor"].as<float>();
        epsilon = config["epsilon"].as<float>();
        angleTolerance = config["angle_tolerance"].as<int>();
        maxThreads = config["max_threads"].as<int>();
        minPointsWindingSwitch = config["min_points_winding_switch"].as<int>();
        minWindingSwitchSheetDistance = config["min_winding_switch_sheet_distance"].as<int>();
        maxWindingSwitchSheetDistance = config["max_winding_switch_sheet_distance"].as<int>();
        windingSwitchSheetScoreFactor = config["winding_switch_sheet_score_factor"].as<float>();
        windingDirection = config["winding_direction"].as<float>();
        enableWindingSwitch = config["enable_winding_switch"].as<bool>();
        surroundingPatchesSize = config["surrounding_patches_size"].as<int>();
        maxSheetClipDistance = config["max_sheet_clip_distance"].as<int>();
        sheetZRange = { config["sheet_z_range"][0].as<float>(), config["sheet_z_range"][1].as<float>() };
        sheetKRange = { config["sheet_k_range"][0].as<int>(), config["sheet_k_range"][1].as<int>() };
        volumeMinCertaintyTotalPercentage = config["volume_min_certainty_total_percentage"].as<float>();
        maxUmbilicusDifference = config["max_umbilicus_difference"].as<float>();
        walkAggregationThreshold = config["walk_aggregation_threshold"].as<int>();
        walkAggregationMaxCurrent = config["walk_aggregation_max_current"].as<int>();
        max_nr_walks = config["max_nr_walks"].as<int>();
        max_unchanged_walks = config["max_unchanged_walks"].as<int>();
        continue_walks = config["continue_walks"].as<bool>();
        max_same_block_jump_range = config["max_same_block_jump_range"].as<int>();
        max_steps = config["max_steps"].as<int>();
        max_tries = config["max_tries"].as<int>();
        min_steps = config["min_steps"].as<int>();
        min_end_steps = config["min_end_steps"].as<int>();
    }
    void print() {
        // Print all configuration parameters
        std::cout << "sample_ratio_score: " << sampleRatioScore << std::endl;
        std::cout << "display: " << display << std::endl;
        std::cout << "print_scores: " << printScores << std::endl;
        std::cout << "picked_scores_similarity: " << pickedScoresSimilarity << std::endl;
        std::cout << "final_score_max: " << finalScoreMax << std::endl;
        std::cout << "final_score_min: " << finalScoreMin << std::endl;
        std::cout << "score_threshold: " << scoreThreshold << std::endl;
        std::cout << "fit_sheet: " << fitSheet << std::endl;
        std::cout << "cost_threshold: " << costThreshold << std::endl;
        std::cout << "cost_percentile: " << costPercentile << std::endl;
        std::cout << "cost_percentile_threshold: " << costPercentileThreshold << std::endl;
        std::cout << "cost_sheet_distance_threshold: " << costSheetDistanceThreshold << std::endl;
        std::cout << "rounddown_best_score: " << rounddownBestScore << std::endl;
        std::cout << "cost_threshold_prediction: " << costThresholdPrediction << std::endl;
        std::cout << "min_prediction_threshold: " << minPredictionThreshold << std::endl;
        std::cout << "nr_points_min: " << nrPointsMin << std::endl;
        std::cout << "nr_points_max: " << nrPointsMax << std::endl;
        std::cout << "min_patch_points: " << minPatchPoints << std::endl;
        std::cout << "multiple_instances_per_batch_factor: " << multipleInstancesPerBatchFactor << std::endl;
        std::cout << "epsilon: " << epsilon << std::endl;
        std::cout << "angle_tolerance: " << angleTolerance << std::endl;
        std::cout << "max_threads: " << maxThreads << std::endl;
        std::cout << "min_points_winding_switch: " << minPointsWindingSwitch << std::endl;
        std::cout << "min_winding_switch_sheet_distance: " << minWindingSwitchSheetDistance << std::endl;
        std::cout << "max_winding_switch_sheet_distance: " << maxWindingSwitchSheetDistance << std::endl;
        std::cout << "winding_switch_sheet_score_factor: " << windingSwitchSheetScoreFactor << std::endl;
        std::cout << "winding_direction: " << windingDirection << std::endl;
        std::cout << "enable_winding_switch: " << enableWindingSwitch << std::endl;
        std::cout << "surrounding_patches_size: " << surroundingPatchesSize << std::endl;
        std::cout << "max_sheet_clip_distance: " << maxSheetClipDistance << std::endl;
        std::cout << "sheet_z_range: " << sheetZRange[0] << " " << sheetZRange[1] << std::endl;
        std::cout << "sheet_k_range: " << sheetKRange[0] << " " << sheetKRange[1] << std::endl;
        std::cout << "volume_min_certainty_total_percentage: " << volumeMinCertaintyTotalPercentage << std::endl;
        std::cout << "max_umbilicus_difference: " << maxUmbilicusDifference << std::endl;
        std::cout << "walk_aggregation_threshold: " << walkAggregationThreshold << std::endl;
        std::cout << "walk_aggregation_max_current: " << walkAggregationMaxCurrent << std::endl;
        std::cout << "max_nr_walks: " << max_nr_walks << std::endl;
        std::cout << "max_unchanged_walks: " << max_unchanged_walks << std::endl;
        std::cout << "continue_walks: " << continue_walks << std::endl;
        std::cout << "max_same_block_jump_range: " << max_same_block_jump_range << std::endl;
        std::cout << "max_steps: " << max_steps << std::endl;
        std::cout << "max_tries: " << max_tries << std::endl;
        std::cout << "min_steps: " << min_steps << std::endl;
        std::cout << "min_end_steps: " << min_end_steps << std::endl;
    }
};

struct VolumeID {
    int x, y, z;

    bool operator==(const VolumeID& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct VolumeIDHash {
   int operator()(const VolumeID& volumeID) const {
        // Combine hashes of individual fields
        return std::hash<int>()(volumeID.x) ^ std::hash<int>()(-volumeID.y) ^ std::hash<int>()(100000 * volumeID.z);
    }
};

using PatchID = int;  // Or any appropriate type
using K = int; // Or any appropriate type

struct Node {
    VolumeID volume_id;           // Struct for VolumeID
    PatchID patch_id;             // Type for PatchID
    std::vector<std::shared_ptr<Node>> next_nodes;  // Vector of pointers to next nodes
    std::vector<std::shared_ptr<Node>> same_block_next_nodes;  // Vector of pointers to next nodes
    std::vector<K> k;                     // Array of 6 integers for 'k'
    std::vector<K> same_block_k;                     // Array of 6 integers for 'k'
    float umbilicus_direction[3]; // Array of 3 floats for 'umbilicus_direction'
    float centroid[3];            // Array of 3 floats for 'centroid'
    float distance;               // Single float for 'distance'
    int index = -1;               // Single integer for 'index' in 'nodes' vector
    bool is_landmark = false;    // Single boolean for 'is_landmark'
    int frontier_nr = 0;         // Single integer for 'frontier_nr'. describes how many unreached nodes are in the frontier of the node
};
// Using shared pointers for Node management
using NodePtr = std::shared_ptr<Node>;

using AggregateKey = std::tuple<int, int, int>; // Start Node, End Node, K

struct KeyHash {
    int operator()(const AggregateKey& key) const {
        auto [x, y, z] = key;
        int res = 17;
        res = res * 31 + std::hash<int>()(x);
        res = res * 31 + std::hash<int>()(y);
        res = res * 31 + std::hash<int>()(z);
        return res;
    }
};

// This type represents the main data structure for storing aggregated connections.
using AggregatedConnections = std::unordered_map<AggregateKey, int, KeyHash>;

using VolumeDict = std::unordered_map<VolumeID, std::unordered_map<PatchID, std::pair<NodePtr, K>>, VolumeIDHash>;
inline bool exists(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    auto it = dict.find(volID);
    if (it != dict.end()) {
        return it->second.find(patchID) != it->second.end();
    }
    return false;
}
inline K getKPrime(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    return dict.at(volID).at(patchID).second; // Add error handling as needed
}
inline NodePtr getNode(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    return dict.at(volID).at(patchID).first; // Add error handling as needed
}
inline bool existsForVolume(const VolumeDict& dict, VolumeID volID) {
    return dict.find(volID) != dict.end();
}
inline std::unordered_map<PatchID, std::pair<NodePtr, K>> getAllForVolume(const VolumeDict& dict, VolumeID volID) {
    return dict.at(volID); // Add error handling as needed
}

using NodeUsageCount = std::unordered_map<NodePtr, std::unordered_map<K, int>>;

std::pair<std::unordered_set<NodePtr>, int> frontier_bfs(NodePtr node, int max_depth = 3) {
    // Run bfs on the node
    std::queue<std::pair<NodePtr, int>> q;
    q.push({node, 0});
    // visited set
    std::unordered_set<NodePtr> visited;
    int nr_unassigned = 0;
    while (!q.empty()) {
        // Get the current node and depth
        auto [current_node, depth] = q.front();
        q.pop();
        // If the node is already visited, skip
        if (visited.find(current_node) != visited.end()) {
            continue;
        }
        // Mark the node as visited
        visited.insert(current_node);
        // If the node is unassigned, increment the count
        if (current_node->index == -1) {
            nr_unassigned++;
        }
        int next_depth = depth + 1;
        // If the depth is greater than max_depth, skip
        if (next_depth > max_depth) {
            continue;
        }
        // Add all the next nodes to the queue
        for (const auto& next_node : current_node->next_nodes) {
            if (next_node) {
                q.push({next_node, next_depth});
            }
        }
        /* Seems to result in less potent picks if used. */
        // // Add all the same block next nodes to the queue
        // for (const auto& next_node : current_node->same_block_next_nodes) {
        //     if (next_node) {
        //         q.push({next_node, next_depth});
        //     }
        // }
    } 
    return {visited, nr_unassigned};
}

void decrement_frontiers(NodePtr node, int max_depth = 3) {
    // Run bfs on the node
    auto [visited, nr_unassigned] = frontier_bfs(node, max_depth);
    // Decrement the frontier_nr for all the visited nodes
    for (const auto& visited_node : visited) {
        visited_node->frontier_nr--;
        if (visited_node->frontier_nr < 0) {
            std::cout << "Frontier nr is negative" << std::endl;
        }
    }
}

std::pair<std::vector<NodePtr>, std::vector<NodePtr>> initializeNodes(
    std::vector<std::vector<int>> start_ids,
    std::vector<std::vector<int>> ids, 
    std::vector<std::vector<std::vector<int>>> nextNodes, 
    std::vector<std::vector<int>> kValues, 
    std::vector<std::vector<bool>> same_block,
    std::vector<std::vector<float>> umbilicusDirections,
    std::vector<std::vector<float>>  centroids
    )
{
    // Assuming the first dimension of each vector is 'n'
    int n = ids.size(); 
    std::vector<NodePtr> nodes;
    nodes.reserve(n); // Reserve space for 'n' nodes

    std::cout << "Begin initialization" << std::endl;
    VolumeDict volume_dict;
    // First pass: Initialize nodes
    for (int i = 0; i < n; ++i) {
        auto node = std::make_shared<Node>();
        node->volume_id = {ids[i][0], ids[i][1], ids[i][2]};
        node->patch_id = ids[i][3];
        node->umbilicus_direction[0] = umbilicusDirections[i][0];
        node->umbilicus_direction[1] = umbilicusDirections[i][1];
        node->umbilicus_direction[2] = umbilicusDirections[i][2];
        node->centroid[0] = centroids[i][0];
        node->centroid[1] = centroids[i][1];
        node->centroid[2] = centroids[i][2];
        // Calculate L2 norm for umbilicus_direction
        node->distance = std::sqrt(node->umbilicus_direction[0] * node->umbilicus_direction[0] + node->umbilicus_direction[1] * node->umbilicus_direction[1] + node->umbilicus_direction[2] * node->umbilicus_direction[2]);
        // Index is the index in the 'nodes' vector
        node->index = -1;
        
        // Add node to volume_dict
        volume_dict[node->volume_id][node->patch_id] = std::make_pair(node, -1);

        nodes.push_back(node);
    }
    std::cout << "First pass done" << std::endl;

    // Second pass: Set next_nodes pointers
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < nextNodes[i].size(); ++j) {
            VolumeID nextVolID = {nextNodes[i][j][0], nextNodes[i][j][1], nextNodes[i][j][2]};
            PatchID nextPatchID = nextNodes[i][j][3];

            // Find the node with the corresponding VolumeID and PatchID
            if (exists(std::cref(volume_dict), nextVolID, nextPatchID)) {
                if (same_block[i][j]) {
                    nodes[i]->same_block_next_nodes.push_back(getNode(std::cref(volume_dict), nextVolID, nextPatchID));
                    nodes[i]->same_block_k.push_back(kValues[i][j]);
                } else {
                    nodes[i]->next_nodes.push_back(getNode(std::cref(volume_dict), nextVolID, nextPatchID));
                    nodes[i]->k.push_back(kValues[i][j]);
                }
            } else {
                std::cout << "Node not found for next node" << std::endl;
                nodes[i]->next_nodes.push_back(nullptr);
            }
        }
    }
    std::cout << "Second pass done" << std::endl;

    if (false) {
        // Third pass: Filter out non-reciprocal next_nodes
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < nodes[i]->next_nodes.size(); ++j) {
                NodePtr nextNode = nodes[i]->next_nodes[j];
                if (nextNode) {
                    bool reciprocal = std::any_of(std::begin(nextNode->next_nodes), std::end(nextNode->next_nodes),
                                                [&nodes, i](const NodePtr n) {
                                                    return n != nullptr && n->volume_id == nodes[i]->volume_id && n->patch_id == nodes[i]->patch_id;
                                                });
                    
                    if (!reciprocal) {
                        nodes[i]->next_nodes[j] = nullptr;
                    }
                }
            }
        }
        std::cout << "Third pass done" << std::endl;
    }

    // Calculate the frontier nr for each node
    for (NodePtr node : nodes) {
        auto [visited_nodes, nr_unassigned] = frontier_bfs(node);
        node->frontier_nr = nr_unassigned;
    }
    std::cout << "Fourth pass done" << std::endl;

    std::vector<NodePtr> start_nodes;

    for(const auto& id : start_ids) {
        VolumeID volID = {id[0], id[1], id[2]};
        PatchID patchID = id[3];
        NodePtr start_node = getNode(std::cref(volume_dict), volID, patchID);
        decrement_frontiers(start_node);
        start_nodes.push_back(start_node);
    }

    std::cout << "Initialization done" << std::endl;
    return std::make_pair(nodes, start_nodes);
}

std::pair<py::array_t<int>, py::array_t<int>> process_result(std::vector<NodePtr>& final_nodes, std::vector<K>& final_ks)
{
    // Correctly specify the shape for the 2D numpy array
    py::array_t<int> result_ids(py::array::ShapeContainer({static_cast<long int>(final_nodes.size()), 4}));
    py::array_t<int> result_ks(final_ks.size()); // 1D array for result_ks

    auto r_ids = result_ids.mutable_unchecked<2>(); // Now correctly a 2D array
    auto r_ks = result_ks.mutable_unchecked<1>(); // 1D array for k values
    K min_ks = *std::min_element(final_ks.begin(), final_ks.end());
    K max_ks = *std::max_element(final_ks.begin(), final_ks.end());
    std::cout << "Min k: " << min_ks << " Max k: " << max_ks << std::endl;

    for (int i = 0; i < final_nodes.size(); ++i) {
        NodePtr node = final_nodes[i];
        K k = final_ks[i];

        r_ids(i, 0) = node->volume_id.x;
        r_ids(i, 1) = node->volume_id.y;
        r_ids(i, 2) = node->volume_id.z;
        r_ids(i, 3) = node->patch_id;
        r_ks(i) = k;
    }

    return std::make_pair(result_ids, result_ks);
}

std::tuple<NodePtr, K, int> pick_start_node(
    std::vector<NodePtr> nodes, 
    std::vector<K> ks, 
    std::vector<int> picked_nrs)
{
    assert(!nodes.empty() && "No nodes to pick from.");

    float total = std::accumulate(picked_nrs.begin(), picked_nrs.end(), 0.0f);
    float mean_ = total / picked_nrs.size();

    auto min_it = std::min_element(picked_nrs.begin(), picked_nrs.end());
    float min_ = *min_it;

    float min_mean_abs = mean_ - min_;
    float threshold = min_ + min_mean_abs * 0.25f;

    std::vector<int> valid_indices;
    for (int i = 0; i < picked_nrs.size(); ++i) {
        if (picked_nrs[i] <= threshold) {
            valid_indices.push_back(i);
        }
    }

    assert(!valid_indices.empty() && "No nodes to pick from.");

    std::uniform_int_distribution<int> dist(0, valid_indices.size() - 1);
    int rand_index = valid_indices[dist(gen)];

    NodePtr node = nodes[rand_index];
    K k = ks[rand_index];

    return {node, k, rand_index};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::vector<int>> pick_start_nodes(
    std::vector<NodePtr> nodes, 
    std::vector<K> ks, 
    std::vector<int> picked_nrs,
    int nr_walks
    )
{
    std::vector<NodePtr> start_nodes;
    std::vector<K> start_ks;
    std::vector<int> start_indices;

    
    assert(!nodes.empty() && "No nodes to pick from.");

    float total = std::accumulate(picked_nrs.begin(), picked_nrs.end(), 0.0f);
    float mean_ = total / picked_nrs.size();

    auto min_it = std::min_element(picked_nrs.begin(), picked_nrs.end());
    float min_ = *min_it;

    float min_mean_abs = mean_ - min_;
    float threshold = min_ + min_mean_abs * 0.25f;

    std::vector<int> valid_indices;
    for (int i = 0; i < picked_nrs.size(); ++i) {
        if (picked_nrs[i] <= threshold) {
            valid_indices.push_back(i);
        }
    }

    assert(!valid_indices.empty() && "No nodes to pick from.");

    std::uniform_int_distribution<int> dist(0, valid_indices.size() - 1);
    for (int i = 0; i < nr_walks; ++i) {
        int rand_index = valid_indices[dist(gen)];

        NodePtr node = nodes[rand_index];
        K k = ks[rand_index];

        start_nodes.push_back(node);
        start_ks.push_back(k);
        start_indices.push_back(rand_index);
    }

    return {start_nodes, start_ks, start_indices};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::vector<int>> pick_start_nodes_precomputed(
    const std::vector<NodePtr>& nodes, 
    const std::vector<K>& ks, 
    const std::vector<int>& valid_indices,
    int nr_walks
    )
{
    std::vector<NodePtr> start_nodes;
    std::vector<K> start_ks;
    std::vector<int> start_indices;

    start_nodes.reserve(nr_walks);  // Reserving space to avoid multiple reallocations
    start_ks.reserve(nr_walks);
    start_indices.reserve(nr_walks);

    for (int i = 0; i < nr_walks; ++i) {
        // int rand_index = valid_indices[dist_pick(gen)];
        int rand_index = dist_frontier(gen);

        NodePtr node = nodes[rand_index];
        K k = ks[rand_index];

        if (node->index != rand_index) {
            std::cout << "Bug in pick_start_nodes_precomputed" << std::endl;
        }

        start_nodes.push_back(node);
        start_ks.push_back(k);
        start_indices.push_back(rand_index);
    }

    return {std::move(start_nodes), std::move(start_ks), std::move(start_indices)};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::vector<int>> pick_start_nodes_precomputed_pyramid_down(
    std::uniform_int_distribution<>& distrib,
    const std::vector<NodePtr>& landmark_nodes,
    const std::vector<K>& landmark_ks, 
    const std::vector<NodePtr>& nodes, 
    const std::vector<K>& ks, 
    const std::vector<int>& valid_indices,
    int nr_walks
    )
{
    std::vector<NodePtr> start_nodes;
    std::vector<K> start_ks;
    std::vector<int> start_indices;

    start_nodes.reserve(nr_walks);  // Reserving space to avoid multiple reallocations
    start_ks.reserve(nr_walks);
    start_indices.reserve(nr_walks);

    for (int i = 0; i < nr_walks; ++i) {
        int p = distrib(gen) % 100;
        if (p < 15) {
            int rand_index = distrib(gen) % landmark_nodes.size();
            NodePtr node = landmark_nodes[rand_index];
            K k = landmark_ks[rand_index];

            start_nodes.push_back(node);
            start_ks.push_back(k);
            start_indices.push_back(rand_index);
        }
        else {
            // int rand_index = valid_indices[dist_pick(gen)];
            int rand_index = dist_frontier(gen);

            NodePtr node = nodes[rand_index];
            K k = ks[rand_index];

            if (node->index != rand_index) {
                std::cout << "Bug in pick_start_nodes_precomputed" << std::endl;
            }

            start_nodes.push_back(node);
            start_ks.push_back(k);
            start_indices.push_back(rand_index);
        }
    }

    return {std::move(start_nodes), std::move(start_ks), std::move(start_indices)};
}

void precompute_pick(const std::vector<long>& picked_nrs, std::vector<int>& valid_indices) {
    double mean_ = 0.0;
    double min_ = std::numeric_limits<double>::max();
    int count = 0;
    valid_indices.clear();
    for (int i = 0; i < picked_nrs.size(); ++i) {
        // Update mean and min statistics
        mean_ += (double)picked_nrs[i];
        count++;
        if (picked_nrs[i] < min_) {
            min_ = (double)picked_nrs[i];
        }
    }

    mean_ = mean_ / (double)count;  

    double min_mean_abs = mean_ - min_;
    double threshold = min_ + min_mean_abs * 0.25;
    // std::cout << "Mean: " << mean_ << " Min: " << min_ << " Threshold: " << threshold << std::endl;

    for (int i = 0; i < picked_nrs.size(); ++i) {
        if ((double)picked_nrs[i] <= threshold) {
            valid_indices.push_back(i);
        }
    }
    gen = std::mt19937(std::random_device{}());
    dist_pick = std::uniform_int_distribution<int>(0, valid_indices.size() - 1);
}

void precompute_pick_frontier(const std::vector<NodePtr> nodes) {
    std::vector<int> frontier_nrs;
    frontier_nrs.reserve(nodes.size());
    float frontier_mean = 0.0f;
    int frontier_min = std::numeric_limits<int>::max();
    int frontier_max = std::numeric_limits<int>::min();
    for (const auto& node : nodes) {
        frontier_nrs.push_back(node->frontier_nr);
        frontier_mean += node->frontier_nr;
        if (node->frontier_nr < frontier_min) {
            frontier_min = node->frontier_nr;
        }
        if (node->frontier_nr > frontier_max) {
            frontier_max = node->frontier_nr;
        }
    }

    frontier_mean /= nodes.size();
    float frontier_threshold_min_mean = frontier_min + (frontier_mean - frontier_min) * 0.25f;
    float frontier_threshold_max_mean = frontier_max - (frontier_max - frontier_mean) * 0.25f;

    for (int i = 0; i < nodes.size(); ++i) {
        frontier_nrs[i] -= frontier_mean;
        if (frontier_nrs[i] < 0) {
            frontier_nrs[i] = 0;
        }
    }

    gen = std::mt19937(std::random_device{}());
    dist_frontier = std::discrete_distribution<int>(frontier_nrs.begin(), frontier_nrs.end());
}

inline std::pair<NodePtr, K> pick_next_node(std::mt19937& gen_, std::uniform_int_distribution<>& distrib, const Node& node, int start_k_diff, int max_same_block_jump_range, bool enable_winding_switch = false) {
    // Check if there are no valid next nodes
    if (node.next_nodes.empty()) {
        return {nullptr, -10};
    }

    int other_block_pick = (enable_winding_switch && !node.same_block_next_nodes.empty()) ? distrib(gen_) % 100 : 0;
    if (other_block_pick < 75) {
        // Return the randomly picked valid next node
        int index = distrib(gen_)%node.next_nodes.size();
        NodePtr next_node = node.next_nodes[index];
        K k = node.k[index];
        return {next_node, k};
    }
    else {
        float p_dir = (1.0f * start_k_diff) / max_same_block_jump_range;
        float p_minus = 0.5f + 0.5f * p_dir;
        p_minus = std::min(1.0f, std::max(0.0f, p_minus));
        float p_plus = 1.0f - p_minus;

        // pick direction -1 or 1 with probabilities p_minus and p_plus
        int direction = ((distrib(gen_) % 100) < (p_minus * 100)) ? -1 : 1;
        assert(direction == -1 || direction == 1);

        for (int i = 0; i < node.same_block_next_nodes.size(); ++i) {
            // Check that the sign of k is the same as the direction
            if (node.same_block_k[i] * direction > 0) {
                NodePtr next_node = node.same_block_next_nodes[i];
                K k = node.same_block_k[i];
                return {next_node, k};
            }
        }
        // No valid next node found
        return {nullptr, -10};
    }
}

std::pair<std::vector<VolumeID>, VolumeID> volumes_of_point(
    const Eigen::Vector3f& point, 
    const VolumeID& last_volume_quadrant,
    std::unordered_set<VolumeID, VolumeIDHash>& computed_volumes,
    int volume_size = 50
    )
{
    int size_half = volume_size / 2;
    VolumeID current_volume_quadrant = {
        static_cast<int>(std::floor(point[0] / size_half)) * size_half,
        static_cast<int>(std::floor(point[1] / size_half)) * size_half,
        static_cast<int>(std::floor(point[2] / size_half)) * size_half
    };

    // If the current quadrant is the same as the last one, return no volumes
    if (current_volume_quadrant == last_volume_quadrant) {
        return std::make_pair(std::vector<VolumeID>(), current_volume_quadrant);
    }

    std::vector<VolumeID> volumes;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            for (int k = -1; k <= 1; ++k) {
                VolumeID new_volume = {
                    current_volume_quadrant.x + i * size_half,
                    current_volume_quadrant.y + j * size_half,
                    current_volume_quadrant.z + k * size_half
                };
                if (computed_volumes.insert(new_volume).second) {
                    volumes.push_back(new_volume);
                }
            }
        }
    }

    return std::make_pair(volumes, current_volume_quadrant);
}

bool check_overlapp_node(
    const NodePtr node, 
    const K k, 
    const VolumeDict& volume_dict,
    float max_umbilicus_difference,
    int step_size = 20, 
    int away_dist_check = 500) 
{
    VolumeID last_volume_quadrant = {INT_MIN, INT_MIN, INT_MIN}; // Initialize with an unlikely value
    std::unordered_set<VolumeID, VolumeIDHash> computed_volumes;

    // Continue if the node is already in volume_dict
    if (exists(std::cref(volume_dict), node->volume_id, node->patch_id)) {
        return true;
    }

    Eigen::Vector3f centroid(Eigen::Vector3f::Map(node->centroid));
    Eigen::Vector3f umbilicus_vec_step(Eigen::Vector3f::Map(node->umbilicus_direction) / node->distance * step_size);

    int nr_steps = static_cast<int>(node->distance / static_cast<float>(step_size));

    for (int j = -away_dist_check / step_size; j < nr_steps; ++j) {
        Eigen::Vector3f step_point = centroid + static_cast<float>(j) * umbilicus_vec_step;

        auto [centroid_volumes, current_volume_quadrant] = volumes_of_point(step_point, last_volume_quadrant, computed_volumes);
        last_volume_quadrant = current_volume_quadrant; // Update the last quadrant

        for (const VolumeID& volume : centroid_volumes) {
            auto itVolume = volume_dict.find(volume);
            if (itVolume != volume_dict.end()) {
                for (const auto& [patch, nodeKPair] : itVolume->second) {
                    NodePtr otherNode = nodeKPair.first;
                    K otherK = nodeKPair.second;
                    
                    if (k != otherK) {
                        continue;
                    }

                    if (max_umbilicus_difference > 0.0f) {
                        float dist_ = otherNode->distance;
                        if (std::abs(node->distance - dist_) > max_umbilicus_difference) {
                            return false;
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool check_overlapp_walk(
    const std::vector<NodePtr>& walk, 
    const std::vector<K>& ks, 
    const VolumeDict& volume_dict,
    float max_umbilicus_difference,
    int step_size = 20, 
    int away_dist_check = 500) 
{
    for (int i = 0; i < walk.size(); ++i) {
        NodePtr node = walk[i];
        K k = ks[i];
        if (!check_overlapp_node(node, k, std::cref(volume_dict), max_umbilicus_difference, step_size, away_dist_check)) {
            return false;
        }
    }
    return true;
}

bool check_inverse_walk(
    const std::vector<NodePtr>& walk, 
    const std::vector<K>& ks
)
{
    for (int i = walk.size() - 1; i > 0; --i) {
        NodePtr node = walk[i];
        K k = ks[i];
        NodePtr next_node = walk[i - 1];
        K next_k = ks[i - 1];
        K k_dif = next_k - k;
        // Check if the next node is in the next_nodes or same_block_next_nodes of the current node
        auto it = std::find(node->next_nodes.begin(), node->next_nodes.end(), next_node);
        auto it_same_block = std::find(node->same_block_next_nodes.begin(), node->same_block_next_nodes.end(), next_node);
        if (it == node->next_nodes.end() && it_same_block == node->same_block_next_nodes.end()) {
            return false;
        }
        // Check if the k value is the same
        if (it != node->next_nodes.end()) {
            int index = std::distance(node->next_nodes.begin(), it);
            if (node->k[index] != k_dif) {
                return false;
            }
        } else {
            int index = std::distance(node->same_block_next_nodes.begin(), it_same_block);
            if (node->same_block_k[index] != k_dif) {
                return false;
            }
        }
    }
    return true;
}

// std::tuple<std::vector<NodePtr>, std::vector<K>, std::string, bool, bool> random_walk(
std::tuple<std::vector<NodePtr>, std::vector<K>, bool, bool> random_walk(
    std::mt19937& gen_,
    std::uniform_int_distribution<>& distrib,
    const NodePtr start_node, 
    const K start_k, 
    const VolumeDict& volume_dict,
    const Eigen::Vector2f& sheet_z_range, 
    const Eigen::Vector2i& sheet_k_range,
    const float max_umbilicus_difference,
    const int max_same_block_jump_range,
    const int max_steps = 20, 
    const int max_tries = 6, 
    const int min_steps = 5,
    const bool enable_winding_switch = false)
{
    std::vector<NodePtr> walk = {start_node};
    std::vector<K> ks = {start_k};
    std::vector<NodePtr> empty_walk = {};
    std::vector<K> empty_ks = {};
    K current_k = start_k;
    std::unordered_map<VolumeID, std::vector<K>, VolumeIDHash> ks_dict = {{start_node->volume_id, {start_k}}};
    int steps = 0;
    NodePtr node_ = nullptr;
    K k;
    bool new_node_flag = false;

    while (steps < max_steps) {
        steps++;
        int tries = 0;

        while (true) {
            if (tries >= max_tries) {
                // return {empty_walk, empty_ks, "Exceeded max_tries", false, false};
                return {empty_walk, empty_ks, false, false};
            }
            tries++;

            auto res = pick_next_node(gen_, distrib, *walk.back(), current_k - start_k, max_same_block_jump_range, enable_winding_switch);
            node_ = res.first;
            k = res.second;
            // continue if nullptr
            if (node_ == nullptr || !node_) {
                if (k == -10) {
                    continue;
                }
                // return {empty_walk, empty_ks, "Inverse loop closure failed", false, false};
                return {empty_walk, empty_ks, false, false};
            }
            // if (k < -1 || k > 1) {
            //     std::cout << "Invalid k value: " << k << std::endl;
            //     continue;
            // }

            // std::cout << "Volume id: " << node_->volume_id.x << " " << node_->volume_id.y << " " << node_->volume_id.z << std::endl;
            if (node_->volume_id.y < sheet_z_range[0] || node_->volume_id.y > sheet_z_range[1]) {
                continue;
            }
            if (current_k + k < sheet_k_range[0] || current_k + k > sheet_k_range[1]) {
                continue;
            }

            auto it = std::find(walk.begin(), walk.end(), node_);
            if (it != walk.end() && node_ != start_node) {
                // get same index in ks
                auto it_ks = ks.begin() + std::distance(walk.begin(), it);
                if (current_k + k != *it_ks) {
                    // return {empty_walk, empty_ks, "Small loop closure failed", false, false};
                    return {walk, ks, !new_node_flag, false};
                }
                continue;
            } else if (!ks_dict[node_->volume_id].empty() && std::find(ks_dict[node_->volume_id].begin(), ks_dict[node_->volume_id].end(), current_k + k) != ks_dict[node_->volume_id].end()) {
                if (node_->volume_id == start_node->volume_id && node_->patch_id == start_node->patch_id && current_k + k == start_k) {
                    if (steps < min_steps) {
                        continue;
                    } else {
                        break; // bad path loop closure, be sure to have this path fail
                    }
                }
                // return {empty_walk, empty_ks, "Already visited volume at current k", false, false};
                return {walk, ks, !new_node_flag, false}; // update the picked nrs if there was no new nodes. walk inside existing sheet, no need to sample there often
            } else {
                // Valid next node found
                break;
            }
        }

        // Update current node and k
        current_k += k;

        // Append the node and k to the walk and ks vectors
        walk.push_back(node_);
        ks.push_back(current_k);

        // Update ks_dict
        if (ks_dict.find(node_->volume_id) == ks_dict.end()) {
            ks_dict[node_->volume_id] = std::vector<K>();
        }
        ks_dict[node_->volume_id].push_back(current_k);

        if (!exists(std::cref(volume_dict), node_->volume_id, node_->patch_id)) {
            new_node_flag = true;
        }

        // Check for loop closure
        if (existsForVolume(std::cref(volume_dict), node_->volume_id)) {
            auto patchKMap = getAllForVolume(std::cref(volume_dict), node_->volume_id);
            for (const auto& [key_patch, k_prime_pair] : patchKMap) {
                if (k_prime_pair.second == current_k) {
                    if (node_->patch_id == k_prime_pair.first->patch_id) {
                        if (steps >= min_steps) {
                            if (new_node_flag) {
                                if (check_inverse_walk(std::cref(walk), std::cref(ks)) && check_overlapp_walk(std::cref(walk), std::cref(ks), std::cref(volume_dict), max_umbilicus_difference)) {
                                    // return {walk, ks, "Loop closed successfully", true, true};
                                    return {walk, ks, true, true};
                                }
                                else {
                                    // return {empty_walk, empty_ks, "Loop has bad overlapp", false, false};
                                    return {empty_walk, empty_ks, false, false};
                                }
                            }
                            else {
                                // return {walk, ks, "Loop closed with no new nodes", true, false};
                                return {walk, ks, true, false};
                            }
                        } 
                    } else {
                        // return {empty_walk, empty_ks, "Loop closure failed with different nodes for same volume id and k", false, false};
                        return {walk, ks, !new_node_flag, false};
                    }
                } else {
                    if (node_->patch_id == k_prime_pair.first->patch_id) {
                        // return {empty_walk, empty_ks, "Loop closure failed with already existing node", false, false};
                        return {walk, ks, !new_node_flag, false};
                    }
                }
            }
        }

        // Node is also start node
        if (node_->volume_id == start_node->volume_id && node_->patch_id == start_node->patch_id && current_k == start_k) {
            if (steps >= min_steps) {
                if (new_node_flag) {
                    if (check_inverse_walk(std::cref(walk), std::cref(ks)) && check_overlapp_walk(std::cref(walk), std::cref(ks), std::cref(volume_dict), max_umbilicus_difference)) {
                    // if (check_inverse_walk(walk, ks)) { //  && check_overlapp_walk(walk, ks, volume_dict, max_umbilicus_difference) can be ommited since a "normal" random walk has the start node in the volume dict, but not if it is a pyramid random walk
                        // return {walk, ks, "Loop closed successfully", true, true};
                        return {walk, ks, true, true};
                    }
                    else {
                        // return {empty_walk, empty_ks, "Loop has bad overlapp", false, false};
                        return {empty_walk, empty_ks, false, false};
                    }
                }
                else {
                    // return {walk, ks, "Loop closed with no new nodes", true, false};
                    return {walk, ks, true, false};
                }
            } 
        }
    }

    // return {empty_walk, empty_ks, "Loop not closed in max_steps", false, false};
    return {empty_walk, empty_ks, false, false};
}

std::tuple<int, bool> walk_aggregation_func(
    std::vector<NodePtr>& nodes_final,
    std::vector<K>& ks_final,
    std::vector<long>& picked_nrs,
    const std::vector<NodePtr>& walk,
    const std::vector<K>& ks,
    VolumeDict& volume_dict, // Keeps track of selected nodes and their k values
    NodeUsageCount& node_usage_count, // Tracks usage count of nodes with specific k values
    float max_umbilicus_difference,
    int walk_aggregation_threshold
    )
{
    std::vector<NodePtr> aggregated_nodes;
    std::vector<K> aggregated_ks;

    for (int i = 0; i < walk.size(); ++i) {
        NodePtr node = walk[i];
        K k = ks[i];

        // Increment the usage count of the node with the specific k value
        int& count = node_usage_count[node][k];
        count++;

        // Aggregate node if it meets criteria and hasn't been aggregated before
        if (count >= walk_aggregation_threshold) {
            // Check if the node is already in volume_dict
            bool isAlreadyAggregated = exists(std::cref(volume_dict), node->volume_id, node->patch_id);
            if (isAlreadyAggregated) {
                // Check if k value is the same
                K k_prime = getKPrime(std::cref(volume_dict), node->volume_id, node->patch_id);
                if (k_prime == k) {
                    continue;
                }
                else {
                    return {0, false};
                }
            }        

            if (!check_overlapp_node(node, k, std::cref(volume_dict), max_umbilicus_difference)) {
                return {0, false};
            }

            aggregated_nodes.push_back(node);
            aggregated_ks.push_back(k);
        }
    }

    // Update volume_dict with the newly aggregated nodes
    for (int i = 0; i < aggregated_nodes.size(); ++i) {
        NodePtr node = aggregated_nodes[i];
        K k = aggregated_ks[i];

        // Update volume_dict with the newly aggregated node
        volume_dict[node->volume_id][node->patch_id] = std::make_pair(node, k);
        // Decrement the frontier nr of the node
        decrement_frontiers(node);
    }

    bool success = !aggregated_nodes.empty();

    if (success) {
        // Append aggregated nodes and ks to nodes and ks vectors
        nodes_final.insert(nodes_final.end(), aggregated_nodes.begin(), aggregated_nodes.end());
        ks_final.insert(ks_final.end(), aggregated_ks.begin(), aggregated_ks.end());
        // Append same number of 0 to picked_nrs vector
        picked_nrs.insert(picked_nrs.end(), aggregated_nodes.size(), 0);
        // Update the index of the aggregated nodes
        for (int i = 0; i < aggregated_nodes.size(); ++i) {
            aggregated_nodes[i]->index = nodes_final.size() - aggregated_nodes.size() + i;
        }
    }

    return {aggregated_nodes.size(), success};
}

void walk_aggregate_connections(
    const std::vector<NodePtr>& walk,
    const std::vector<K>& ks,
    AggregatedConnections &aggregated_connections
    )
{
    NodePtr start_node = walk[0];
    K start_k = ks[0];
    int start_index = start_node->index;

    assert(start_node->is_landmark && "First node in walk must be a landmark");

    int nr_landmarks_direction1 = 0;
    int i = 1;
    for (; i < walk.size(); ++i) {
        NodePtr end_node = walk[i];
        K k = ks[i];
        K k_inv = -k;
        int end_index = end_node->index;
        AggregateKey key = {start_index, end_index, k};
        AggregateKey key_inv = {end_index, start_index, k_inv};
        if (!(end_node->is_landmark)) {
            continue;
        }
        assert(end_node->index >= 0 && "End node index must be non-negative");

        // Disregard loopback volume edges
        if (start_node == end_node) {
            continue;
        }

        // Aggregate the connection
        ++aggregated_connections[key];
        ++aggregated_connections[key_inv];

        if (nr_landmarks_direction1++ > 1) {
            break;
        }
    }
    // look from other direction
    int nr_landmarks_direction2 = 0;
    for (int u = walk.size() - 1; u > i; --u) {
        NodePtr end_node = walk[u];
        K k = ks[u];
        K k_inv = -k;
        int end_index = end_node->index;
        AggregateKey key = {start_index, end_index, k};
        AggregateKey key_inv = {end_index, start_index, k_inv};
        if (!(end_node->is_landmark)) {
            continue;
        }
        assert(end_node->index >= 0 && "End node index must be non-negative");

        // Disregard loopback volume edges
        if (start_node == end_node) {
            continue;
        }

        // Aggregate the connection
        ++aggregated_connections[key];
        ++aggregated_connections[key_inv];

        if (nr_landmarks_direction2++ > 1) {
            break;
        }
    }
}

inline void update_picked_nr(const std::vector<NodePtr>& nodes, std::vector<long>& picked_nrs, int index, int value) {
    if (index < 0) {
        return;
    }
    picked_nrs[index] += value;
    if (picked_nrs[index] < 0) {
        picked_nrs[index] = 0;
    }
    if (picked_nrs[index] > 100) {
        picked_nrs[index] = 100;
    }
    assert (nodes[index]->index == index);
}

struct ThreadResult {
    std::vector<std::vector<NodePtr>> walks;
    std::vector<std::vector<K>> ks;
    // std::vector<std::string> messages;
    std::vector<bool> successes;
    std::vector<bool> new_nodes;
};

ThreadResult threadRandomWalk(
    int nrWalks,
    const std::vector<NodePtr> start_nodes,
    const std::vector<K> start_ks,
    const VolumeDict& volume_dict,
    const Eigen::Vector2f& sheet_z_range, 
    const Eigen::Vector2i& sheet_k_range,
    const float max_umbilicus_difference,
    const int max_same_block_jump_range,
    const int max_steps = 20, 
    const int max_tries = 6, 
    const int min_steps = 5,
    const bool enable_winding_switch = false
    )
{
    std::mt19937 gen_ = std::mt19937(std::random_device{}());
    std::vector<std::vector<NodePtr>> walks;
    std::vector<std::vector<K>> ks;
    // std::vector<std::string> messages;
    std::vector<bool> successes;
    std::vector<bool> new_nodes;
    // Generate a random index from 0 to valid_indices
    std::uniform_int_distribution<> distrib(0, 2*3*4*5*6);

    for (int i = 0; i < nrWalks; ++i) {
        // auto [walk, walk_ks, message, success, new_node] = random_walk(gen_, start_nodes[i], start_ks[i], volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps);
        auto [walk, walk_ks, success, new_node] = random_walk(gen_, distrib, start_nodes[i], start_ks[i], std::cref(volume_dict), std::cref(sheet_z_range), std::cref(sheet_k_range), max_umbilicus_difference, max_same_block_jump_range, max_steps, max_tries, min_steps, enable_winding_switch);
        walks.push_back(walk);
        ks.push_back(walk_ks);
        // messages.push_back(message);
        successes.push_back(success);
        new_nodes.push_back(new_node);
    }

    // return {walks, ks, messages, successes, new_nodes};
    return {walks, ks, successes, new_nodes};
}

std::unordered_map<int, std::unordered_map<K, int>> translate_node_usage_count_python(const NodeUsageCount& node_usage_count) {
    std::unordered_map<int, std::unordered_map<K, int>> translated_node_usage_count;
    for (const auto& [node, k_count] : node_usage_count) {
        int node_index = node->index;
        for (const auto& [k, count] : k_count) {
            translated_node_usage_count[node_index][k] = count;
        }
    }
    return translated_node_usage_count;
}

NodeUsageCount translate_node_usage_count_cpp(const std::unordered_map<int, std::unordered_map<K, int>> node_usage_count, const std::vector<NodePtr>& nodes) {
    NodeUsageCount translated_node_usage_count;
    for (const auto& [node_index, k_count] : node_usage_count) {
        // test if node_index is valid
        if (node_index < 0 || node_index >= nodes.size()) {
            std::cout << "Invalid node index: " << node_index << std::endl;
            continue;
        }
        NodePtr node = nodes[node_index];
        for (const auto& [k, count] : k_count) {
            translated_node_usage_count[node][k] = count;
        }
    }
    return translated_node_usage_count;
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::unordered_map<int, std::unordered_map<K, int>>> solve(
    std::vector<NodePtr> all_nodes,
    std::vector<NodePtr> start_nodes,
    std::vector<K> start_ks,
    std::unordered_map<int, std::unordered_map<K, int>> python_node_usage_count,
    Config& config,
    int numThreads = 28,
    bool return_every_hundrethousandth = false,
    int walksPerThread = 10000
    ) 
{
    // Map to count the frequency of each message
    std::unordered_map<std::string, int> message_count;

    // Extracting parameters from config
    const Eigen::Vector2f& sheet_z_range = config.sheetZRange;
    const Eigen::Vector2i& sheet_k_range = config.sheetKRange;
    float max_umbilicus_difference = config.maxUmbilicusDifference;
    int walk_aggregation_threshold_start = config.walkAggregationThreshold;
    int walk_aggregation_threshold = walk_aggregation_threshold_start;
    int walk_aggregation_max_current = config.walkAggregationMaxCurrent;
    int max_nr_walks = config.max_nr_walks;
    int max_unchanged_walks = config.max_unchanged_walks;
    bool continue_walks = config.continue_walks;
    int max_same_block_jump_range = config.max_same_block_jump_range;
    int max_steps = config.max_steps;
    int max_tries = config.max_tries;
    int min_steps = config.min_steps;
    int min_steps_start = min_steps;
    int min_end_steps = config.min_end_steps;
    bool enable_winding_switch = config.enableWindingSwitch;

    VolumeDict volume_dict;
    std::vector<NodePtr> nodes;
    std::vector<K> ks;
    std::vector<long> picked_nrs;
    std::vector<int> valid_indices;

    if (!continue_walks) {
        for (int i = 0; i < start_nodes.size(); ++i) {
            NodePtr start_node = start_nodes[i];
            K start_k = start_ks[i];
            start_node->is_landmark = true;
            start_node->index = i;
            nodes.push_back(start_node);
            ks.push_back(start_k);
            picked_nrs.push_back(0);
            volume_dict[start_node->volume_id][start_node->patch_id] = std::make_pair(start_node, start_k);
        }
        // precompute_pick(std::cref(picked_nrs), valid_indices);
        precompute_pick_frontier(std::cref(nodes));
        // Add start_node to volume_dict
    }

    std::cout << "Here 1" << std::endl;
    long long int nr_unchanged_walks = 0;
    std::cout << "Here 2" << std::endl;
    NodeUsageCount node_usage_count = translate_node_usage_count_cpp(std::cref(python_node_usage_count), std::cref(all_nodes)); // Map to track node usage count with specific k values
    std::cout << "Here 3" << std::endl;
    long long int walk_aggregation_count = 0;
    long long int total_walks = 0;
    std::cout << "Here 4" << std::endl;
    long long int nrWalks = static_cast<long long int>(walksPerThread) * static_cast<long long int>(numThreads);
    std::cout << "Here 5" << std::endl;
    // Run a minimum nr of random walks before adaptively changing the parameters.
    // Ensures to warm up the nr picked and with that the starting node sampling logic
    // Ensures to warmup the aggregation logic
    long long int warmup_nr_walks_ = static_cast<long long int>(2500000) + static_cast<long long int>(walk_aggregation_threshold_start) * static_cast<long long int>(min_steps_start) * static_cast<long long int>(start_nodes.size()) / static_cast<long long int>(5);
    long long int warmup_nr_walks = warmup_nr_walks_;
    // long long int warmup_first = warmup_nr_walks * 100; // actual warmup from a cold aggregation start mid run
    long long int warmup_first = 0; // placeholder for warmup from a cold aggregation start mid run
    std::cout << "Here 6" << std::endl;
    // Yellow print
    std::cout << "\033[1;33m" << "[ThaumatoAnakalyptor]: Starting " << warmup_nr_walks << " warmup random walks. Nr good nodes: " << start_nodes.size() << "\033[0m" << std::endl;

    // numm threads gens
    std::vector<std::mt19937> gen_;
    // making initialization distribution between 0 and 100000000
    std::uniform_int_distribution<> dist(0, 100000000);
    
    for (int i = 0; i < numThreads; ++i) {
        // std::mt19937 gen_t_(std::random_device{}());
        // fixed seed
        std::mt19937 gen_t_(dist(gen));
        gen_.push_back(gen_t_);
    }

    long long int current_nr_walks = 0;
    while (walk_aggregation_count < 100000 || !return_every_hundrethousandth)
    {
        current_nr_walks += nrWalks;
        // std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << nr_unchanged_walks << " random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        // Display message counts at the end of solve function
        if (total_walks++ % 10 == 0)
        {
            std::cout << "\033[1;36m" << "[ThaumatoAnakalyptor]: Random Walk Messages Summary:" << "\033[0m" << std::endl;
            for (const auto& pair : message_count) {
                std::cout << "\033[1;36m" << "  \"" << pair.first << "\": " << pair.second << "\033[0m" << std::endl;
            }
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << total_walks * nrWalks << "th random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        }
        // if (nr_unchanged_walks > max_unchanged_walks && (walk_aggregation_count != 0 || warmup_nr_walks < current_nr_walks)) { //  && (/* More checks*/)
        if (nr_unchanged_walks > max_unchanged_walks && warmup_nr_walks < current_nr_walks && warmup_first < current_nr_walks) { //  && (/* More checks*/)
            // Reset the unchanged walks counter
            nr_unchanged_walks = 0;
            // warmup_nr_walks = warmup_nr_walks_ / 10;
            warmup_nr_walks = warmup_nr_walks_;
            warmup_first = 0;
            current_nr_walks = 0;
            // // set picked_nrs to 0
            // for (int i = 0; i < picked_nrs.size(); ++i) {
            //     picked_nrs[i] = 0;
            // }
            // precompute_pick(std::cref(picked_nrs), valid_indices);
            precompute_pick_frontier(std::cref(nodes));

            // implement min_steps size logic
            if (min_steps > min_end_steps) {
                min_steps = min_steps / 2;
                // Info cout in blue color
                std::cout << "\033[1;34m" << "[ThaumatoAnakalyptor]: Max unchanged walks reached. Adjusting min_steps to " << min_steps << "\033[0m" << std::endl;
            }
            else if (walk_aggregation_threshold > 1) {
                min_steps = min_steps_start;
                walk_aggregation_threshold = walk_aggregation_threshold / 2;
                std::cout << "\033[1;34m" << "[ThaumatoAnakalyptor]: Max unchanged walks reached. Adjusting walk_aggregation_threshold to " << walk_aggregation_threshold << "\033[0m" << std::endl;
            }
            else {
                std::cout << "\033[1;34m" << "[ThaumatoAnakalyptor]: Max unchanged walks reached. Finishing the random walks." << "\033[0m" << std::endl;
                break;
            }
        }

        // pick nrWalks start nodes
        std::vector<NodePtr> sns;
        std::vector<K> sks;
        std::vector<int> indices_s;
        int nrWalks = walksPerThread * numThreads;
        std::tie(sns, sks, indices_s) = pick_start_nodes_precomputed(std::cref(nodes), std::cref(ks), std::cref(valid_indices), nrWalks);
        // for (int i = 0; i < indices_s.size(); ++i) {
        //     update_picked_nr(std::cref(nodes), picked_nrs, indices_s[i], 1);
        // }


        std::vector<std::future<ThreadResult>> futures(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            auto start_nodes = std::vector<NodePtr>(sns.begin() + i * walksPerThread, sns.begin() + (i + 1) * walksPerThread);
            auto start_ks = std::vector<K>(sks.begin() + i * walksPerThread, sks.begin() + (i + 1) * walksPerThread);
            futures[i] = std::async(std::launch::async, threadRandomWalk, walksPerThread, start_nodes, start_ks, std::cref(volume_dict), std::cref(sheet_z_range), std::cref(sheet_k_range), max_umbilicus_difference, max_same_block_jump_range, max_steps, max_tries, min_steps, enable_winding_switch);
        }

        std::vector<std::vector<NodePtr>> walks_futures;
        std::vector<std::vector<K>> walk_ks_futures;
        std::vector<bool> successes_futures;
        std::vector<bool> new_nodes_futures;
        // Use a loop to wait and process futures as they become ready
        while (!futures.empty()) {
            for (auto it = futures.begin(); it != futures.end(); ) {
                if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    // Safe to get the future since it's ready and being accessed first time
                    auto [walks_, walk_ks_, successes_, new_nodes_] = it->get();

                    for (int j = 0; j < walks_.size(); ++j) {
                        walks_futures.push_back(std::move(walks_[j]));
                        walk_ks_futures.push_back(std::move(walk_ks_[j]));
                        successes_futures.push_back(successes_[j]);
                        new_nodes_futures.push_back(new_nodes_[j]);
                    }

                    // Erase the future from the vector after processing
                    it = futures.erase(it);
                } else {
                    ++it;  // Only increment if not erased
                }
            }

            // Sleep briefly to reduce CPU load if no futures were ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // // single threaded call to threadRandomWalk
        // auto [walks_futures, walk_ks_futures, messages_futures, successes_futures, new_nodes_futures] = threadRandomWalk(gen_[0], nrWalks, sns, sks, volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps);

        // for (int i = 0; i < messages_futures.size(); ++i) {
        //     message_count[messages_futures[i]]++;
        // }
        // Increment the count for the returned message
        // message_count[message]++;

        // Loop over all the walks by iterating trough them
        bool total_aggregation_success = false;
        for (int i = 0; i < walks_futures.size(); ++i) {
            auto& walk = walks_futures[i];
            auto& walk_ks = walk_ks_futures[i];
            bool success = successes_futures[i];
            bool new_node = new_nodes_futures[i];

            // Post-processing after each walk
            if (!success) {
                nr_unchanged_walks++;
                // Handle unsuccessful walk
                continue;
            }

            // // update the picked_nr of each node in the walk -/+5 depending on the success
            // for (NodePtr node : walk) {
            //     update_picked_nr(std::cref(nodes), picked_nrs, node->index, new_node ? -10 : 1);
            // }

            if (!new_node) {
                nr_unchanged_walks++;
                continue;
            }

            auto [walk_aggregated_size, success_aggregated] = walk_aggregation_func(
                nodes, ks, picked_nrs, walk, walk_ks, volume_dict, node_usage_count, max_umbilicus_difference, walk_aggregation_threshold);

            if (!success_aggregated) {
                nr_unchanged_walks++;
                continue;
            }
            else {
                total_aggregation_success = true;
            }

            nr_unchanged_walks = 0;
            walk_aggregation_count += walk_aggregated_size;

            
            // yellow color
            // std::cout << "\033[1;33m" << "[ThaumatoAnakalyptor]: Added " << walk_aggregation_count << " sheet patches." << "\033[0m" << std::endl;
        }
        // Update valid_indices from picked_nrs
        if (total_aggregation_success) {
            // precompute_pick(std::cref(picked_nrs), valid_indices);
            precompute_pick_frontier(std::cref(nodes));
        }
    }
    python_node_usage_count = translate_node_usage_count_python(std::cref(node_usage_count));

    return {nodes, ks, python_node_usage_count};
}

AggregatedConnections solveUp(
    std::vector<NodePtr> landmark_nodes,
    Config& config,
    int numThreads = 28,
    int walksPerThread = 1000
    ) 
{
    // Map to count the frequency of each message
    std::unordered_map<std::string, int> message_count;
    AggregatedConnections aggregated_connections;

    // Extracting parameters from config
    const Eigen::Vector2f& sheet_z_range = config.sheetZRange;
    const Eigen::Vector2i& sheet_k_range = config.sheetKRange;
    float max_umbilicus_difference = config.maxUmbilicusDifference;
    int walk_aggregation_threshold = config.walkAggregationThreshold;
    int walk_aggregation_max_current = config.walkAggregationMaxCurrent;
    int max_nr_walks = config.max_nr_walks;
    int max_unchanged_walks = config.max_unchanged_walks;
    bool continue_walks = config.continue_walks;
    int max_same_block_jump_range = config.max_same_block_jump_range;
    int max_steps = config.max_steps;
    int max_tries = config.max_tries;
    int min_steps = config.min_steps;
    int min_steps_start = min_steps;
    int min_end_steps = config.min_end_steps;
    bool enable_winding_switch = config.enableWindingSwitch;

    VolumeDict volume_dict;
    std::vector<NodePtr> nodes;
    std::vector<K> ks;
    std::vector<long> picked_nrs;
    std::vector<int> valid_indices;

    for (int i = 0; i < landmark_nodes.size(); ++i) {
        NodePtr landmark_node = landmark_nodes[i];
        landmark_node->is_landmark = true;
        landmark_node->index = i;
        nodes.push_back(landmark_node);
        ks.push_back(0);
        picked_nrs.push_back(0);
    }
    precompute_pick(std::cref(picked_nrs), valid_indices);

    size_t nr_unchanged_walks = 0;
    NodeUsageCount node_usage_count; // Map to track node usage count with specific k values
    size_t walk_aggregation_count = 0;
    size_t total_walks = 0;
    size_t nrWalks = walksPerThread * numThreads;

    // numm threads gens
    std::vector<std::mt19937> gen_;
    // making initialization distribution between 0 and 100000000
    std::uniform_int_distribution<> dist(0, 100000000);
    
    for (int i = 0; i < numThreads; ++i) {
        // std::mt19937 gen_t_(std::random_device{}());
        // fixed seed
        std::mt19937 gen_t_(dist(gen));
        gen_.push_back(gen_t_);
    }

    double duration1 = 0;
    double duration2 = 0;
    double duration3_0 = 0;
    double duration3_1 = 0;
    double duration4 = 0;
    int count_durations = 0;

    while (total_walks * nrWalks < max_nr_walks * landmark_nodes.size())
    {
        auto start = std::chrono::high_resolution_clock::now();
        // std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << nr_unchanged_walks << " random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        // Display message counts at the end of solve function
        if (total_walks++ % 100 == 0)
        {
            std::cout << "\033[1;36m" << "[ThaumatoAnakalyptor]: Random Walk Messages Summary:" << "\033[0m" << std::endl;
            for (const auto& pair : message_count) {
                std::cout << "\033[1;36m" << "  \"" << pair.first << "\": " << pair.second << "\033[0m" << std::endl;
            }
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << total_walks * nrWalks << "th random walk of " << max_nr_walks * landmark_nodes.size() << " . Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 1: " << duration1 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 2: " << duration2 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 3_0: " << duration3_0 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 3_1: " << duration3_1 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 4: " << duration4 / count_durations << " ms" << "\033[0m" << std::endl;
            duration1 = 0;
            duration2 = 0;
            duration3_0 = 0;
            duration3_1 = 0;
            duration4 = 0;
            count_durations = 0;
        }

        auto end1 = std::chrono::high_resolution_clock::now();

        // pick nrWalks start nodes
        std::vector<NodePtr> sns;
        std::vector<K> sks;
        std::vector<int> indices_s;
        std::tie(sns, sks, indices_s) = pick_start_nodes_precomputed(std::cref(nodes), std::cref(ks), std::cref(valid_indices), nrWalks);

        auto end2 = std::chrono::high_resolution_clock::now();

        std::vector<std::future<ThreadResult>> futures(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            auto start_nodes = std::vector<NodePtr>(sns.begin() + i * walksPerThread, sns.begin() + (i + 1) * walksPerThread);
            auto start_ks = std::vector<K>(sks.begin() + i * walksPerThread, sks.begin() + (i + 1) * walksPerThread);
            futures[i] = std::async(std::launch::async, threadRandomWalk, walksPerThread, start_nodes, start_ks, std::cref(volume_dict), std::cref(sheet_z_range), std::cref(sheet_k_range), max_umbilicus_difference, max_same_block_jump_range, max_steps, max_tries, min_steps, enable_winding_switch);
        }

        auto end2_5 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<NodePtr>> walks_futures;
        std::vector<std::vector<K>> walk_ks_futures;
        std::vector<bool> successes_futures;
        std::vector<bool> new_nodes_futures;
        // Use a loop to wait and process futures as they become ready
        while (!futures.empty()) {
            for (auto it = futures.begin(); it != futures.end(); ) {
                if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    // Safe to get the future since it's ready and being accessed first time
                    auto [walks_, walk_ks_, successes_, new_nodes_] = it->get();

                    for (int j = 0; j < walks_.size(); ++j) {
                        walks_futures.push_back(std::move(walks_[j]));
                        walk_ks_futures.push_back(std::move(walk_ks_[j]));
                        successes_futures.push_back(successes_[j]);
                        new_nodes_futures.push_back(new_nodes_[j]);
                    }

                    // Erase the future from the vector after processing
                    it = futures.erase(it);
                } else {
                    ++it;  // Only increment if not erased
                }
            }

            // Sleep briefly to reduce CPU load if no futures were ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto end3 = std::chrono::high_resolution_clock::now();

        // Loop over all the walks by iterating trough them
        for (int i = 0; i < walks_futures.size(); ++i) {
            auto& walk = walks_futures[i];
            auto& walk_ks = walk_ks_futures[i];
            bool success = successes_futures[i];
            bool new_node = new_nodes_futures[i];

            // Post-processing after each walk
            if (!success) {
                nr_unchanged_walks++;
                // Handle unsuccessful walk
                continue;
            }

            if (!new_node) {
                nr_unchanged_walks++;
                continue;
            }

            walk_aggregate_connections(walk, walk_ks, aggregated_connections);

            nr_unchanged_walks = 0;
        }

        auto end4 = std::chrono::high_resolution_clock::now();
        count_durations++;
        duration1 += std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count();
        duration2 += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count();
        duration3_0 += std::chrono::duration_cast<std::chrono::milliseconds>(end2_5 - end2).count();
        duration3_1 += std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2_5).count();
        duration4 += std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end3).count();
    }
    return aggregated_connections;
}

std::tuple<std::vector<NodePtr>, std::vector<K>> solveDown(
    int graph_n,
    std::vector<NodePtr> start_nodes,
    std::vector<K> start_ks,
    Config& config,
    int nr_walks_per_node,
    int numThreads = 28,
    int walksPerThread = 1000
    ) 
{
    std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting solveDown with " << graph_n << " nodes. And walks per node: " << nr_walks_per_node << "\033[0m" << std::endl;

    // TODO: adjust for multiple start nodes, only works for one so far:
    NodePtr start_node = start_nodes[0];

    // Map to count the frequency of each message
    std::unordered_map<std::string, int> message_count;

    // Extracting parameters from config
    const Eigen::Vector2f& sheet_z_range = config.sheetZRange;
    const Eigen::Vector2i& sheet_k_range = config.sheetKRange;
    float max_umbilicus_difference = config.maxUmbilicusDifference;
    int walk_aggregation_threshold = config.walkAggregationThreshold;
    int walk_aggregation_max_current = config.walkAggregationMaxCurrent;
    int max_nr_walks = config.max_nr_walks;
    int max_unchanged_walks = config.max_unchanged_walks;
    bool continue_walks = config.continue_walks;
    int max_same_block_jump_range = config.max_same_block_jump_range;
    int max_steps = config.max_steps;
    int max_tries = config.max_tries;
    int min_steps = config.min_steps;
    int min_steps_start = min_steps;
    int min_end_steps = config.min_end_steps;
    bool enable_winding_switch = config.enableWindingSwitch;

    VolumeDict volume_dict;
    std::vector<NodePtr> nodes;
    std::vector<K> ks;
    std::vector<long> picked_nrs;
    std::vector<int> valid_indices;

    if (!continue_walks) {
        for (int i = 0; i < start_nodes.size(); ++i) {
            NodePtr start_node = start_nodes[i];
            K start_k = start_ks[i];
            start_node->is_landmark = true;
            start_node->index = i;
            nodes.push_back(start_node);
            ks.push_back(start_k);
            picked_nrs.push_back(0);
            volume_dict[start_node->volume_id][start_node->patch_id] = std::make_pair(start_node, start_k);
        }
        precompute_pick(std::cref(picked_nrs), valid_indices);
        // Add start_node to volume_dict
    }

    size_t nr_unchanged_walks = 0;
    NodeUsageCount node_usage_count; // Map to track node usage count with specific k values
    size_t walk_aggregation_count = 0;
    size_t total_walks = 0;
    size_t nrWalks = walksPerThread * numThreads;
    size_t nr_node_walks = nr_walks_per_node * graph_n;

    // numm threads gens
    std::vector<std::mt19937> gen_;
    // making initialization distribution between 0 and 100000000
    std::uniform_int_distribution<> dist(0, 100000000);
    
    for (int i = 0; i < numThreads; ++i) {
        // std::mt19937 gen_t_(std::random_device{}());
        // fixed seed
        std::mt19937 gen_t_(dist(gen));
        gen_.push_back(gen_t_);
    }

    double duration1 = 0;
    double duration2 = 0;
    double duration3_0 = 0;
    double duration3_1 = 0;
    double duration4 = 0;
    int count_durations = 0;

    while (((total_walks < 1000) || (total_walks * nrWalks < nr_node_walks)) && (max_nr_walks > 0))
    {
        auto start = std::chrono::high_resolution_clock::now();
        // std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << nr_unchanged_walks << " random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        // Display message counts at the end of solve function
        if (total_walks++ % 100 == 0)
        {
            std::cout << "\033[1;36m" << "[ThaumatoAnakalyptor]: Random Walk Messages Summary:" << "\033[0m" << std::endl;
            for (const auto& pair : message_count) {
                std::cout << "\033[1;36m" << "  \"" << pair.first << "\": " << pair.second << "\033[0m" << std::endl;
            }
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << total_walks * nrWalks << "th random walk of " << nr_node_walks << " . Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 1: " << duration1 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 2: " << duration2 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 3_0: " << duration3_0 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 3_1: " << duration3_1 / count_durations << " ms" << "\033[0m" << std::endl;
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Duration 4: " << duration4 / count_durations << " ms" << "\033[0m" << std::endl;
            duration1 = 0;
            duration2 = 0;
            duration3_0 = 0;
            duration3_1 = 0;
            duration4 = 0;
            count_durations = 0;
        }
        if (nr_unchanged_walks > max_unchanged_walks && walk_aggregation_count != 0) { //  && (/* More checks*/)
            nr_unchanged_walks = 0;
            // set picked_nrs to 0
            for (int i = 0; i < picked_nrs.size(); ++i) {
                picked_nrs[i] = 0;
            }
            precompute_pick(std::cref(picked_nrs), valid_indices);
        }

        auto end1 = std::chrono::high_resolution_clock::now();

        // pick nrWalks start nodes
        std::vector<NodePtr> sns;
        std::vector<K> sks;
        std::vector<int> indices_s;
        int nrWalks = walksPerThread * numThreads;
        std::tie(sns, sks, indices_s) = pick_start_nodes_precomputed_pyramid_down(dist, std::cref(start_nodes), std::cref(start_ks), std::cref(nodes), std::cref(ks), std::cref(valid_indices), nrWalks);
        for (int i = 0; i < indices_s.size(); ++i) {
            update_picked_nr(std::cref(nodes), picked_nrs, indices_s[i], 1);
        }

        auto end2 = std::chrono::high_resolution_clock::now();

        std::vector<std::future<ThreadResult>> futures(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            auto start_nodes = std::vector<NodePtr>(sns.begin() + i * walksPerThread, sns.begin() + (i + 1) * walksPerThread);
            auto start_ks = std::vector<K>(sks.begin() + i * walksPerThread, sks.begin() + (i + 1) * walksPerThread);
            futures[i] = std::async(std::launch::async, threadRandomWalk, walksPerThread, start_nodes, start_ks, std::cref(volume_dict), std::cref(sheet_z_range), std::cref(sheet_k_range), max_umbilicus_difference, max_same_block_jump_range, max_steps, max_tries, min_steps, enable_winding_switch);
        }

        auto end2_5 = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<NodePtr>> walks_futures;
        std::vector<std::vector<K>> walk_ks_futures;
        std::vector<bool> successes_futures;
        std::vector<bool> new_nodes_futures;
        // Use a loop to wait and process futures as they become ready
        while (!futures.empty()) {
            for (auto it = futures.begin(); it != futures.end(); ) {
                if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                    // Safe to get the future since it's ready and being accessed first time
                    auto [walks_, walk_ks_, successes_, new_nodes_] = it->get();

                    for (int j = 0; j < walks_.size(); ++j) {
                        walks_futures.push_back(std::move(walks_[j]));
                        walk_ks_futures.push_back(std::move(walk_ks_[j]));
                        successes_futures.push_back(successes_[j]);
                        new_nodes_futures.push_back(new_nodes_[j]);
                    }

                    // Erase the future from the vector after processing
                    it = futures.erase(it);
                } else {
                    ++it;  // Only increment if not erased
                }
            }

            // Sleep briefly to reduce CPU load if no futures were ready
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        auto end3 = std::chrono::high_resolution_clock::now();

        // Loop over all the walks by iterating trough them
        bool total_aggregation_success = false;
        for (int i = 0; i < walks_futures.size(); ++i) {
            auto& walk = walks_futures[i];
            auto& walk_ks = walk_ks_futures[i];
            bool success = successes_futures[i];
            bool new_node = new_nodes_futures[i];

            // Post-processing after each walk
            if (!success) {
                nr_unchanged_walks++;
                // Handle unsuccessful walk
                continue;
            }

            if (!new_node) {
                nr_unchanged_walks++;
                continue;
            }

            // update the picked_nr of each node in the walk -/+5 depending on the success
            for (NodePtr node : walk) {
                update_picked_nr(std::cref(nodes), picked_nrs, node->index, new_node ? -5 : 0);
            }

            auto [walk_aggregated_size, success_aggregated] = walk_aggregation_func(
                nodes, ks, picked_nrs, walk, walk_ks, volume_dict, node_usage_count, max_umbilicus_difference, walk_aggregation_threshold);

            if (!success_aggregated) {
                nr_unchanged_walks++;
                continue;
            }
            else {
                total_aggregation_success = true;
            }

            nr_unchanged_walks = 0;
            walk_aggregation_count += walk_aggregated_size;
        }
        // Update valid_indices from picked_nrs
        if (total_aggregation_success) {
            precompute_pick(std::cref(picked_nrs), valid_indices);
        }
        
        auto end4 = std::chrono::high_resolution_clock::now();
        count_durations++;
        duration1 += std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start).count();
        duration2 += std::chrono::duration_cast<std::chrono::milliseconds>(end2 - end1).count();
        duration3_0 += std::chrono::duration_cast<std::chrono::milliseconds>(end2_5 - end2).count();
        duration3_1 += std::chrono::duration_cast<std::chrono::milliseconds>(end3 - end2_5).count();
        duration4 += std::chrono::duration_cast<std::chrono::milliseconds>(end4 - end3).count();
    }
    return {nodes, ks};
}

std::tuple<py::array_t<int>, py::array_t<int>, std::unordered_map<int, std::unordered_map<K, int>>> solveRandomWalk(
    std::vector<std::vector<int>> start_ids,
    std::vector<int> start_ks,
    std::unordered_map<int, std::unordered_map<K, int>> initial_node_usage_count,
    const std::string &overlappThresholdFile,
    std::vector<std::vector<int>> ids, 
    std::vector<std::vector<std::vector<int>>> nextNodes, 
    std::vector<std::vector<int>> kValues, 
    std::vector<std::vector<bool>> same_block,
    std::vector<std::vector<float>> umbilicusDirections,
    std::vector<std::vector<float>>  centroids,
    bool return_every_hundrethousandth
    ) 
{
    std::cout << "Begin solveRandomWalk" << std::endl;

    Config config;
    config.load(overlappThresholdFile);
    // config.print();

    std::cout << "Config loaded" << std::endl;

    auto init_res = initializeNodes(start_ids, ids, nextNodes, kValues, same_block, umbilicusDirections, centroids);
    std::vector<NodePtr> nodes = init_res.first;
    std::vector<NodePtr> start_nodes = init_res.second;

    std::cout << "Nodes initialized" << std::endl;

    const int numThreads = std::max((int)(1), (int)((std::thread::hardware_concurrency() * 4) / 5));
    auto [final_nodes, final_ks, python_node_usage_count] = solve(nodes, start_nodes, start_ks, initial_node_usage_count, config, numThreads, return_every_hundrethousandth);

    std::cout << "Solve done" << std::endl;

    // Convert final_nodes and final_ks to a format suitable for Python
    auto [fns, fks] = process_result(final_nodes, final_ks);
    return {fns, fks, python_node_usage_count};
}

AggregatedConnections solvePyramidRandomWalkUp(
    std::vector<std::vector<int>> landmark_ids,
    const std::string &overlappThresholdFile,
    std::vector<std::vector<int>> ids, 
    std::vector<std::vector<std::vector<int>>> nextNodes, 
    std::vector<std::vector<int>> kValues, 
    std::vector<std::vector<bool>> same_block,
    std::vector<std::vector<float>> umbilicusDirections,
    std::vector<std::vector<float>>  centroids,
    int max_nr_walks,
    int max_steps,
    int max_tries,
    int min_steps
    ) 
{
    std::cout << "Begin solvePyramidRandomWalkUp" << std::endl;

    Config config;
    config.load(overlappThresholdFile);

    // adjust config parameters
    config.max_nr_walks = max_nr_walks;
    config.max_steps = max_steps;
    config.max_tries = max_tries;
    config.min_steps = min_steps;

    // config.print();

    std::cout << "Config loaded" << std::endl;

    auto init_res = initializeNodes(landmark_ids, ids, nextNodes, kValues, same_block, umbilicusDirections, centroids);
    std::vector<NodePtr> nodes = init_res.first;
    std::vector<NodePtr> start_nodes = init_res.second;

    std::cout << "Nodes initialized" << std::endl;

    const int numThreads = std::max((int)(1), (int)((std::thread::hardware_concurrency() * 4) / 5));
    AggregatedConnections agg_con = solveUp(start_nodes, config, numThreads);

    std::cout << "Solve done" << std::endl;

    return agg_con;
}

std::pair<py::array_t<int>, py::array_t<int>> solvePyramidRandomWalkDown(
    std::vector<std::vector<int>> landmark_ids,
    std::vector<int> landmark_ks,
    const std::string &overlappThresholdFile,
    std::vector<std::vector<int>> ids, 
    std::vector<std::vector<std::vector<int>>> nextNodes, 
    std::vector<std::vector<int>> kValues, 
    std::vector<std::vector<bool>> same_block,
    std::vector<std::vector<float>> umbilicusDirections,
    std::vector<std::vector<float>>  centroids,
    int max_nr_walks,
    int nr_walks_per_node,
    int max_unchanged_walks,
    int max_steps,
    int max_tries,
    int min_steps,
    int min_end_steps
    ) 
{
    std::cout << "Begin solvePyramidRandomWalkDown" << std::endl;

    Config config;
    config.load(overlappThresholdFile);

    // adjust config parameters
    config.max_nr_walks = max_nr_walks;
    config.max_unchanged_walks = max_unchanged_walks;
    config.max_steps = max_steps;
    config.max_tries = max_tries;
    config.min_steps = min_steps;

    // config.print();

    std::cout << "Config loaded" << std::endl;

    auto init_res = initializeNodes(landmark_ids, ids, nextNodes, kValues, same_block, umbilicusDirections, centroids);
    std::vector<NodePtr> nodes = init_res.first;
    std::vector<NodePtr> landmark_nodes = init_res.second;

    std::cout << "Nodes initialized" << std::endl;

    const int numThreads = std::max((int)(1), (int)((std::thread::hardware_concurrency() * 4) / 5));
    auto [final_nodes, final_ks] = solveDown(nodes.size(), landmark_nodes, landmark_ks, config, nr_walks_per_node, numThreads);

    std::cout << "Solve done" << std::endl;

    // Convert final_nodes and final_ks to a format suitable for Python
    return process_result(final_nodes, final_ks);
}

int count_ks(uint64_t k) {
    // counts number of 1s in the binary representation of k
    int count = 0;
    while (k) {
        if (k & 1) {
            count++;
        }
        k >>= 1;
    }
    return count;
}

std::vector<int> ks_in_k(uint64_t k) {
    std::vector<int> ks;
    int position = 0;
    while (k) {
        if (k & 1) {
            ks.push_back(position-32);
        }
        k >>= 1;
        position++;
    }
    return ks;
}

int k_in_k(uint64_t k) {
    if (k == 0) {
        // empty k
        return -10000;
    }    
    int position = 0;
    while (k) {
        if (k & 1) {
            return position - 32;
        }
        k >>= 1;
        position++;
    }
    return -10000;
}

int64_t* DP_to_k(uint64_t* DP, int nodes_length) {
    int64_t* k_values = new int64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            k_values[i*nodes_length + j] = k_in_k(DP[i*nodes_length + j]);
        }
    }
    return k_values;
}

void processChunk(int start, int end, int nodes_length, uint64_t* initialDP, uint64_t* initialDP_copy) {
    for (int this_node = start; this_node < end; this_node++) {
        for (int adjacent_node = 0; adjacent_node < nodes_length; adjacent_node++) {
            if (initialDP[this_node*nodes_length + adjacent_node] == 0) {
                continue;
            }
            auto this_ks = ks_in_k(initialDP[this_node*nodes_length + adjacent_node]);
            for (int this_k : this_ks) {
                // For each adjacent node, find its adjacent nodes and good k values
                for (int adjacent_next_node = 0; adjacent_next_node < nodes_length; adjacent_next_node++) {
                    if (initialDP[adjacent_node*nodes_length + adjacent_next_node] == 0) {
                        continue;
                    }
                    std::vector<int> other_ks = ks_in_k(initialDP[adjacent_node*nodes_length + adjacent_next_node]);
                    for (int other_k : other_ks) {
                        if (this_k + other_k < -32) {
                            continue;
                        }
                        else if (this_k + other_k < 32) {
                            continue;
                        }
                        initialDP_copy[this_node*nodes_length + adjacent_next_node] |= (1 << int(32 + this_k + other_k));
                    }
                }
            }
        }
    }
}

uint64_t* computeAdjacencyTransitionsParallel(
    int nodes_length,
    uint64_t* initialDP,
    bool freeDP
    )
{
    // Deep Copy the initialDP
    uint64_t* initialDP_copy = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
                initialDP_copy[i*nodes_length + j] = initialDP[i*nodes_length + j];
        }
    }
    
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Calculate chunk size per thread
    int chunkSize = nodes_length / numThreads;

    for (int i = 0; i < numThreads; i++) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? nodes_length : start + chunkSize;

        futures.push_back(std::async(std::launch::async, processChunk, start, end, nodes_length, initialDP, initialDP_copy));
    }

    // Wait for all threads to complete
    for (auto &f : futures) {
        f.get();
    }

    if (freeDP) {
        delete[] initialDP;
    }

    return initialDP_copy;
}

void processSameBlockChunk(int start, int end, int nodes_length, uint64_t* sameBlockDP, uint64_t* initialDP_copy) {
    for (int this_node = start; this_node < end; this_node++) {
        for (int adjacent_node = 0; adjacent_node < nodes_length; adjacent_node++) {
            if (initialDP_copy[this_node*nodes_length + adjacent_node] == 0) {
                continue;
            }
            auto this_ks = ks_in_k(initialDP_copy[this_node*nodes_length + adjacent_node]);
            for (int this_k : this_ks) {
                // For each adjacent node, find its adjacent nodes and good k values
                for (int adjacent_next_node = 0; adjacent_next_node < nodes_length; adjacent_next_node++) {
                    if (sameBlockDP[adjacent_node*nodes_length + adjacent_next_node] == 0) {
                        continue;
                    }
                    std::vector<int> other_ks = ks_in_k(sameBlockDP[adjacent_node*nodes_length + adjacent_next_node]);
                    for (int other_k : other_ks) {
                        if (this_k + other_k < -32) {
                            continue;
                        }
                        else if (this_k + other_k < 32) {
                            continue;
                        }
                        initialDP_copy[this_node*nodes_length + adjacent_next_node] |= (1 << int(32 + this_k + other_k));
                    }
                }
            }
        }
    }
}

uint64_t* computeAdjacencyTransitionsSameBlockParallel(
    int nodes_length,
    uint64_t* filteredDP,
    uint64_t* sameBlockDP,
    bool freeDP
    )
{
    // Deep Copy the sameBlockDP
    uint64_t* filteredDP_copy = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
                filteredDP_copy[i*nodes_length + j] = filteredDP[i*nodes_length + j];
        }
    }
    
    const int numThreads = std::thread::hardware_concurrency();
    std::vector<std::future<void>> futures;

    // Calculate chunk size per thread
    int chunkSize = nodes_length / numThreads;

    for (int i = 0; i < numThreads; i++) {
        int start = i * chunkSize;
        int end = (i == numThreads - 1) ? nodes_length : start + chunkSize;

        futures.push_back(std::async(std::launch::async, processSameBlockChunk, start, end, nodes_length, sameBlockDP, filteredDP_copy));
    }

    // Wait for all threads to complete
    for (auto &f : futures) {
        f.get();
    }

    if (freeDP) {
        delete[] filteredDP;
    }

    return filteredDP_copy;
}

uint64_t* computeAdjacencyTransitions(
    int nodes_length,
    uint64_t* initialDP,
    bool freeDP
    )
{
    // Deep Copy the initialDP
    uint64_t* initialDP_copy = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
                initialDP_copy[i*nodes_length + j] = initialDP[i*nodes_length + j];
        }
    }

    // For each node, check if the next node is adjacent
    for (int this_node = 0; this_node < nodes_length; this_node++) {
        for (int adjacent_node = 0; adjacent_node < nodes_length; adjacent_node++) {
            if (initialDP[this_node*nodes_length + adjacent_node] == 0) {
                continue;
            }
            auto this_ks = ks_in_k(initialDP[this_node*nodes_length + adjacent_node]);
            for (int this_k : this_ks) {
                // For each adjacent node, find its adjacent nodes and good k values
                for (int adjacent_next_node = 0; adjacent_next_node < nodes_length; adjacent_next_node++) {
                    if (initialDP[adjacent_node*nodes_length + adjacent_next_node] == 0) {
                        continue;
                    }
                    std::vector<int> other_ks = ks_in_k(initialDP[adjacent_node*nodes_length + adjacent_next_node]);
                    for (int other_k : other_ks) {
                        if (this_k + other_k < -32) {
                            continue;
                        }
                        else if (this_k + other_k < 32) {
                            continue;
                        }
                        initialDP_copy[this_node*nodes_length + adjacent_next_node] |= (1 << int(32 + this_k + other_k));
                    }
                }
            }
        }
    }

    if (freeDP) {
        delete[] initialDP;
    }

    return initialDP_copy;
}

std::vector<int> count_overlap(
    int nodes_length,
    uint64_t* filteredGraph
) 
{
    // Count the number of True values per node
    std::vector<int> overlapCounts;
    for (int i = 0; i < nodes_length; i++) {
        int count_total = 0;
        for (int j = 0; j < nodes_length; j++) {
            int count_ = count_ks(filteredGraph[i*nodes_length + j]) -1; // Do not count itself
            if (count_ > 0) {
                count_total += count_;
            }
        }
        overlapCounts.push_back(count_total);
    }
    return overlapCounts;
}

std::vector<std::pair<int, int>> localEdgeMaxima(
    int nodes_length,
    uint64_t* initialDP,
    uint64_t* sameBlockDP,
    std::vector<int> overlapCounts
) 
{
    // for Each node Check all the adjacent nodes and determine if the edge is a local maxima
    std::vector<int> localMaximaNode;
    for (int this_node = 0; this_node < nodes_length; this_node++) {
        // Skip if the node has no overlap
        if (overlapCounts[this_node] == 0) {
            continue;
        }
        bool isLocalMaxima = true;
        for (int adjacent_node = 0; adjacent_node < nodes_length; adjacent_node++) {
            if (this_node == adjacent_node) {
                continue;
            }
            if (initialDP[this_node*nodes_length + adjacent_node] && (sameBlockDP[this_node*nodes_length + adjacent_node] == 0)) {
                if (overlapCounts[this_node] < overlapCounts[adjacent_node]) {
                    isLocalMaxima = false;
                    break;
                }
            }
        }
        if (isLocalMaxima) {
            localMaximaNode.push_back(this_node);
        }
    }

    // For each local maxima node, find the adjacent node with the highest overlap count
    std::vector<std::pair<int, int>> localMaximaPairs;
    for (int i = 0; i < localMaximaNode.size(); i++) {
        int this_node = localMaximaNode[i];
        int maxOverlap = 0;
        int maxOverlapNode = -1;
        for (int adjacent_node = 0; adjacent_node < nodes_length; adjacent_node++) {
            if (this_node == adjacent_node) {
                continue;
            }
            if (initialDP[this_node*nodes_length + adjacent_node] && (sameBlockDP[this_node*nodes_length + adjacent_node] == 0)) {
                if (overlapCounts[adjacent_node] > maxOverlap) {
                    maxOverlap = overlapCounts[adjacent_node];
                    maxOverlapNode = adjacent_node;
                }
            }
        }
        if (maxOverlapNode != -1) {
            localMaximaPairs.push_back(std::make_pair(this_node, maxOverlapNode));
        }
    }

    // at most take top 1% of nodes length at each pass
    std::sort(localMaximaPairs.begin(), localMaximaPairs.end(), [&overlapCounts](std::pair<int, int> a, std::pair<int, int> b) {
        return overlapCounts[a.first] > overlapCounts[b.first];
    });
    localMaximaPairs = std::vector<std::pair<int, int>>(localMaximaPairs.begin(), localMaximaPairs.begin() + std::min(int(localMaximaPairs.size()), 1 + int(nodes_length/100)));

    return localMaximaPairs;
}

void adjacency_count(
    int nodes_length,
    uint64_t* initialDP
) 
{
    // count adjacency numbers in the graph
    int adjacencyCount = 0;
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            if (initialDP[i*nodes_length + j] > 0) {
                adjacencyCount++;
            }
        }
    }
    std::cout << "Adjacency count: " << adjacencyCount << std::endl;
}

uint64_t* filterGraph(
    int nodes_length,
    uint64_t* initialDP,
    uint64_t* sameBlockDP
    )
{
    int iterations = 4;
    while (true) {
        // Some performance tracking:
        adjacency_count(nodes_length, initialDP);

        // Variables setup
        uint64_t* filteredGraph = initialDP;
        std::vector<std::pair<int, int>> localMaximaPairs;
        int iteration_index = 0;
        // iterations < std::sqrt(nodes_length) && 
        while (iterations < 17) {
            for(; iteration_index < iterations; iteration_index++) {
                std::cout << "Iteration: " << iteration_index << " / " << iterations << std::endl;
                filteredGraph = computeAdjacencyTransitionsParallel(nodes_length, filteredGraph, iteration_index > 0);
            }
            // Same block transitions before counting overlap
            // uint64_t* filteredGraph_ = computeAdjacencyTransitionsSameBlockParallel(nodes_length, filteredGraph, sameBlockDP, false);

            // Overlap count
            std::vector<int> overlapCounts = count_overlap(nodes_length, filteredGraph);

            // print overlap counts
            std::cout << "Overlap counts: ";
            for (auto count : overlapCounts) {
                if (count != 0) {
                    std::cout << count << " ";
                }
            }
            std::cout << std::endl;

            // Local edge maxima
            localMaximaPairs = localEdgeMaxima(nodes_length, initialDP, sameBlockDP, overlapCounts);
            
            // Check if found maximas
            if (!localMaximaPairs.empty()) {
                break;
            }
            else {
                iterations++;
            }
        }

        // Free the filteredGraph
        if (filteredGraph != initialDP) {
            std::cout << "Freeing filteredGraph" << std::endl;
            delete[] filteredGraph;
        }

        std::cout << "Local edge maxima computed" << std::endl;

        // print local maxima pairs
        std::cout << "Local maxima pairs: ";
        for (auto pair : localMaximaPairs) {
            std::cout << "(" << pair.first << ", " << pair.second << ") ";
        }
        std::cout << std::endl;

        // Break if no local edge maxima
        if (localMaximaPairs.empty()) {
            break;
        }

        // Delete local edge maxima edges
        for (int i = 0; i < localMaximaPairs.size(); i++) {
            int this_node = localMaximaPairs[i].first;
            int maxOverlapNode = localMaximaPairs[i].second;
            initialDP[this_node*nodes_length + maxOverlapNode] = 0;
            initialDP[maxOverlapNode*nodes_length + this_node] = 0;
        }
    }

    // Some performance tracking:
    adjacency_count(nodes_length, initialDP);

    std::cout << "Filtering done" << std::endl;
    return initialDP;
}

// Input is; Nodes Length, Initial DP (shape: (nodes_length, nodes_length, 64), type: bool)
py::array_t<int64_t> skeletonFilterGraph(
    int nodes_length,
    py::array_t<int64_t> initialNonTransitionDP_,
    py::array_t<int64_t> initialTransitionDP_,
    py::array_t<int64_t> sameBlockDP_   
    )
{
    auto dp_non_transition_unchecked = initialNonTransitionDP_.unchecked<2>(); // Access without bounds checking
    // Convert initialNonTransitionDP to a C++ array
    uint64_t* initialNonTransitionDP = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            initialNonTransitionDP[i*nodes_length + j] = dp_non_transition_unchecked(i, j);
        }
    }

    auto dp_transition_unchecked = initialTransitionDP_.unchecked<2>(); // Access without bounds checking
    // Convert initialTransitionDP to a C++ array
    uint64_t* initialTransitionDP = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            initialTransitionDP[i*nodes_length + j] = dp_transition_unchecked(i, j);
        }
    }

    auto sameBlock_dp_unchecked = sameBlockDP_.unchecked<2>(); // Access without bounds checking
    // Convert sameBlockDP to a C++ array
    uint64_t* sameBlockDP = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            sameBlockDP[i*nodes_length + j] = sameBlock_dp_unchecked(i, j);
        }
    }


    std::cout << "Initial DP converted to C++ array" << std::endl;

    std::cout << "No loop Same Block edges computation" << std::endl;
    uint64_t* emptyDP = new uint64_t[nodes_length*nodes_length];
    sameBlockDP = filterGraph(nodes_length, sameBlockDP, emptyDP);
    initialTransitionDP = filterGraph(nodes_length, initialTransitionDP, emptyDP);

    uint64_t* initialDP = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            initialDP[i*nodes_length + j] = initialNonTransitionDP[i*nodes_length + j] | initialTransitionDP[i*nodes_length + j];
        }
    }

    uint64_t* transitionDP = new uint64_t[nodes_length*nodes_length];
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            transitionDP[i*nodes_length + j] = initialTransitionDP[i*nodes_length + j] | sameBlockDP[i*nodes_length + j];
        }
    }

    // convert to k values
    int64_t* k_values_initial = DP_to_k(initialDP, nodes_length);
    // find all unique k values
    std::set<int> k_values_set;
    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
            k_values_set.insert(k_values_initial[i*nodes_length + j]);
        }
    }
    // print unique k values
    std::cout << "Unique k values: ";
    for (auto k : k_values_set) {
        std::cout << k << " ";
    }
    std::cout << std::endl;

    // Call the C++ function to filter the graph
    uint64_t* filteredGraph = filterGraph(nodes_length, initialDP, transitionDP);

    // convert to k values
    int64_t* k_values = DP_to_k(filteredGraph, nodes_length);

    // Convert the filteredGraph to a NumPy array 2D
    py::array_t<int64_t> result_DP(py::array::ShapeContainer({static_cast<long int>(nodes_length), static_cast<long int>(nodes_length)}));
    auto r_DP = result_DP.mutable_unchecked<2>(); // Now correctly a 3D array

    for (int i = 0; i < nodes_length; i++) {
        for (int j = 0; j < nodes_length; j++) {
                int64_t k = k_values[i*nodes_length + j];
                r_DP(i, j) = k;
        }
    }

    std::cout << "Filtered DP converted to NumPy array" << std::endl;

    return result_DP;
}

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

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual(int length_individual, int* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, int legth_initial_component, int* initial_component, bool build_valid_edges) {
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

// Input is; Nodes Length, Initial DP (shape: (nodes_length, nodes_length, 64), type: bool)
std::pair<py::array_t<int>, double> build_graph_from_individual_init(
    int length_individual,
    py::array_t<int> individual,
    int length_edges,
    py::array_t<int> edges,
    double factor_0,
    double factor_not_0,
    int legth_initial_component,
    py::array_t<int> initial_component,
    bool build_valid_edges
    )
{
    // Check length Individual equals length Edges
    if (length_individual != length_edges) {
        std::cout << "Length of individual and edges must be equal" << std::endl;
        throw std::invalid_argument("Length of individual and edges must be equal");
    }

    // Directly use the pointer to the data in the individual array
    int* individual_cpp = static_cast<int*>(individual.request().ptr);

    // Directly use the pointer to the data in the edges array
    int* edges_cpp = static_cast<int*>(edges.request().ptr);

    // Directly use the pointer to the data in the initial_component array
    int* initial_component_cpp = static_cast<int*>(initial_component.request().ptr);

    auto res = build_graph_from_individual(length_individual, individual_cpp, length_edges, edges_cpp, factor_0, factor_not_0, legth_initial_component, initial_component_cpp, build_valid_edges);

    double valid_edges_count = res.first;
    int* valid_edges = res.second;

    // Convert the valid_edges_python to a NumPy array 1D
    int fill_length = build_valid_edges ? length_edges : 1;

    py::array_t<int64_t> valid_edges_python(py::array::ShapeContainer({static_cast<long int>(fill_length)}));
    auto valid_edges_res = valid_edges_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (int i = 0; i < fill_length; i++) {
        valid_edges_res(i) = valid_edges[i];
    }

    return {valid_edges_python, valid_edges_count};
}

// Define a hash function for the tuple
struct hash_tuple {
    template <class T>
    int operator()(const T& tuple) const {
        auto hash1 = std::hash<int>{}(std::get<0>(tuple));
        auto hash2 = std::hash<int>{}(std::get<1>(tuple));
        auto hash3 = std::hash<int>{}(std::get<2>(tuple));
        auto hash4 = std::hash<int>{}(std::get<3>(tuple));

        return hash1 ^ hash2 ^ hash3 ^ hash4;  // Combine the hash values
    }
};

// Main function to build the graph from given inputs
std::pair<double, int*> build_graph_from_individual_patch(int length_individual, int* individual, int graph_raw_length, int* graph_raw, double factor_0, double factor_not_0, bool build_valid_edges) {
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

// Input is; Nodes Length, Initial DP (shape: (nodes_length, nodes_length, 64), type: bool)
std::pair<py::array_t<int>, double> build_graph_from_individual_patch_init(
    int length_individual,
    py::array_t<int> individual,
    int length_edges,
    py::array_t<int> edges,
    double factor_0,
    double factor_not_0,
    bool build_valid_edges
    )
{
    // Check length Individual equals length Edges
    if (length_individual != length_edges) {
        std::cout << "Length of individual and edges must be equal" << std::endl;
        throw std::invalid_argument("Length of individual and edges must be equal");
    }
    // Directly use the pointer to the data in the individual array
    int* individual_cpp = static_cast<int*>(individual.request().ptr);

    // Directly use the pointer to the data in the edges array
    int* edges_cpp = static_cast<int*>(edges.request().ptr);

    auto res = build_graph_from_individual_patch(length_individual, individual_cpp, length_edges, edges_cpp, factor_0, factor_not_0, build_valid_edges);

    double valid_edges_count = res.first;
    int* valid_edges = res.second;

    // Convert the valid_edges_python to a NumPy array 1D
    int fill_length = build_valid_edges ? length_edges : 1;

    py::array_t<int64_t> valid_edges_python(py::array::ShapeContainer({static_cast<long int>(fill_length)}));
    auto valid_edges_res = valid_edges_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (int i = 0; i < fill_length; i++) {
        valid_edges_res(i) = valid_edges[i];
    }

    return {valid_edges_python, valid_edges_count};
}

PYBIND11_MODULE(sheet_generation, m) {
    m.doc() = "pybind11 random walk solver for ThaumatoAnakalyptor"; // Optional module docstring

    m.def("solve_pyramid_random_walk_up", &solvePyramidRandomWalkUp, "Function to solve the pyramid random walk up problem in C++");

    m.def("solve_pyramid_random_walk_down", &solvePyramidRandomWalkDown, "Function to the pyramid random walk down problem in C++");

    m.def("solve_random_walk", &solveRandomWalk, 
            "Function to solve random walk problem in C++",
            py::arg("start_ids"),
            py::arg("start_ks"),
            py::arg("initial_node_usage_count"),
            py::arg("overlappThresholdFile"),
            py::arg("ids"),
            py::arg("nextNodes"),
            py::arg("kValues"),
            py::arg("same_block"),
            py::arg("umbilicusDirections"),
            py::arg("centroids"),
            py::arg("return_every_hundrethousandth") = false);

    m.def("graph_skeleton_filter", &skeletonFilterGraph, "Function to filter a Graph per DP skeleton in C++");

    m.def("build_graph_from_individual_cpp", &build_graph_from_individual_init, "Function to build graph from individual in C++");

    m.def("build_graph_from_individual_patch_cpp", &build_graph_from_individual_patch_init, "Function to build graph from individual in C++");
}
