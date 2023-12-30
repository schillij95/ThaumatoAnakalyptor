#include <Eigen/Dense>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <yaml-cpp/yaml.h>
#include <string>
#include <iostream>
#include <vector>
#include <random>
#include <future>
#include <thread>

// Global random number generator - initialized once for performance
std::mt19937 gen(std::random_device{}());
std::uniform_int_distribution<size_t> dist_pick;
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
   size_t operator()(const VolumeID& volumeID) const {
        // Combine hashes of individual fields
        return std::hash<int>()(volumeID.x) ^ std::hash<int>()(-volumeID.y) ^ std::hash<int>()(100000 * volumeID.z);
    }
};

using PatchID = int;  // Or any appropriate type
using K = int; // Or any appropriate type

struct Node {
    VolumeID volume_id;           // Struct for VolumeID
    PatchID patch_id;             // Type for PatchID
    std::shared_ptr<Node> next_nodes[6];          // Array of pointers to next nodes
    int valid_indices;            // Single integer for 'valid_indices'
    K k[6];                     // Array of 6 integers for 'k'
    float umbilicus_direction[3]; // Array of 3 floats for 'umbilicus_direction'
    float centroid[3];            // Array of 3 floats for 'centroid'
    float distance;               // Single float for 'distance'
    int index;                 // Single integer for 'index' in 'nodes' vector
};
// Using shared pointers for Node management
using NodePtr = std::shared_ptr<Node>;

using VolumeDict = std::unordered_map<VolumeID, std::unordered_map<PatchID, std::pair<NodePtr, K>>, VolumeIDHash>;
bool exists(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    auto it = dict.find(volID);
    if (it != dict.end()) {
        return it->second.find(patchID) != it->second.end();
    }
    return false;
}
K getKPrime(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    return dict.at(volID).at(patchID).second; // Add error handling as needed
}
NodePtr getNode(const VolumeDict& dict, VolumeID volID, PatchID patchID) {
    return dict.at(volID).at(patchID).first; // Add error handling as needed
}
bool existsForVolume(const VolumeDict& dict, VolumeID volID) {
    return dict.find(volID) != dict.end();
}
std::unordered_map<PatchID, std::pair<NodePtr, K>> getAllForVolume(const VolumeDict& dict, VolumeID volID) {
    return dict.at(volID); // Add error handling as needed
}

using NodeUsageCount = std::unordered_map<NodePtr, std::unordered_map<K, int>>;


std::pair<std::vector<NodePtr>, NodePtr> initializeNodes(
    VolumeID volID,
    PatchID patchID,
    py::array_t<int> ids, 
    py::array_t<int> nextNodes, 
    py::array_t<int> validIndices,
    py::array_t<int> kValues, 
    py::array_t<float> umbilicusDirections,
    py::array_t<float> centroids
    )
{
    size_t n = ids.shape(0); // Assuming the first dimension of each array is 'n'
    std::vector<NodePtr> nodes;
    nodes.reserve(n); // Reserve space for 'n' nodes

    std::cout << "Begin initialization" << std::endl;
    VolumeDict volume_dict;
    // First pass: Initialize nodes
    for (size_t i = 0; i < n; ++i) {
        auto node = std::make_shared<Node>();
        node->volume_id = {ids.mutable_unchecked<2>()(i, 0), 
                          ids.mutable_unchecked<2>()(i, 1), 
                          ids.mutable_unchecked<2>()(i, 2)};
        node->patch_id = ids.mutable_unchecked<2>()(i, 3);
        node->valid_indices = validIndices.mutable_unchecked<1>()(i);
        std::memcpy(node->k, kValues.mutable_unchecked<2>().data(i, 0), 6 * sizeof(int));
        std::memcpy(node->umbilicus_direction, umbilicusDirections.mutable_unchecked<2>().data(i, 0), 3 * sizeof(float));
        std::memcpy(node->centroid, centroids.mutable_unchecked<2>().data(i, 0), 3 * sizeof(float));
        // Calculate L2 norm for umbilicus_direction
        node->distance = std::sqrt(node->umbilicus_direction[0] * node->umbilicus_direction[0] + node->umbilicus_direction[1] * node->umbilicus_direction[1] + node->umbilicus_direction[2] * node->umbilicus_direction[2]);
        // Index is the index in the 'nodes' vector
        node->index = -1;
        
        // Add node to volume_dict
        volume_dict[node->volume_id][node->patch_id] = std::make_pair(node, -1);

        nodes.push_back(node);

        // assert valid indices to be in range 0 to 6 and k values to be in range -1 to 1
        assert(node->valid_indices >= 0 && node->valid_indices <= 6 && "Valid indices out of range");
        for (int j = 0; j < node->valid_indices; ++j) {
            assert(node->k[j] >= -1 && node->k[j] <= 1 && "K values out of range");
        }
    }
    std::cout << "First pass done" << std::endl;

    // Second pass: Set next_nodes pointers
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < nodes[i]->valid_indices; ++j) {
            VolumeID nextVolID = {nextNodes.mutable_unchecked<3>()(i, j, 0),
                                  nextNodes.mutable_unchecked<3>()(i, j, 1),
                                  nextNodes.mutable_unchecked<3>()(i, j, 2)};
            PatchID nextPatchID = nextNodes.mutable_unchecked<3>()(i, j, 3);

            // Find the node with the corresponding VolumeID and PatchID
            if (exists(volume_dict, nextVolID, nextPatchID)) {
                nodes[i]->next_nodes[j] = getNode(volume_dict, nextVolID, nextPatchID);
            } else {
                nodes[i]->next_nodes[j] = nullptr;
            }
        }
        for (int j = nodes[i]->valid_indices; j < 6; ++j) {
            nodes[i]->next_nodes[j] = nullptr;
        }
    }
    std::cout << "Second pass done" << std::endl;

    // Third pass: Filter out non-reciprocal next_nodes
    for (size_t i = 0; i < n; ++i) {
        for (int j = 0; j < nodes[i]->valid_indices; ++j) {
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
    NodePtr start_node = getNode(volume_dict, volID, patchID);
    return std::make_pair(nodes, start_node);
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

    for (size_t i = 0; i < final_nodes.size(); ++i) {
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

std::tuple<NodePtr, K, size_t> pick_start_node(
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

    std::vector<size_t> valid_indices;
    for (size_t i = 0; i < picked_nrs.size(); ++i) {
        if (picked_nrs[i] <= threshold) {
            valid_indices.push_back(i);
        }
    }

    assert(!valid_indices.empty() && "No nodes to pick from.");

    std::uniform_int_distribution<size_t> dist(0, valid_indices.size() - 1);
    size_t rand_index = valid_indices[dist(gen)];

    NodePtr node = nodes[rand_index];
    K k = ks[rand_index];

    return {node, k, rand_index};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::vector<size_t>> pick_start_nodes(
    std::vector<NodePtr> nodes, 
    std::vector<K> ks, 
    std::vector<int> picked_nrs,
    int nr_walks
    )
{
    std::vector<NodePtr> start_nodes;
    std::vector<K> start_ks;
    std::vector<size_t> start_indices;

    
    assert(!nodes.empty() && "No nodes to pick from.");

    float total = std::accumulate(picked_nrs.begin(), picked_nrs.end(), 0.0f);
    float mean_ = total / picked_nrs.size();

    auto min_it = std::min_element(picked_nrs.begin(), picked_nrs.end());
    float min_ = *min_it;

    float min_mean_abs = mean_ - min_;
    float threshold = min_ + min_mean_abs * 0.25f;

    std::vector<size_t> valid_indices;
    for (size_t i = 0; i < picked_nrs.size(); ++i) {
        if (picked_nrs[i] <= threshold) {
            valid_indices.push_back(i);
        }
    }

    assert(!valid_indices.empty() && "No nodes to pick from.");

    std::uniform_int_distribution<size_t> dist(0, valid_indices.size() - 1);
    for (int i = 0; i < nr_walks; ++i) {
        size_t rand_index = valid_indices[dist(gen)];

        NodePtr node = nodes[rand_index];
        K k = ks[rand_index];

        start_nodes.push_back(node);
        start_ks.push_back(k);
        start_indices.push_back(rand_index);
    }

    return {start_nodes, start_ks, start_indices};
}

std::tuple<NodePtr, K, size_t> pick_start_node_precomputed(
    std::vector<NodePtr>& nodes, 
    std::vector<K>& ks, 
    std::vector<size_t>& valid_indices)
{
    std::uniform_int_distribution<size_t> dist(0, valid_indices.size() - 1);
    size_t rand_index = valid_indices[dist(gen)];

    NodePtr node = nodes[rand_index];
    K k = ks[rand_index];

    if (node->index != rand_index) {
        std::cout << "Bug in pick_start_nodes_precomputed" << std::endl;
    }

    return {node, k, rand_index};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, std::vector<size_t>> pick_start_nodes_precomputed(
    std::vector<NodePtr>& nodes, 
    std::vector<K>& ks, 
    std::vector<size_t>& valid_indices,
    int nr_walks
    )
{
    std::vector<NodePtr> start_nodes;
    std::vector<K> start_ks;
    std::vector<size_t> start_indices;

    for (int i = 0; i < nr_walks; ++i) {
        size_t rand_index = valid_indices[dist_pick(gen)];

        NodePtr node = nodes[rand_index];
        K k = ks[rand_index];

        if (node->index != rand_index) {
            std::cout << "Bug in pick_start_nodes_precomputed" << std::endl;
        }

        start_nodes.push_back(node);
        start_ks.push_back(k);
        start_indices.push_back(rand_index);
    }

    return {start_nodes, start_ks, start_indices};
}

void precompute_pick(std::vector<NodePtr>& nodes, std::vector<long>& picked_nrs, std::vector<size_t>& valid_indices) {
    double mean_ = 0.0;
    double min_ = std::numeric_limits<double>::max();
    int count = 0;
    valid_indices.clear();
    std::vector<size_t> valid_indices_temp;
    for (size_t i = 0; i < picked_nrs.size(); ++i) {
        int neighbours = 0;
        for (size_t j = 0; j < nodes[i]->valid_indices; ++j) {
            if (nodes[i]->next_nodes[j] && nodes[i]->next_nodes[j]->index != -1) {
                ++neighbours;
            }
        }
        if (neighbours <= 3) {
            if (neighbours <= 2) { 
                valid_indices_temp.push_back(i); // possibly pick nodes with 3 or 4 neighbours
            } else {
                continue;
            }
            // Update mean and min statistics
            mean_ += (double)picked_nrs[i];
            count++;
            if (picked_nrs[i] < min_) {
                min_ = (double)picked_nrs[i];
            }
        }
    }

    mean_ = mean_ / (double)count;  

    double min_mean_abs = mean_ - min_;
    double threshold = min_mean_abs;

    for (size_t i = 0; i < valid_indices_temp.size(); ++i) {
        size_t index = valid_indices_temp[i];
        if ((double)picked_nrs[index] <= threshold) {
            valid_indices.push_back(index);
        }
    }
    gen = std::mt19937(std::random_device{}());
    dist_pick = std::uniform_int_distribution<size_t>(0, valid_indices.size() - 1);
}

std::pair<NodePtr, K> pick_next_node(std::mt19937& gen_, std::uniform_int_distribution<>& distrib, const Node& node) {
    // Check if there are no valid next nodes
    if (node.valid_indices == -1) {
        return {nullptr, -10};
    }

    // Return the randomly picked valid next node
    int index = distrib(gen_)%node.valid_indices;
    auto next_node = node.next_nodes[index];
    auto k = node.k[index];
    return {next_node, k};
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
    NodePtr node, 
    K k, 
    const VolumeDict& volume_dict,
    float max_umbilicus_difference,
    int step_size = 20, 
    int away_dist_check = 500) 
{
    VolumeID last_volume_quadrant = {INT_MIN, INT_MIN, INT_MIN}; // Initialize with an unlikely value
    std::unordered_set<VolumeID, VolumeIDHash> computed_volumes;

    // Continue if the node is already in volume_dict
    if (exists(volume_dict, node->volume_id, node->patch_id)) {
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
    for (size_t i = 0; i < walk.size(); ++i) {
        NodePtr node = walk[i];
        K k = ks[i];
        if (!check_overlapp_node(node, k, volume_dict, max_umbilicus_difference, step_size, away_dist_check)) {
            return false;
        }
    }
    return true;
}

// std::tuple<std::vector<NodePtr>, std::vector<K>, std::string, bool, bool> random_walk(
std::tuple<std::vector<NodePtr>, std::vector<K>, bool, bool> random_walk(
    std::mt19937& gen_,
    std::uniform_int_distribution<>& distrib,
    NodePtr start_node, 
    K start_k, 
    const VolumeDict& volume_dict,
    const Eigen::Vector2f& sheet_z_range, 
    const Eigen::Vector2i& sheet_k_range,
    float max_umbilicus_difference,
    int max_steps = 20, 
    int max_tries = 6, 
    int min_steps = 5)
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

            auto res = pick_next_node(gen_, distrib, *walk.back());
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
            if (k < -1 || k > 1) {
                std::cout << "Invalid k value: " << k << std::endl;
                continue;
            }

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

        if (!exists(volume_dict, node_->volume_id, node_->patch_id)) {
            new_node_flag = true;
        }

        // Check for loop closure
        if (existsForVolume(volume_dict, node_->volume_id)) {
            auto patchKMap = getAllForVolume(volume_dict, node_->volume_id);
            for (const auto& [key_patch, k_prime_pair] : patchKMap) {
                if (k_prime_pair.second == current_k) {
                    if (node_->patch_id == k_prime_pair.first->patch_id) {
                        if (steps >= min_steps) {
                            if (new_node_flag) {
                                if (check_overlapp_walk(walk, ks, volume_dict, max_umbilicus_difference)) {
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
    }

    // return {empty_walk, empty_ks, "Loop not closed in max_steps", false, false};
    return {empty_walk, empty_ks, false, false};
}

std::tuple<std::vector<NodePtr>, std::vector<K>, bool> walk_aggregation_func(
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

    for (size_t i = 0; i < walk.size(); ++i) {
        NodePtr node = walk[i];
        K k = ks[i];

        // Increment the usage count of the node with the specific k value
        int& count = node_usage_count[node][k];
        count++;

        // Aggregate node if it meets criteria and hasn't been aggregated before
        if (count >= walk_aggregation_threshold) {
            // Check if the node is already in volume_dict
            bool isAlreadyAggregated = exists(volume_dict, node->volume_id, node->patch_id);
            if (isAlreadyAggregated) {
                continue;
            }        

            if (!check_overlapp_node(node, k, volume_dict, max_umbilicus_difference)) {
                return {aggregated_nodes, aggregated_ks, false};
            }

            aggregated_nodes.push_back(node);
            aggregated_ks.push_back(k);
        }
    }

    // Update volume_dict with the newly aggregated nodes
    for (size_t i = 0; i < aggregated_nodes.size(); ++i) {
        NodePtr node = aggregated_nodes[i];
        K k = aggregated_ks[i];

        // Update volume_dict with the newly aggregated node
        volume_dict[node->volume_id][node->patch_id] = std::make_pair(node, k);
    }

    bool success = !aggregated_nodes.empty();

    if (success) {
        // Append aggregated nodes and ks to nodes and ks vectors
        nodes_final.insert(nodes_final.end(), aggregated_nodes.begin(), aggregated_nodes.end());
        ks_final.insert(ks_final.end(), aggregated_ks.begin(), aggregated_ks.end());
        // Append same number of 0 to picked_nrs vector
        picked_nrs.insert(picked_nrs.end(), aggregated_nodes.size(), 0);
        // Update the index of the aggregated nodes
        for (size_t i = 0; i < aggregated_nodes.size(); ++i) {
            aggregated_nodes[i]->index = nodes_final.size() - aggregated_nodes.size() + i;
        }
    }

    return {aggregated_nodes, aggregated_ks, success};
}

void update_picked_nr(std::vector<NodePtr>& nodes, std::vector<long>& picked_nrs, int index, int value) {
    if (index < 0) {
        return;
    }
    picked_nrs[index] += value;
    if (picked_nrs[index] < 0) {
        picked_nrs[index] = 0;
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
    std::vector<NodePtr> start_nodes,
    std::vector<K> start_ks,
    const VolumeDict& volume_dict,
    const Eigen::Vector2f& sheet_z_range, 
    const Eigen::Vector2i& sheet_k_range,
    float max_umbilicus_difference,
    int max_steps = 20, 
    int max_tries = 6, 
    int min_steps = 5
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
        auto [walk, walk_ks, success, new_node] = random_walk(gen_, distrib, start_nodes[i], start_ks[i], volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps);
        walks.push_back(walk);
        ks.push_back(walk_ks);
        // messages.push_back(message);
        successes.push_back(success);
        new_nodes.push_back(new_node);
    }

    // return {walks, ks, messages, successes, new_nodes};
    return {walks, ks, successes, new_nodes};
}


std::tuple<std::vector<NodePtr>, std::vector<K>> solve(
    NodePtr start_node,
    K start_k,
    Config& config,
    int walksPerThread = 1000,
    int numThreads = 28
    ) 
{
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
    int max_steps = config.max_steps;
    int max_tries = config.max_tries;
    int min_steps = config.min_steps;
    int min_steps_start = min_steps;
    int min_end_steps = config.min_end_steps;

    VolumeDict volume_dict;
    std::vector<NodePtr> nodes;
    std::vector<K> ks;
    std::vector<long> picked_nrs;
    start_node->index = 0;
    std::vector<size_t> valid_indices;

    if (!continue_walks) {
        nodes.push_back(start_node);
        ks.push_back(start_k);
        picked_nrs.push_back(0);
        precompute_pick(nodes, picked_nrs, valid_indices);
        // Add start_node to volume_dict
        volume_dict[start_node->volume_id][start_node->patch_id] = std::make_pair(start_node, start_k);
    }

    int nr_unchanged_walks = 0;
    NodeUsageCount node_usage_count; // Map to track node usage count with specific k values
    int walk_aggregation_count = 0;
    int total_walks = 0;

    // numm threads gens
    std::vector<std::mt19937> gen_;
    // making initialization distribution between 0 and 1000000
    std::uniform_int_distribution<> dist(0, 1000000);
    
    for (int i = 0; i < numThreads; ++i) {
        // std::mt19937 gen_t_(std::random_device{}());
        // fixed seed
        std::mt19937 gen_t_(dist(gen));
        gen_.push_back(gen_t_);
    }

    while (true)
    {
        std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << nr_unchanged_walks << " random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        // Display message counts at the end of solve function
        if (total_walks++ % 100 == 0)
        {
            std::cout << "\033[1;36m" << "[ThaumatoAnakalyptor]: Random Walk Messages Summary:" << "\033[0m" << std::endl;
            for (const auto& pair : message_count) {
                std::cout << "\033[1;36m" << "  \"" << pair.first << "\": " << pair.second << "\033[0m" << std::endl;
            }
            std::cout << "\033[1;32m" << "[ThaumatoAnakalyptor]: Starting " << nr_unchanged_walks << " random walk. Nr good nodes: " << nodes.size() << "\033[0m" << std::endl;
        }
        if (nr_unchanged_walks > max_unchanged_walks && walk_aggregation_count != 0) { //  && (/* More checks*/)
            nr_unchanged_walks = 0;
            // set picked_nrs to 0
            for (size_t i = 0; i < picked_nrs.size(); ++i) {
                picked_nrs[i] = 0;
            }
            precompute_pick(nodes, picked_nrs, valid_indices);

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
        std::vector<size_t> indices_s;
        int nrWalks = walksPerThread * numThreads;
        std::tie(sns, sks, indices_s) = pick_start_nodes_precomputed(nodes, ks, valid_indices, nrWalks);
        for (size_t i = 0; i < indices_s.size(); ++i) {
            update_picked_nr(nodes, picked_nrs, indices_s[i], 1);
        }


        std::vector<std::future<ThreadResult>> futures;
        for (int i = 0; i < numThreads; ++i) {
            auto start_nodes = std::vector<NodePtr>(sns.begin() + i * walksPerThread, sns.begin() + (i + 1) * walksPerThread);
            auto start_ks = std::vector<K>(sks.begin() + i * walksPerThread, sks.begin() + (i + 1) * walksPerThread);
            // futures.push_back(std::async(std::launch::async, threadRandomWalk, std::ref(gen_[i]), walksPerThread, start_nodes, start_ks, volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps));
            futures.push_back(std::async(std::launch::async, threadRandomWalk, walksPerThread, start_nodes, start_ks, volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps));
        }

        // auto [walk, walk_ks, success] = random_walk(sn, sk, volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps);


        std::vector<std::vector<NodePtr>> walks_futures;
        std::vector<std::vector<K>> walk_ks_futures;
        // std::vector<std::string> messages_futures;
        std::vector<bool> successes_futures;
        std::vector<bool> new_nodes_futures;
        for (auto& future : futures) {
            // auto [walks_, walk_ks_, messages_, successes_, new_nodes_] = future.get();
            auto [walks_, walk_ks_, successes_, new_nodes_] = future.get();
            for (size_t i = 0; i < walks_.size(); ++i) {
                walks_futures.push_back(walks_[i]);
                walk_ks_futures.push_back(walk_ks_[i]);
                // messages_futures.push_back(messages_[i]);
                successes_futures.push_back(successes_[i]);
                new_nodes_futures.push_back(new_nodes_[i]);
            }
        }

        // // single threaded call to threadRandomWalk
        // auto [walks_futures, walk_ks_futures, messages_futures, successes_futures, new_nodes_futures] = threadRandomWalk(gen_[0], nrWalks, sns, sks, volume_dict, sheet_z_range, sheet_k_range, max_umbilicus_difference, max_steps, max_tries, min_steps);

        // for (size_t i = 0; i < messages_futures.size(); ++i) {
        //     message_count[messages_futures[i]]++;
        // }
        // Increment the count for the returned message
        // message_count[message]++;

        // Loop over all the walks by iterating trough them
        bool total_aggregation_success = false;
        for (size_t i = 0; i < walks_futures.size(); ++i) {
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

            // update the picked_nr of each node in the walk -/+5 depending on the success
            for (NodePtr node : walk) {
                update_picked_nr(nodes, picked_nrs, node->index, new_node ? -5 : 5);
            }

            if (!new_node) {
                nr_unchanged_walks++;
                continue;
            }

            auto [walk_aggregated, walk_ks_aggregated, success_aggregated] = walk_aggregation_func(
                nodes, ks, picked_nrs, walk, walk_ks, volume_dict, node_usage_count, max_umbilicus_difference, walk_aggregation_threshold);

            if (!success_aggregated) {
                nr_unchanged_walks++;
                continue;
            }
            else {
                total_aggregation_success = true;
            }

            nr_unchanged_walks = 0;
            walk_aggregation_count += walk_aggregated.size();

            
            // yellow color
            // std::cout << "\033[1;33m" << "[ThaumatoAnakalyptor]: Added " << walk_aggregation_count << " sheet patches." << "\033[0m" << std::endl;
        }
        // Update valid_indices from picked_nrs
        if (total_aggregation_success) {
            precompute_pick(nodes, picked_nrs, valid_indices);
        }
    }
    return {nodes, ks};
}

std::pair<std::vector<double>, std::vector<double>> process_array(py::array_t<double> input_array, int some_integer, bool some_boolean) {
    // Example processing (to be replaced with actual logic)
    // Assuming the input array is 2D, flattening it into 1D
    py::buffer_info buf_info = input_array.request();
    auto *ptr = static_cast<double *>(buf_info.ptr);

    std::vector<double> output1, output2;

    for (size_t i = 0; i < buf_info.shape[0]; i++) {
        for (size_t j = 0; j < buf_info.shape[1]; j++) {
            double val = ptr[i * buf_info.shape[1] + j];
            std::cout << val << std::endl;
            output1.push_back(val * some_integer); // Example operation
            if (some_boolean) {
                output2.push_back(val * 2); // Another example operation
            }
        }
        std::cout << "New row" << std::endl;
    }

    return {output1, output2};
}

void loadOverlappThreshold(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);

    // Example of accessing a value
    if (config["sample_ratio_score"]) {
        float sampleRatioScore = config["sample_ratio_score"].as<float>();
        std::cout << "sample_ratio_score: " << sampleRatioScore << std::endl;
    }

    // Load 'sheet_k_range' as a pair of integers
    if (config["sheet_k_range"]) {
        // Assuming 'sheet_k_range' is a sequence of two integers in the YAML file
        std::pair<int, int> sheetKRange;
        sheetKRange.first = config["sheet_k_range"][0].as<int>(); // First value
        sheetKRange.second = config["sheet_k_range"][1].as<int>(); // Second value

        std::cout << "sheet_k_range: (" << sheetKRange.first << ", " << sheetKRange.second << ")" << std::endl;
    }

    // Add code to load other parameters as needed
}

std::pair<py::array_t<int>, py::array_t<int>> solveRandomWalk(
    int vol1,
    int vol2,
    int vol3,
    int patchID,
    const std::string &overlappThresholdFile,
    py::array_t<int> ids, 
    py::array_t<int> nextNodes, 
    py::array_t<int> validIndices,
    py::array_t<int> kValues, 
    py::array_t<float> umbilicusDirections,
    py::array_t<float> centroids) 
{
    std::cout << "Begin solveRandomWalk" << std::endl;
    std::cout << "Starting node: " << vol1 << " " << vol2 << " " << vol3 << " " << patchID << std::endl;
    VolumeID start_vol_id = {vol1, vol2, vol3};
    PatchID start_patchID = patchID;

    Config config;
    config.load(overlappThresholdFile);
    config.print();

    std::cout << "Config loaded" << std::endl;

    auto init_res = initializeNodes(start_vol_id, start_patchID, ids, nextNodes, validIndices, kValues, umbilicusDirections, centroids);
    std::vector<NodePtr> nodes = init_res.first;
    NodePtr start_node = init_res.second;

    std::cout << "Nodes initialized" << std::endl;

    K start_k = 0; // Modify as needed

    auto [final_nodes, final_ks] = solve(start_node, start_k, config);

    std::cout << "Solve done" << std::endl;

    // Convert final_nodes and final_ks to a format suitable for Python
    return process_result(final_nodes, final_ks);
}

PYBIND11_MODULE(sheet_generation, m) {
    m.doc() = "pybind11 random walk solver for ThaumatoAnakalyptor"; // Optional module docstring

    m.def("solve_random_walk", &solveRandomWalk, "Function to solve random walk problem in C++");
}
