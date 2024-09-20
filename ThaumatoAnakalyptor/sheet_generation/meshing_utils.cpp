/*
Author: Julian Schilliger - ThaumatoAnakalyptor - 2024
*/ 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <tuple>
#include <algorithm>
#include <execution>  // C++17 parallel execution policies
#include <thread>
#include <mutex>  // for thread-safe access to shared resources

// Define 3D point and Triangle types
using Point3D = std::array<float, 3>;
using Triangle = std::array<Point3D, 3>;

// Thread-safe vector appending with mutex
std::mutex pairs_mutex;

// Helper function to compute the centroid of a triangle
Point3D compute_centroid(const Triangle& tri) {
    Point3D centroid = {0, 0, 0};
    for (const auto& vertex : tri) {
        centroid[0] += vertex[0];
        centroid[1] += vertex[1];
        centroid[2] += vertex[2];
    }
    centroid[0] /= 3.0f;
    centroid[1] /= 3.0f;
    centroid[2] /= 3.0f;
    return centroid;
}

// Helper function to compute the Euclidean distance between two points
float distance(const Point3D& a, const Point3D& b) {
    return std::sqrt(std::pow(a[0] - b[0], 2) + std::pow(a[1] - b[1], 2) + std::pow(a[2] - b[2], 2));
}

// k-d tree node
struct KDNode {
    Point3D point;     // Centroid of the triangle
    size_t index;         // Index of the triangle
    KDNode* left = nullptr;
    KDNode* right = nullptr;
};

// Function to build a k-d tree from the centroids of triangles
KDNode* build_kd_tree(std::vector<std::pair<Point3D, size_t>>& centroids, size_t depth = 0) {
    if (centroids.empty()) return nullptr;

    size_t axis = depth % 3;  // Cycle through x, y, z axes
    size_t median = centroids.size() / 2;

    // Sort the centroids by the current axis
    std::nth_element(centroids.begin(), centroids.begin() + median, centroids.end(),
        [axis](const std::pair<Point3D, size_t>& a, const std::pair<Point3D, size_t>& b) {
            return a.first[axis] < b.first[axis];
        });

    // Create a new node and recursively build the left and right subtrees
    KDNode* node = new KDNode{centroids[median].first, centroids[median].second};
    std::vector<std::pair<Point3D, size_t>> left(centroids.begin(), centroids.begin() + median);
    std::vector<std::pair<Point3D, size_t>> right(centroids.begin() + median + 1, centroids.end());

    node->left = build_kd_tree(left, depth + 1);
    node->right = build_kd_tree(right, depth + 1);
    
    return node;
}

// Function to query the k-d tree for pairs of centroids within a given radius
void query_radius(KDNode* node, const Point3D& point, float radius, size_t depth, 
                  std::vector<std::pair<size_t, size_t>>& pairs, size_t current_index) {
    if (!node) return;

    // Compute the distance between the current point and the node's point
    float d = distance(point, node->point);
    if (d <= radius && node->index != current_index) {
        pairs.emplace_back(current_index, node->index);  // Store the pair of indices
    }

    // Check which side of the splitting plane to search
    size_t axis = depth % 3;
    float diff = point[axis] - node->point[axis];

    if (diff < 0) {
        // Search the left subtree
        query_radius(node->left, point, radius, depth + 1, pairs, current_index);
        if (std::fabs(diff) <= radius) {
            query_radius(node->right, point, radius, depth + 1, pairs, current_index);
        }
    } else {
        // Search the right subtree
        query_radius(node->right, point, radius, depth + 1, pairs, current_index);
        if (std::fabs(diff) <= radius) {
            query_radius(node->left, point, radius, depth + 1, pairs, current_index);
        }
    }
}

// Function to compute cross product of two vectors
std::array<float, 3> cross_product(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

// Function to compute dot product of two vectors
float dot_product(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Function to subtract two 3D points (vector difference)
std::array<float, 3> subtract(const std::array<float, 3>& a, const std::array<float, 3>& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

// Helper function to compute edge directions
std::array<std::array<float, 3>, 3> edge_directions(const Triangle& tri) {
    return {subtract(tri[1], tri[0]), subtract(tri[2], tri[1]), subtract(tri[0], tri[2])};
}

// Function to compute the normal of a triangle
std::array<float, 3> normal(const Triangle& tri) {
    return cross_product(subtract(tri[1], tri[0]), subtract(tri[2], tri[0]));
}

// Project triangle vertices onto an axis
std::pair<float, float> project(const Triangle& tri, const std::array<float, 3>& axis) {
    float min_proj = dot_product(tri[0], axis);
    float max_proj = min_proj;
    for (int i = 1; i < 3; ++i) {
        float proj = dot_product(tri[i], axis);
        if (proj < min_proj) min_proj = proj;
        if (proj > max_proj) max_proj = proj;
    }
    return {min_proj, max_proj};
}

// Check if two projections overlap
bool overlaps(float min1, float max1, float min2, float max2) {
    return max1 >= min2 && max2 >= min1;
}

// Perform the Separating Axis Test (SAT)
bool separating_axis_test(const Triangle& tri1, const Triangle& tri2, const std::array<float, 3>& axis) {
    auto [min1, max1] = project(tri1, axis);
    auto [min2, max2] = project(tri2, axis);
    return overlaps(min1, max1, min2, max2);
}

// Check if two triangles intersect using SAT
bool check_triangle_intersection(const Triangle& tri1, const Triangle& tri2) {
    std::vector<std::array<float, 3>> axis_tests;

    // Add triangle normals
    axis_tests.push_back(normal(tri1));
    axis_tests.push_back(normal(tri2));

    // Add edge cross products
    auto edges1 = edge_directions(tri1);
    auto edges2 = edge_directions(tri2);

    for (const auto& edge1 : edges1) {
        for (const auto& edge2 : edges2) {
            axis_tests.push_back(cross_product(edge1, edge2));
        }
    }

    // Perform SAT on all axes
    for (const auto& axis : axis_tests) {
        if (std::abs(dot_product(axis, axis)) < 1e-6) continue;  // Skip degenerate axes
        if (!separating_axis_test(tri1, tri2, axis)) {
            return false;
        }
    }
    return true;  // No separating axis found, so triangles intersect
}

// Check if two triangles have winding angles within the allowed range
bool check_winding_angle(float winding1, float winding2, float angle_range) {
    return std::abs(winding1 - winding2) < angle_range;
}

// Function to compute intersecting triangles based on a specified radius and winding angle difference
std::vector<bool> compute_intersections_and_winding_angles(
    const std::vector<Triangle>& triangles, 
    const std::vector<float>& winding_angles, 
    float angle_range, 
    float radius) {

    std::cout << "Computing intersections and winding angles..." << std::endl;

    size_t num_triangles = triangles.size();
    std::vector<bool> intersecting_triangles(num_triangles, false);

    // Compute centroids for the triangles
    std::vector<std::pair<Point3D, size_t>> centroids(num_triangles);
    for (size_t i = 0; i < num_triangles; ++i) {
        centroids[i] = std::make_pair(compute_centroid(triangles[i]), i);
    }

    std::cout << "Centroid size: " << centroids.size() << " num_triangles: " << num_triangles << std::endl;

    std::cout << "Building k-d tree..." << std::endl;

    // Build k-d tree for the centroids
    KDNode* kd_root = build_kd_tree(centroids); // Centroids will be reordered!

    std::cout << "Querying k-d tree..." << std::endl;

    // Find pairs of triangles whose centroids are within the given radius
    std::vector<std::pair<size_t, size_t>> pairs;

    // Parallel query of the k-d tree
    std::for_each(std::execution::par, centroids.begin(), centroids.end(), [&](const std::pair<Point3D, size_t>& centroid) {
        std::vector<std::pair<size_t, size_t>> local_pairs;
        query_radius(kd_root, centroid.first, radius, 0, local_pairs, centroid.second);

        // Append local pairs to the global pairs vector in a thread-safe way
        std::lock_guard<std::mutex> lock(pairs_mutex);
        pairs.insert(pairs.end(), local_pairs.begin(), local_pairs.end());
    });

    std::cout << "Checking intersections and winding angles..." << std::endl;

    // Parallel processing of the triangle pairs for intersection and winding angle checks
    std::for_each(std::execution::par, pairs.begin(), pairs.end(), [&](const std::pair<size_t, size_t>& pair) {
        size_t i = pair.first;
        size_t j = pair.second;

        if (!check_winding_angle(winding_angles[i], winding_angles[j], angle_range)) {
            if (check_triangle_intersection(triangles[i], triangles[j])) {
                intersecting_triangles[i] = true;
                intersecting_triangles[j] = true;
            }
        }
    });

    std::cout << "Done!" << std::endl;

    // Clean up the k-d tree
    delete kd_root;

    return intersecting_triangles;
}

PYBIND11_MODULE(meshing_utils, m) {
    m.doc() = "pybind11 module for mesh processing";

    m.def("compute_intersections_and_winding_angles", &compute_intersections_and_winding_angles, "Function to compute intersecting triangles based on a specified radius and winding angle difference.");
}