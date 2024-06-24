/*
Author: Julian Schilliger - ThaumatoAnakalyptor - 2024
*/ 

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <archive.h>
#include <archive_entry.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <string>
#include <mutex>
#include <execution>
#include <thread>
#include <iomanip>
#include <cstdio>    // For mkstemp
#include <unistd.h>  // For unlink
#include <filesystem>
#include "happly.h"
#include "nanoflann.h"

namespace fs = std::filesystem;
namespace py = pybind11;
namespace nf = nanoflann;

struct Point {
    float x, y, z, w;   // Coordinates and winding angle
    float nx, ny, nz;   // Normal vector components
    unsigned char r, g, b; // Color components
    bool marked_for_deletion = false;  // Flag to mark points for deletion

    // Default constructor
    Point() = default;

    // Define a constructor that initializes all members
    explicit Point(float x, float y, float z, float w,
          float nx, float ny, float nz,
          unsigned char r, unsigned char g, unsigned char b,
          bool marked_for_deletion)
        : x(x), y(y), z(z), w(w),
          nx(nx), ny(ny), nz(nz),
          r(r), g(g), b(b),
          marked_for_deletion(marked_for_deletion) {}
};

class PointCloud {
public:
    std::vector<Point> pts;

    // Default constructor
    PointCloud() = default;

    // Initialize with a vector of points
    explicit PointCloud(const std::vector<Point>& points) : pts(points) {}

    // Initialize with a NumPy array
    explicit PointCloud(py::array_t<float> points, py::array_t<float> normals, py::array_t<float> colors) {
        auto points_r = points.unchecked<2>();
        auto normals_r = normals.unchecked<2>();
        auto colors_r = colors.unchecked<2>();

        size_t total_points = points_r.shape(0);
        pts.reserve(total_points);

        for (size_t i = 0; i < total_points; ++i) {
            pts.emplace_back(
                points_r(i, 0), points_r(i, 1), points_r(i, 2), points_r(i, 3), // coordinates and winding angle
                normals_r(i, 0), normals_r(i, 1), normals_r(i, 2),  // normal vector components
                static_cast<unsigned char>(colors_r(i, 0) * 255), static_cast<unsigned char>(colors_r(i, 1) * 255), static_cast<unsigned char>(colors_r(i, 2) * 255),  // color components
                false  // not marked for deletion
            );
        }
    }

    // Initialize with a NumPy array, points only
    explicit PointCloud(py::array_t<float> points) {
        auto points_r = points.unchecked<2>();

        size_t total_points = points_r.shape(0);
        pts.reserve(total_points);

        for (size_t i = 0; i < total_points; ++i) {
            pts.emplace_back(
                points_r(i, 0), points_r(i, 1), points_r(i, 2), points_r(i, 3), // coordinates and winding angle
                0.0f, 0.0f, 0.0f,  // normal vector components (defaults)
                0, 0, 0,  // color components (defaults)
                false  // not marked for deletion (defaults)
            );
        }
    }

    // Add a point to the cloud
    void addPoint(const Point& point) {
        pts.push_back(point);
    }

    // Get number of points in the cloud
    size_t size() const {
        return pts.size();
    }

    // Clear all points
    void clear() {
        pts.clear();
    }

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const { return pts.size(); }

    // Returns the dim'th component of the idx'th point in the class:
    inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
        if (dim == 0) return pts[idx].x;
        else if (dim == 1) return pts[idx].y;
        else if (dim == 2) return pts[idx].z;
        return 0; // Should never be reached
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    template <class BBOX>
    bool kdtree_get_bbox(BBOX&) const { return false; }
};

typedef nf::KDTreeSingleIndexAdaptor<
    nf::L2_Simple_Adaptor<double, PointCloud>,
    PointCloud,
    3, // dimensionality
    size_t // using size_t for indexing
> MyKDTree;

std::string format_filename(int number) {
    std::ostringstream stream;
    stream << std::setw(6) << std::setfill('0') << number;
    return stream.str();
}

class PointCloudLoader {
public:
    PointCloudLoader(const std::vector<std::tuple<std::vector<int>, int, double>>& node_data, const std::string& base_path, bool verbose)
        : node_data_(node_data), base_path_(base_path), verbose(verbose) {}

    bool extract_ply_from_tar(const std::string& tar_path, const std::string& ply_filename, std::string& out_ply_content) {
        struct archive *a;
        struct archive_entry *entry;
        int r;

        a = archive_read_new();
        archive_read_support_filter_all(a);
        archive_read_support_format_all(a);
        r = archive_read_open_filename(a, tar_path.c_str(), 65536); // 64 KB = 65,536 Bytes is the buffer size.
        if (r != ARCHIVE_OK) {
            return false;
        }

        while (archive_read_next_header(a, &entry) == ARCHIVE_OK) {
            std::string currentFile = archive_entry_pathname(entry);
            if (currentFile == ply_filename) {
                const void* buff;
                size_t size;
                la_int64_t offset;

                std::ostringstream oss;
                while (archive_read_data_block(a, &buff, &size, &offset) == ARCHIVE_OK) {
                    oss.write(static_cast<const char*>(buff), size);
                }
                out_ply_content = oss.str();
                archive_read_free(a);
                return true;
            }
            archive_read_data_skip(a);  // Skip files that do not match
        }
        archive_read_free(a);
        return false;
    }

    void process_node(size_t start, size_t end) {
        for (size_t index = start; index < end; ++index) {
            size_t numpy_offset = offset_per_node[index];
            auto& node = node_data_[index];
            const auto& xyz = std::get<0>(node);
            size_t patch_nr = std::get<1>(node);
            double winding_angle = std::get<2>(node);

            std::string tar_path = base_path_ + "/" + format_filename(xyz[0]) + "_" + format_filename(xyz[1]) + "_" + format_filename(xyz[2]) + ".tar";
            std::string ply_file_name = "surface_" + std::to_string(patch_nr) + ".ply";
            std::string ply_content;

            if (!extract_ply_from_tar(tar_path, ply_file_name, ply_content)) {
                std::cerr << "Failed to extract PLY file from TAR archive" << std::endl;
                return;
            }

            try {
                // Use std::istringstream to read PLY content from string
                std::istringstream plyStream(ply_content);

                // Load PLY content using hapPLY
                happly::PLYData plyIn(plyStream);

                std::tuple<std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>, std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>, std::tuple<std::vector<unsigned char>, std::vector<unsigned char>, std::vector<unsigned char>>> vertices = plyIn.getVertexData();

                std::vector<double> x = std::get<0>(std::get<0>(vertices));
                std::vector<double> y = std::get<1>(std::get<0>(vertices));
                std::vector<double> z = std::get<2>(std::get<0>(vertices));

                std::vector<double> nx = std::get<0>(std::get<1>(vertices));
                std::vector<double> ny = std::get<1>(std::get<1>(vertices));
                std::vector<double> nz = std::get<2>(std::get<1>(vertices));

                std::vector<unsigned char> r = std::get<0>(std::get<2>(vertices));
                std::vector<unsigned char> g = std::get<1>(std::get<2>(vertices));
                std::vector<unsigned char> b = std::get<2>(std::get<2>(vertices));

                std::vector<Point> points;
                points.reserve(x.size());
                for (size_t i = 0; i < x.size(); ++i) {
                    points.emplace_back(
                        static_cast<float>(x[i]), static_cast<float>(y[i]), static_cast<float>(z[i]), static_cast<float>(winding_angle), // coordinates and winding angle
                        static_cast<float>(nx[i]), static_cast<float>(ny[i]), static_cast<float>(nz[i]),  // normal vector components
                        static_cast<unsigned char>(r[i]), static_cast<unsigned char>(g[i]), static_cast<unsigned char>(b[i]),  // color components
                        false  // not marked for deletion
                    );
                }

                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
                all_points.insert(all_points.end(), points.begin(), points.end());
            } catch (const std::exception& e) {
                std::cerr << "Error processing node: " << e.what() << std::endl;
            }
        }
    }

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    void find_vertex_counts(size_t start, size_t end) {
        for (size_t index = start; index < end; ++index) {
            const auto& xyz = std::get<0>(node_data_[index]);
            size_t patch_nr = std::get<1>(node_data_[index]);

            std::string tar_path = base_path_ + "/" + format_filename(xyz[0]) + "_" + format_filename(xyz[1]) + "_" + format_filename(xyz[2]) + ".tar";
            std::string ply_file_name = "surface_" + std::to_string(patch_nr) + ".ply";
            std::string ply_content;

            if (extract_ply_from_tar(tar_path, ply_file_name, ply_content)) {
                std::istringstream plyStream(ply_content);
                happly::PLYData plyData(plyStream);

                if (plyData.hasElement("vertex")) {
                    offset_per_node[index] = plyData.getElement("vertex").getProperty<double>("x").size();
                }
            }
            {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
            }
        }
    }

    size_t find_total_points() {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t total_nodes = node_data_.size();
        size_t chunk_size = std::ceil(static_cast<double>(total_nodes) / static_cast<double>(num_threads));
        
        // Set up progress tracking
        problem_size = total_nodes;
        progress = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::find_vertex_counts, this, start, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Reset progress
        if (verbose) {
            std::cout << std::endl;
        }
        problem_size = -1;
        progress = 0;

        // Calculate offsets and total points
        size_t total_points = 0;
        size_t total_points_temp = 0;
        for (size_t i = 0; i < total_nodes; ++i) {
            total_points_temp = total_points;
            total_points += offset_per_node[i];
            offset_per_node[i] = total_points_temp;
        }
        return total_points;
    }

    void load_all() {
        size_t total_nodes = node_data_.size();
        offset_per_node = std::make_unique<size_t[]>(total_nodes); // smart pointer
        if (verbose) {
            std::cout << "Loading all nodes..." << std::endl;
        }
        size_t total_points = find_total_points();
        all_points.reserve(total_points);
        if (verbose) {
            std::cout << "Total points: " << total_points << std::endl;
        }

        size_t num_threads = std::thread::hardware_concurrency(); // Number of threads
        std::vector<std::thread> threads;
        size_t chunk_size = std::ceil(static_cast<double>(total_nodes) / static_cast<double>(num_threads));

        // Set up progress tracking
        problem_size = total_nodes;
        progress = 0;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::process_node, this, start, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }
        if (verbose) {
            std::cout << std::endl;
            std::cout << "All nodes have been processed." << std::endl;
        }   
    }

    PointCloud get_results() {
        return PointCloud(std::move(all_points));  // Move the points instead of copying
    }

private:
    std::vector<std::tuple<std::vector<int>, int, double>> node_data_;
    // Preallocated NumPy arrays
    // py::array_t<float> points, normals, colors;
    std::vector<Point> all_points;
    std::unique_ptr<size_t[]> offset_per_node;
    std::string base_path_;
    mutable std::mutex mutex_;
    size_t progress = 0;
    size_t problem_size = -1;
    bool verbose;
};

class PointCloudProcessor {
public:
    explicit PointCloudProcessor(PointCloud& cloud, bool verbose) : cloud_(cloud), verbose(verbose) {}

    void deleteMarkedPoints() {
        std::cout << "Deleting marked points..." << std::endl;
        try {
            cloud_.pts.erase(std::remove_if(cloud_.pts.begin(), cloud_.pts.end(), [](const Point& p) {
                return p.marked_for_deletion;
            }), cloud_.pts.end());
            // cloud_.pts.shrink_to_fit(); // Shrink to fit after erasing marked points
        }
        catch (...) {
            std::cerr << "Error deleting marked points" << std::endl;
        }
    }

    void sortPointsWZYX() {
        std::sort(std::execution::par_unseq, cloud_.pts.begin(), cloud_.pts.end(), [](const Point& a, const Point& b) {
            if (a.w != b.w) return a.w < b.w;
            if (a.z != b.z) return a.z < b.z;
            if (a.y != b.y) return a.y < b.y;
            return a.x < b.x;
        });
    }

    void sortPointsXYZW() {
        std::sort(std::execution::par_unseq, cloud_.pts.begin(), cloud_.pts.end(), [](const Point& a, const Point& b) {
            if (a.x != b.x) return a.x < b.x;
            if (a.y != b.y) return a.y < b.y;
            if (a.z != b.z) return a.z < b.z;
            return a.w < b.w;
        });
    }

    void processDuplicates() {
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        size_t total_points = cloud_.size();
        size_t chunk_size = total_points / num_threads;

        std::vector<size_t> chunk_starts = getChunkStarts(num_threads, total_points, chunk_size);

        for (size_t i = 0; i < num_threads; ++i) {
            threads.emplace_back(&PointCloudProcessor::processDuplicatesThreaded, this, chunk_starts[i], (i + 1 < num_threads) ? chunk_starts[i + 1] : total_points);
        }

        for (auto& thread : threads) {
            thread.join();
        }
    }

    void filterPointsUsingKDTree(double spatial_threshold, double angle_threshold) {
        MyKDTree* index = nullptr;
        // Create a KD-tree for 3D points
        try {
            std::cout << "Building KD-tree..." << std::endl;
            index = new MyKDTree(3 /*dim*/, cloud_, nf::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        }
        catch (...) {
            std::cerr << "Error creating KD-tree" << std::endl;
            return;
        }
        if (index == nullptr) {
            std::cerr << "Error creating KD-tree, returning" << std::endl;
            return;
        }

        try {
            std ::cout << "Building index..." << std::endl;
            index->buildIndex();
        }
        catch (...) {
            std::cerr << "Error building KD-tree" << std::endl;
            return;
        }

        const size_t num_threads = std::thread::hardware_concurrency(); // Number of concurrent threads supported
        std::vector<std::thread> threads(num_threads);
        size_t part_length = cloud_.pts.size() / num_threads;
        progress = 0;
        problem_size = cloud_.pts.size() / 1000;

        std::cout << "Processing points using KD-tree..." << std::endl;
        for (size_t i = 0; i < num_threads; ++i) {
            try {
                size_t start = i * part_length;
                size_t end = (i == num_threads - 1) ? cloud_.pts.size() : start + part_length;
                threads[i] = std::thread(&PointCloudProcessor::processSubset, this, index, start, end, spatial_threshold, angle_threshold);
            }
            catch (...) {
                std::cerr << "Error processing subset" << std::endl;
            }
        }

        // Join all threads
        for (auto& thread : threads) {
            thread.join();
        }
        if (verbose) {
            std::cout << std::endl;
        }
        progress = 0;

        std :: cout << "Finished processing points using KD-tree" << std::endl;
        // Apply deletions
        deleteMarkedPoints();
    }

    PointCloud get_results() {
        return PointCloud(std::move(cloud_));  // Move the points instead of copying
    }

private:
    PointCloud& cloud_;
    size_t progress = 0;
    size_t problem_size = -1;
    bool verbose;
    std::mutex mutex_;

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    std::vector<size_t> getChunkStarts(size_t num_threads, size_t total_points, size_t chunk_size) {
        std::vector<size_t> chunk_starts(num_threads);
        size_t start = 0;
        for (size_t i = 0; i < num_threads; ++i) {
            chunk_starts[i] = start;
            size_t end = std::min(start + chunk_size, total_points);
            if (end < total_points) {
                // Advance end to the next change in xyz values
                while (end < total_points && cloud_.pts[end - 1].x == cloud_.pts[end].x && cloud_.pts[end - 1].y == cloud_.pts[end].y && cloud_.pts[end - 1].z == cloud_.pts[end].z) {
                    ++end;
                }
            }
            start = end;
        }
        return chunk_starts;
    }

    void processDuplicatesThreaded(size_t start, size_t end) {
        auto it = cloud_.pts.begin() + start;
        auto finish = cloud_.pts.begin() + end;

        while (it != finish) {
            auto next = it + 1;

            // Find the end of the current range of points with the same x, y, z
            while (next != finish && next->x == it->x && next->y == it->y && next->z == it->z) {
                ++next;
            }

            // Process all points in the range with the same x, y, z
            processRange(it, next);

            // Move iterator to the start of the next group of points
            it = next;
        }
    }

    void processRange(std::vector<Point>::iterator begin, std::vector<Point>::iterator end) {
        if (begin == end) return;

        // check if all points in the range have the same x y z values
        double x = begin->x;
        double y = begin->y;
        double z = begin->z;
        for (auto it = begin + 1; it != end; ++it) {
            if (it->x != x || it->y != y || it->z != z) {
                std::cout << "Error: Points in range do not have the same x y z values" << std::endl;
                std::cout << "Range: " << begin->x << " " << begin->y << " " << begin->z << " to " << (end - 1)->x << " " << (end - 1)->y << " " << (end - 1)->z << std::endl;
                return;
            }
        }

        size_t best_w_index_range = 0;
        double best_w = begin->w;  // Initialize best_w with the first w value in the range
        double sum_selected_w = 0;
        size_t count = 0;

        auto start_it = begin;
        auto end_it = begin;

        while (end_it != end) {
            sum_selected_w += end_it->w;
            ++count;
            double mean_iterative_w = sum_selected_w / count;

            // Adjust the start_it to maintain the constraint mean_iterative_w ± 90
            while (start_it != end_it && !(mean_iterative_w - 90 <= start_it->w && start_it->w <= mean_iterative_w + 90)) {
                sum_selected_w -= start_it->w;
                --count;
                ++start_it;
                mean_iterative_w = sum_selected_w / count;  // Recalculate mean after adjusting start
            }

            // Check and update the best_w if the current range is the largest
            if (end_it - start_it + 1 > best_w_index_range) {
                best_w_index_range = end_it - start_it + 1;
                best_w = mean_iterative_w;
            }

            ++end_it;  // Expand the end index to include more points
        }

        // Now mark all points for deletion except the best mean w value
        for (auto it = begin; it != end; ++it) {
            it->marked_for_deletion = true;  // Mark all for deletion
        }

        // Unmark the point that is part of the best range
        begin->marked_for_deletion = false;
        begin->w = best_w;  // Update the w value to the best_w
    }

    void processSubset(MyKDTree* index, size_t start, size_t end, double spatial_threshold, double angle_threshold) {
        for (size_t i = start; i < end; ++i) {
            if (i >= cloud_.pts.size()) {
                continue;
            }
            if (i < 0) {
                continue;
            }
            try {
                std::vector<nf::ResultItem<size_t, double>> ret_matches;
                nf::SearchParameters params;
                const double query_pt[3] = { cloud_.pts[i].x, cloud_.pts[i].y, cloud_.pts[i].z };

                // Perform the radius search
                const double radius = spatial_threshold * spatial_threshold;
                index->radiusSearch(&query_pt[0], radius, ret_matches, params);

                for (auto& match : ret_matches) {
                    if (match.first >= cloud_.pts.size()) {
                        continue;
                    }
                    if (match.first < 0) {
                        continue;
                    }
                    if (i != match.first && std::abs(cloud_.pts[i].w - cloud_.pts[match.first].w) > angle_threshold) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        cloud_.pts[i].marked_for_deletion = true;
                        cloud_.pts[match.first].marked_for_deletion = true;
                    }
                }
                {
                    if (i % 1000 == 0) {
                        std::lock_guard<std::mutex> lock(mutex_);
                        print_progress();
                    }
                }
            }
            catch (...) {
                // std::cerr << "Error processing point: " << e.what() << std::endl;
            }
        }
    }

    
};

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> to_array(const PointCloud& cloud) {
    // Create NumPy arrays for points, normals, and colors
    size_t total_points = cloud.size();
    py::array_t<float> points, normals, colors;
    points = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)4});
    normals = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)3});
    colors = py::array_t<float>(py::array::ShapeContainer{total_points, (size_t)3});

    auto pts = points.mutable_unchecked<2>();  // for direct access without bounds checking
    auto nrm = normals.mutable_unchecked<2>();
    auto clr = colors.mutable_unchecked<2>();

    // add the data to the numpy arrays
    for (size_t i = 0; i < total_points; ++i) {
        // add points x y z and winding angle to points
        pts(i, 0) = cloud.pts[i].x;
        pts(i, 1) = cloud.pts[i].y;
        pts(i, 2) = cloud.pts[i].z;
        pts(i, 3) = cloud.pts[i].w;

        // add normals nx ny nz to normals
        nrm(i, 0) = cloud.pts[i].nx;
        nrm(i, 1) = cloud.pts[i].ny;
        nrm(i, 2) = cloud.pts[i].nz;

        // add colors r g b to colors
        clr(i, 0) = static_cast<float>(cloud.pts[i].r) / 255.0;
        clr(i, 1) = static_cast<float>(cloud.pts[i].g) / 255.0;
        clr(i, 2) = static_cast<float>(cloud.pts[i].b) / 255.0;
    }

    return std::make_tuple(points, normals, colors);
}

py::array_t<bool> vector_to_array(std::vector<bool> selected_originals) {
    // Create NumPy arrays for points mask
    size_t total_points = selected_originals.size();
    py::array_t<bool> points_mask;
    points_mask = py::array_t<bool>(py::array::ShapeContainer{total_points});

    auto pts_mask = points_mask.mutable_unchecked<1>();  // for direct access without bounds checking

    // add the data to the numpy arrays
    for (size_t i = 0; i < total_points; ++i) {
        // add mask entry
        pts_mask(i) = selected_originals[i];
    }

    return points_mask;
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> load_pointclouds(const std::vector<std::tuple<std::vector<int>, int, double>>& nodes, const std::string& path, bool verbose = true) {
    PointCloudLoader loader(nodes, path, verbose);
    loader.load_all();
    PointCloud vector_points = loader.get_results();
    PointCloudProcessor processor(vector_points, verbose);
    processor.sortPointsXYZW();
    if (verbose) {
        std::cout << "Sorted points by XYZW" << std::endl;
    }
    processor.processDuplicates();
    if (verbose) {
        std::cout << "Processed duplicates" << std::endl;
    }
    processor.deleteMarkedPoints();
    if (verbose) {
        std::cout << "Deleted marked points" << std::endl;
    }
    processor.filterPointsUsingKDTree(2.0, 90.0);
    if (verbose) {
        std::cout << "Filtered points using KDTree" << std::endl;
    }
    processor.sortPointsWZYX();
    if (verbose) {
        std::cout << "Sorted points by WZYX" << std::endl;
    }
    PointCloud processed_points = processor.get_results();
    return to_array(std::move(processed_points));
}

py::array_t<bool> upsample_pointclouds(py::array_t<float> original_points, py::array_t<float> selected_subsamples, py::array_t<float> unselected_subsamples) {
    // Create KD-trees for selected and unselected points
    PointCloud selectedPC(selected_subsamples);
    PointCloud unselectedPC(unselected_subsamples);
    MyKDTree tree_selected(3 /*dim*/, selectedPC, {10 /* max leaf */});
    MyKDTree tree_unselected(3 /*dim*/, unselectedPC, {10 /* max leaf */});
    tree_selected.buildIndex();
    tree_unselected.buildIndex();

    std::vector<bool> selected_originals;

    // Assign each original point to selected or unselected based on closest distance
    auto original_points_r = original_points.unchecked<2>();
    std::cout << "Original points shape: " << original_points_r.shape(0) << std::endl;
    for (size_t i = 0; i < original_points.shape(0); ++i) {
        Point pt;
        pt.x = original_points_r(i, 0);
        pt.y = original_points_r(i, 1);
        pt.z = original_points_r(i, 2);
        pt.w = original_points_r(i, 3);
        double query_pt[3] = { pt.x, pt.y, pt.z };
        size_t closest_idx;
        double out_dist_sqr;

        tree_selected.knnSearch(&query_pt[0], 1, &closest_idx, &out_dist_sqr);
        double dist_selected = out_dist_sqr;
        double w_selected = selectedPC.pts[closest_idx].w;  // Assuming PointCloud and MyKDTree provide access to points

        tree_unselected.knnSearch(&query_pt[0], 1, &closest_idx, &out_dist_sqr);
        double dist_unselected = out_dist_sqr;
        double w_unselected = unselectedPC.pts[closest_idx].w;  // Assuming PointCloud and MyKDTree provide access to points

        if (dist_selected > 2.0 && dist_unselected > 2.0) {
            selected_originals.push_back(false);
        }
        else if (dist_selected <= dist_unselected) {
            if (std::abs(pt.w - w_selected) <= 90) {
                selected_originals.push_back(true);
            }
            else {
                selected_originals.push_back(false);
            }
        }
        else {
            if (std::abs(pt.w - w_unselected) <= 90) {
                selected_originals.push_back(true);
            }
            else {
                selected_originals.push_back(false);
            }
        }
    }
    std::cout << "Selected originals size: " << selected_originals.size() << std::endl;
    // Convert selected originals back to numpy array
    return vector_to_array(selected_originals);
}

std::vector<float> vector_subtract(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(3);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::minus<float>());
    return result;
}

std::vector<float> vector_scalar_multiply(const std::vector<float>& v, float scalar) {
    std::vector<float> result(3);
    std::transform(v.begin(), v.end(), result.begin(), [scalar](float x) { return x * scalar; });
    return result;
}

float vector_dot(const std::vector<float>& a, const std::vector<float>& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0);
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> result(3);
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<float>());
    return result;
}

float vector_norm(const std::vector<float>& v) {
    return std::sqrt(vector_dot(v, v));
}

// Calculates closest points and vector scalar t to a line in cylindrical coordinates
std::tuple<std::vector<float>, std::vector<float>, std::vector<std::vector<float>>> closestPointsAndDistancesCylindrical(const std::vector<std::vector<float>>& points,
                                      const std::vector<std::vector<float>>& normals,
                                      const std::vector<float>& linePoint, 
                                      const std::vector<float>& lineVector,
                                      float max_eucledian_distance = 20) {
    std::vector<float> distances;
    std::vector<float> ts;
    std::vector<std::vector<float>> closestnormals;

    float lineVectorNorm = vector_norm(lineVector);
    
    for (size_t i = 0; i < points.size(); ++i) {
        std::vector<float> point = points[i];
        std::vector<float> normal = normals[i];
        // skip computation of points that are too far away
        if (std::abs(point[1]-linePoint[1]) > max_eucledian_distance) {
            continue;
        }
        std::vector<float> pointsToLinePoint = vector_subtract(point, linePoint);
        float radiusXY = std::sqrt(std::pow(pointsToLinePoint[2], 2) + std::pow(pointsToLinePoint[0], 2));

        float t = vector_dot(pointsToLinePoint, lineVector) / vector_dot(lineVector, lineVector);
        float signT = (t > 0) ? 1 : -1; // Equivalent to np.sign in Python when handling t
        float tsValue = signT * radiusXY / lineVectorNorm;
        
        std::vector<float> closestPoint = vector_add(linePoint, vector_scalar_multiply(lineVector, tsValue));
        std::vector<float> displacement = vector_subtract(point, closestPoint);
        distances.push_back(vector_norm(displacement));
        ts.push_back(tsValue); // no idea why -1
        closestnormals.push_back(normal);
    }

    return std::make_tuple(distances, ts, closestnormals);
}

bool comp_lower_bound(const std::vector<float>& pt, float value) {
    return pt[3] < value;
}

std::pair<size_t, size_t> pointsAtWindingAngle(const std::vector<std::vector<float>>& points, float windingAngle, size_t last_start_index, size_t last_end_index, float maxAngleDiff = 30) {
    // Find the start and end indices of points within the maxAngleDiff range
    size_t startIndex = last_start_index;
    size_t endIndex = last_end_index;

    for (size_t i = startIndex; i < points.size(); ++i) {
        if (points[i][3] >= windingAngle - maxAngleDiff) {
            startIndex = i;
            break;
        }
    }

    if (endIndex < startIndex) {
        endIndex = startIndex;
    }
    for (size_t i = startIndex; i < points.size(); ++i) {
        if (points[i][3] > windingAngle + maxAngleDiff) {
            endIndex = i;
            break;
        }
    }

    return {startIndex, endIndex};
}

std::vector<float> umbilicus_xz_at_y(const std::vector<std::vector<float>>& points_array, float y_new) {
    // Resultant vector of interpolated points
    std::vector<float> interpolated_point = points_array[0];
    interpolated_point[1] = y_new;

    // Check if points_array is not empty
    if (points_array.empty()) {
        throw std::invalid_argument("The points array cannot be empty.");
    }

    // Linear interpolation function
    auto linear_interp = [](float x0, float x1, float y0, float y1, float y) {
        if (y0 == y1) {
            return x0;
        }
        return x0 + (x1 - x0) * (y - y0) / (y1 - y0);
    };

    if (points_array[0][1] >= y_new && points_array[0][1] >= y_new) {
        interpolated_point = points_array[0];
        interpolated_point[1] = y_new;
    }
    else if (points_array[points_array.size() - 1][1] <= y_new && points_array[points_array.size() - 1][1] <= y_new) {
        interpolated_point = points_array[points_array.size() - 1];
        interpolated_point[1] = y_new;
    }
    else {
        // Iterate over each segment in the points array
        for (size_t i = 0; i < points_array.size() - 1; ++i) {
            if ((points_array[i][1] <= y_new && points_array[i + 1][1] >= y_new) ||
                (points_array[i][1] >= y_new && points_array[i + 1][1] <= y_new)) {
                // Perform interpolation
                float x_new = linear_interp(points_array[i][0], points_array[i + 1][0], points_array[i][1], points_array[i + 1][1], y_new);
                float z_new = linear_interp(points_array[i][2], points_array[i + 1][2], points_array[i][1], points_array[i + 1][1], y_new);

                // Add the new point to the list of interpolated points
                interpolated_point = {x_new, y_new, z_new};
                break;
            }
        }
    }

    return interpolated_point;
}

std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>> calculatePointsAtAngle(
    const std::vector<std::vector<float>>& umbilicus_points, 
    const std::vector<std::vector<float>>& points, 
    const std::vector<std::vector<float>>& normals, 
    const std::vector<float>& z_positions, 
    float angle, 
    float max_eucledian_distance = 20) 
{
    float angle_radians = static_cast<float>(static_cast<int>(angle + static_cast<int>(2 - static_cast<int>(angle) / 360) * 360) % 360) * M_PI / 180.0;
    std::vector<float> angle_vector = { std::cos(angle_radians), 0.0, -std::sin(angle_radians) };
    
    size_t z_positions_length = z_positions.size();
    std::vector<std::vector<float>> ordered_pointset(z_positions_length);
    std::vector<std::vector<std::vector<float>>> ordered_normals(z_positions_length);
    std::vector<std::vector<float>> ordered_umbilicus_points(z_positions_length);


    for (size_t i = 0; i < z_positions_length; ++i) {
        std::vector<float> umbilicus_position = umbilicus_xz_at_y(umbilicus_points, z_positions[i]);
        ordered_umbilicus_points[i] = umbilicus_position;
        auto [distances, ts, normals_closest] = closestPointsAndDistancesCylindrical(points, normals, umbilicus_position, angle_vector, max_eucledian_distance);

        // sort distances and ts from smalles to largest distance
        std::vector<size_t> indices(distances.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&distances](size_t i1, size_t i2) { return distances[i1] < distances[i2]; });

        // reorder distances and ts
        std::vector<float> sorted_distances(distances.size());
        std::vector<float> sorted_ts(distances.size());
        std::vector<std::vector<float>> sorted_normals(distances.size());
        for (size_t j = 0; j < distances.size(); ++j) {
            sorted_distances[j] = distances[indices[j]];
            sorted_ts[j] = ts[indices[j]];
            sorted_normals[j] = normals_closest[indices[j]];
        }

        std::vector<float> valid_ts;
        std::vector<std::vector<float>> valid_normals;
        size_t max_number_closest_points = 40;
        size_t current_number_closest_points = 0;

        for (size_t j = 0; j < distances.size(); ++j) {
            if (sorted_distances[j] < max_eucledian_distance && sorted_ts[j] < 0) {
                valid_ts.push_back(sorted_ts[j]);
                valid_normals.push_back(sorted_normals[j]);
                current_number_closest_points++;
            }
            if (current_number_closest_points >= max_number_closest_points) {
                break;
            }
        }

        ordered_pointset[i] = valid_ts;
        ordered_normals[i] = valid_normals;
    }

    return std::make_tuple(ordered_pointset, ordered_normals, ordered_umbilicus_points, angle_vector);
}

std::tuple<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>, size_t, size_t> processWindingAngle(
    const std::vector<std::vector<float>>& umbilicus_points,
    const std::vector<std::vector<float>>& points, 
    const std::vector<std::vector<float>>& normals,
    const std::vector<float>& z_positions,
    float windingAngle,
    size_t last_start_index,
    size_t last_end_index,
    float maxEucledianDistance = 10) 
{
    std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>> result;

    // Find the start and end indices of points within the specified winding angle range
    auto [startIndex, endIndex] = pointsAtWindingAngle(points, windingAngle, last_start_index, last_end_index);

    // Extract the points and normals within the specified index range
    std::vector<std::vector<float>> extractedPoints(endIndex - startIndex);
    std::vector<std::vector<float>> extractedNormals(endIndex - startIndex);
    for (size_t i = startIndex; i < endIndex; ++i) {
        extractedPoints[i - startIndex] = points[i];
        extractedNormals[i - startIndex] = normals[i];
    }

    // Process the points at the specified angle
    result = calculatePointsAtAngle(umbilicus_points, extractedPoints, extractedNormals, z_positions, windingAngle, maxEucledianDistance);
    auto result_all = std::make_tuple(result, startIndex, endIndex);

    return result_all;
}

// Helper to determine min and max from vector of points
std::pair<float, float> findMinMaxWindingAngles(const std::vector<std::vector<float>>& points) {
    float minAngle = std::numeric_limits<float>::max();
    float maxAngle = std::numeric_limits<float>::lowest();
    for (const auto& point : points) {
        if (point[3] < minAngle) minAngle = point[3];
        if (point[3] > maxAngle) maxAngle = point[3];
    }
    return {minAngle, maxAngle};
}

// Helper to determine min and max from vector of points
std::pair<float, float> findMinMaxZ(const std::vector<std::vector<float>>& points) {
    float minZ = std::numeric_limits<float>::max();
    float maxZ = std::numeric_limits<float>::lowest();
    for (const auto& point : points) {
        if (point[1] < minZ) minZ = point[1];
        if (point[1] > maxZ) maxZ = point[1];
    }
    return {minZ, maxZ};
}

class RolledPointsetProcessor {
public:
    RolledPointsetProcessor(bool verbose)
        : verbose(verbose) {}

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> create_ordered_pointset_processor(
        py::array_t<float> original_points,
        py::array_t<float> original_normals,
        py::array_t<float> umbilicus_points,
        float angleStep = 6, int z_spacing = 10, float max_eucledian_distance = 10
        )
    {
        // Check the input dimensions and types are as expected
        if (original_points.ndim() != 2 || original_normals.ndim() != 2) {
            throw std::runtime_error("Expected two-dimensional array for points and normals.");
        }

        if (original_points.shape(1) != 4 || original_normals.shape(1) != 3) {
            throw std::runtime_error("Expected each point to have four and each normal to have three components.");
        }

        // Access the data
        auto points_buf = original_points.unchecked<2>(); // Accessing the data without bounds checking for performance
        auto normals_buf = original_normals.unchecked<2>();
        auto umbilicus_points_buf = umbilicus_points.unchecked<2>();

        // create a vector of vectors to hold the processed points
        std::vector<std::vector<float>> processed_points;
        processed_points.reserve(original_points.shape(0)); // reserve space for all points to improve performance

        // create a vector of vectors to hold the processed normals
        std::vector<std::vector<float>> processed_normals;
        processed_normals.reserve(original_normals.shape(0)); // reserve space for all normals to improve performance

        // create a vector of vectors to hold the umbilicus points
        std::vector<std::vector<float>> umbilicus_points_vector;
        umbilicus_points_vector.reserve(umbilicus_points.shape(0)); // reserve space for all umbilicus points to improve performance

        // Process points: just a placeholder for actual operations
        for (size_t i = 0; i < original_points.shape(0); ++i) {
            std::vector<float> point = {
                points_buf(i, 0), // x coordinate
                points_buf(i, 1), // y coordinate
                points_buf(i, 2),  // z coordinate
                points_buf(i, 3),  // winding angle
            };
            processed_points.push_back(point);

            std::vector<float> normal = {
                normals_buf(i, 0), // x component
                normals_buf(i, 1), // y component
                normals_buf(i, 2), // z component
            };
            processed_normals.push_back(normal);
        }

        // Process umbilicus points
        for (size_t i = 0; i < umbilicus_points.shape(0); ++i) {
            std::vector<float> umbilicus_point = {
                umbilicus_points_buf(i, 0), // x coordinate
                umbilicus_points_buf(i, 1), // y coordinate
                umbilicus_points_buf(i, 2),  // z coordinate
            };
            umbilicus_points_vector.push_back(umbilicus_point);
        }

        auto result = this->rolledOrderedPointset(umbilicus_points_vector, processed_points, processed_normals, static_cast<int>(std::thread::hardware_concurrency()), true, angleStep, z_spacing, max_eucledian_distance);

        return std::move(result);
    }

private:
    std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> rolledOrderedPointset(
            std::vector<std::vector<float>> umbilicus_points,
            std::vector<std::vector<float>> points,
            std::vector<std::vector<float>> normals,
            int numThreads, bool debug = false, float angleStep = 6, int z_spacing = 10, float max_eucledian_distance = 10
        ) 
    {
        auto [minWind, maxWind] = findMinMaxWindingAngles(points);
        auto [minZ, maxZ] = findMinMaxZ(points);
        if (verbose) {
            std::cout << "Number of threads: " << numThreads << " angle step: " << angleStep << " z spacing: " << z_spacing << " max eucledian distance: " << max_eucledian_distance << std::endl;
            std::cout << "Min and max winding angles: " << minWind << ", " << maxWind << std::endl;
            std::cout << "First point: " << points[0][0] << ", " << points[0][1] << ", " << points[0][2] << ", " << points[0][3] << " last point: " << points[points.size() - 1][0] << ", " << points[points.size() - 1][1] << ", " << points[points.size() - 1][2] << ", " << points[points.size() - 1][3] << std::endl;
        }

        // Calculate z positions based on spacing
        std::vector<float> zPositions;
        for (float z = minZ; z <= maxZ; z += z_spacing) {
            zPositions.push_back(z);
        }

        // Calculate total number of angles to process
        size_t totalAngles = std::ceil((maxWind - minWind) / angleStep);

        // Set up progress tracking
        problem_size = totalAngles;

        std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> results(totalAngles);

        std::vector<std::thread> threads;
        size_t anglesPerThread = totalAngles / numThreads;
        size_t anglesLeft = totalAngles % numThreads;
        size_t resultIndex = 0;
        float angleStart = minWind;
        // Launch threads
        for (size_t i = 0; i < numThreads; ++i) {
            size_t angles_this_thread = anglesPerThread + (i < anglesLeft ? 1 : 0);
            if (angles_this_thread == 0) {
                continue;
            }
            float angleEnd = angleStart + angles_this_thread * angleStep;
            if (i == numThreads - 1 || angleEnd > maxWind) {
                angleEnd = maxWind;
            }
            size_t startIndex = resultIndex;  // Assign the starting index for results for each thread
            threads.push_back(std::thread([this, &points, &normals, &umbilicus_points, &zPositions, angleStart, angleEnd, angleStep, max_eucledian_distance, &results, startIndex]() {
                this->workerFunction(points, normals, umbilicus_points, zPositions, angleStart, angleEnd, angleStep, max_eucledian_distance, results, startIndex);
            }));
            angleStart = angleEnd;
            resultIndex += angles_this_thread;  // Increment the start index for the next thread
        }

        // Join threads
        for (auto& thread : threads) {
            thread.join();
        }

        // Results are now populated in order in the 'results' vector
        return std::move(results);
    }

    void workerFunction(const std::vector<std::vector<float>>& points,
                    const std::vector<std::vector<float>>& normals,
                    const std::vector<std::vector<float>>& umbilicus_points,
                    const std::vector<float>& z_positions,
                    float angleStart,
                    float angleEnd,
                    float angleStep,
                    float maxEucledianDistance,
                    std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>>& results,
                    size_t startIndex
        )
    {
        // Find the start and end indices of points within the specified winding angle range
        auto [startAngleIndex, endAngleIndex] = pointsAtWindingAngle(points, angleStart + ((angleEnd - angleStart) / 2.0), 0, 0, (angleEnd - angleStart) / 2.0);

        size_t index = startIndex;
        size_t totalAngles = std::ceil((angleEnd - angleStart) / angleStep);
        for (float angle = angleStart; angle < angleEnd; angle += angleStep) {
            auto [result, last_angle_start_index_, last_angle_end_index_] = processWindingAngle(umbilicus_points, points, normals, z_positions, angle, startAngleIndex, endAngleIndex, maxEucledianDistance);
            startAngleIndex = last_angle_start_index_;
            endAngleIndex = last_angle_end_index_;
            // std::cout << index - startIndex + 1 << " / " << totalAngles << std::endl;
            results[index++] = result;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
            }
        }
    }

    mutable std::mutex mutex_;
    size_t progress = 0;
    size_t problem_size = -1;
    bool verbose;
};


std::vector<std::tuple<std::vector<std::vector<float>>, std::vector<std::vector<std::vector<float>>>, std::vector<std::vector<float>>, std::vector<float>>> create_ordered_pointset(
    py::array_t<float> original_points,
    py::array_t<float> original_normals,
    py::array_t<float> umbilicus_points,
    float angleStep, int z_spacing, float max_eucledian_distance, bool verbose
    )
{
    RolledPointsetProcessor processor(verbose);
    if (verbose) {
        std::cout << "Creating ordered pointset" << std::endl;
    }
    return std::move(processor.create_ordered_pointset_processor(original_points, original_normals, umbilicus_points, angleStep, z_spacing, max_eucledian_distance));
    if (verbose) {
        std::cout << "Finished creating ordered pointset" << std::endl;
    }
}

class OrderedPointsetOptimizer {
public:
    OrderedPointsetOptimizer(
        std::vector<std::vector<float>> input_ordered_pointset,
        std::vector<std::vector<bool>> fixed_points,
        std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices,
        float learning_rate = 0.1,
        size_t iterations = 3,
        float error_val_d = 0.01,
        float unfix_factor = 3.0,
        bool verbose = false)
        :   input_ordered_pointset(input_ordered_pointset), 
            new_interpolated_ts(input_ordered_pointset), 
            fixed_points(fixed_points), 
            neighbours_indices(neighbours_indices), 
            learning_rate(learning_rate),
            iterations(iterations),
            error_val_d(error_val_d),
            verbose(verbose) {}

    void print_progress() {
        if (!verbose) {
            return;
        }
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    std::vector<std::vector<float>> optimize_ordered_pointset_processor() {
        // Set up progress tracking
        problem_size = iterations;
        progress = 0;
        std::cout << "Optimizing ordered pointset" << std::endl;
        // Iterate over the number of iterations
        for (size_t iter = 0; iter < iterations; ++iter) {
            // Progress tracking
            {
                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();
            }

            // Calculate the total number of vertices and fixed points
            size_t nr_vertices = 0;
            size_t nr_fixed = 0;
            float last_error_val = std::numeric_limits<float>::max();

            // Calculate the total number of vertices and fixed points
            for (const auto& row : fixed_points) {
                nr_vertices += row.size();
                nr_fixed += std::count(row.begin(), row.end(), true);
            }
            size_t nr_floating = nr_vertices - nr_fixed;

            for (size_t opt_iter = 0; opt_iter < 10000; ++opt_iter) { // Maximum of 10000 optimization steps
                float error_val = compute_interpolated_adjacent();

                error_val /= nr_floating; // Normalize error by the number of floating vertices
                // std::cout << "Error value per floating vertex: " << std::setprecision(5) << error_val << std::endl;

                // Check for convergence or if the error increased
                if ((std::abs(last_error_val - error_val) < error_val_d) || (last_error_val - error_val < 0)) {
                    break;
                }
                last_error_val = error_val;
            }

            // Detect and unfix wrong fixed adjacent if necessary
            detect_and_unfix_wrong_fixed_adjacent();
        }
        // Finish progress tracking
        {
            if (verbose) {
                std::cout << std::endl;
            }
        }

        return new_interpolated_ts;
    }

private:
    std::pair<float, bool> get_front(size_t i, size_t j) {
        size_t f_i = neighbours_indices[i][j][0][0];
        size_t f_j = neighbours_indices[i][j][0][1];
        if (f_i == -1 || f_j == -1) {
            return {1, false};
        }
        float ts_f = input_ordered_pointset[f_i][f_j];
        bool fixed_f = fixed_points[f_i][f_j];
        return {ts_f, fixed_f};
    }

    std::pair<float, bool> get_back(size_t i, size_t j) {
        size_t b_i = neighbours_indices[i][j][1][0];
        size_t b_j = neighbours_indices[i][j][1][1];
        if (b_i == -1 || b_j == -1) {
            return {1, false};
        }
        float ts_b = input_ordered_pointset[b_i][b_j];
        bool fixed_b = fixed_points[b_i][b_j];
        return {ts_b, fixed_b};
    }

    std::pair<float, bool> get_top(size_t i, size_t j) {
        size_t t_i = neighbours_indices[i][j][2][0];
        size_t t_j = neighbours_indices[i][j][2][1];
        if (t_i == -1 || t_j == -1) {
            return {1, false};
        }
        float ts_t = input_ordered_pointset[t_i][t_j];
        bool fixed_t = fixed_points[t_i][t_j];
        return {ts_t, fixed_t};
    }

    std::pair<float, bool> get_bottom(size_t i, size_t j) {
        size_t bo_i = neighbours_indices[i][j][3][0];
        size_t bo_j = neighbours_indices[i][j][3][1];
        if (bo_i == -1 || bo_j == -1) {
            return {1, false};
        }
        float ts_bo = input_ordered_pointset[bo_i][bo_j];
        bool fixed_bo = fixed_points[bo_i][bo_j];
        return {ts_bo, fixed_bo};
    }

    std::pair<float, bool> get_left(size_t i, size_t j) {
        size_t l_i = neighbours_indices[i][j][4][0];
        size_t l_j = neighbours_indices[i][j][4][1];
        if (l_i == -1 || l_j == -1) {
            return {1, false};
        }
        float ts_l = input_ordered_pointset[l_i][l_j];
        bool fixed_l = fixed_points[l_i][l_j];
        return {ts_l, fixed_l};
    }

    std::pair<float, bool> get_right(size_t i, size_t j) {
        size_t r_i = neighbours_indices[i][j][5][0];
        size_t r_j = neighbours_indices[i][j][5][1];
        if (r_i == -1 || r_j == -1) {
            return {1, false};
        }
        float ts_r = input_ordered_pointset[r_i][r_j];
        bool fixed_r = fixed_points[r_i][r_j];
        return {ts_r, fixed_r};
    }

    std::pair<float, bool> get_left_new(size_t i, size_t j) {
        size_t l_i = neighbours_indices[i][j][4][0];
        size_t l_j = neighbours_indices[i][j][4][1];
        if (l_i == -1 || l_j == -1) {
            return {1, false};
        }
        float ts_l = new_interpolated_ts[l_i][l_j];
        bool fixed_l = fixed_points[l_i][l_j];
        return {ts_l, fixed_l};
    }

    std::pair<float, bool> get_right_new(size_t i, size_t j) {
        size_t r_i = neighbours_indices[i][j][5][0];
        size_t r_j = neighbours_indices[i][j][5][1];
        if (r_i == -1 || r_j == -1) {
            return {1, false};
        }
        float ts_r = new_interpolated_ts[r_i][r_j];
        bool fixed_r = fixed_points[r_i][r_j];
        return {ts_r, fixed_r};
    }

    float solve_for_t_individual(float r, float l, float m_r, bool valid_mr, float m_l, bool valid_ml, float m_ts, bool valid_mts, float a = 1.0) {
        float t_ts = m_ts;
        float t_total = t_ts;
        float count_total = a;
        if (r != 1 && valid_mr) {
            float t_r = r - m_r;
            t_total += t_r;
            count_total += 1.0;
        }
        if (l != 1 && valid_ml) {
            float t_l = l - m_l;
            t_total += t_l;
            count_total += 1.0;
        }
        t_total /= count_total;

        if (t_total > 0.0) {
            t_total = 0.0;
        }

        return t_total;
    }

    std::pair<bool, bool> side_of(float ts_, float n) {
        return {ts_ > n, ts_ == n};
    }

    float respect_non_overlapping(size_t i, size_t j, float new_ts_d) {
        float old_ts = input_ordered_pointset[i][j];

        auto [ts_l, fixed_l] = get_left(i, j);  // old left
        auto [ts_r, fixed_r] = get_right(i, j); // old right
        auto [ts_ln, fixed_ln] = get_left_new(i, j);  // new left
        auto [ts_rn, fixed_rn] = get_right_new(i, j); // new right

        // Check left boundary
        if (ts_l != 1) {
            auto [side_old_l, invalid_l] = side_of(old_ts, ts_l);
            assert(!invalid_l);
            if (!invalid_l) {
                auto [side_new_l, invalid_new_l] = side_of(old_ts + new_ts_d, ts_l);
                if (side_old_l != side_new_l) {
                    new_ts_d = (ts_l - old_ts) * 0.5;
                }
            }
        }

        // Check right boundary
        if (ts_r != 1) {
            auto [side_old_r, invalid_r] = side_of(old_ts, ts_r);
            assert(!invalid_r);
            if (!invalid_r) {
                auto [side_new_r, invalid_new_r] = side_of(old_ts + new_ts_d, ts_r);
                if (side_old_r != side_new_r) {
                    new_ts_d = (ts_r - old_ts) * 0.5;
                }
            }
        }

        // Check new left boundary
        if (ts_ln != 1) {
            auto [side_old_ln, invalid_ln] = side_of(old_ts, ts_ln);
            assert(!invalid_ln);
            if (!invalid_ln) {
                auto [side_new_ln, invalid_new_ln] = side_of(old_ts + new_ts_d, ts_ln);
                if (side_old_ln != side_new_ln) {
                    new_ts_d = (ts_ln - old_ts) * 0.5;
                }
            }
        }

        // Check new right boundary
        if (ts_rn != 1) {
            auto [side_old_rn, invalid_rn] = side_of(old_ts, ts_rn);
            assert(!invalid_rn);
            if (!invalid_rn) {
                auto [side_new_rn, invalid_new_rn] = side_of(old_ts + new_ts_d, ts_rn);
                if (side_old_rn != side_new_rn) {
                    new_ts_d = (ts_rn - old_ts) * 0.5;
                }
            }
        }

        return new_ts_d;
    }

    std::tuple<float, float, float, bool, float, bool, float, bool> calculate_neighbors_values(size_t i, size_t j) {
        std::pair<size_t, size_t> dict_key = {i, j};
        std::vector<std::vector<size_t>> same_sheet_neighbors;

        // Populate neighbors
        auto [f, fixed_f] = get_front(i, j);
        if (f != 1) {
            size_t f_i = neighbours_indices[i][j][0][0];
            size_t f_j = neighbours_indices[i][j][0][1];
            same_sheet_neighbors.push_back({f_i, f_j});
        }
        auto [b, fixed_b] = get_back(i, j);
        if (b != 1) {
            size_t b_i = neighbours_indices[i][j][1][0];
            size_t b_j = neighbours_indices[i][j][1][1];
            same_sheet_neighbors.push_back({b_i, b_j});
        }
        auto [t, fixed_t] = get_top(i, j);
        if (t != 1) {
            size_t t_i = neighbours_indices[i][j][2][0];
            size_t t_j = neighbours_indices[i][j][2][1];
            same_sheet_neighbors.push_back({t_i, t_j});
        }
        auto [bo, fixed_bo] = get_bottom(i, j);
        if (bo != 1) {
            size_t bo_i = neighbours_indices[i][j][3][0];
            size_t bo_j = neighbours_indices[i][j][3][1];
            same_sheet_neighbors.push_back({bo_i, bo_j});
        }

        float l = get_left(i, j).first;
        float r = get_right(i, j).first;
        float m_r = 0.0;
        float m_l = 0.0;
        float m_ts = 0.0;

        size_t count_r = 0, count_l = 0;

        // Assuming 'neighbors' includes the indices for the front-back and top-bottom neighbors
        for (auto& n : same_sheet_neighbors) {
            float ts_n = input_ordered_pointset[n[0]][n[1]];
            assert(ts_n <= 0.0);
            
            m_ts += ts_n;

            auto [l_n, fixed_l_n] = get_left(n[0], n[1]);
            if (l_n != 1) {
                m_l += (l_n - ts_n);
                count_l++;
            }
            auto [r_n, fixed_r_n] = get_right(n[0], n[1]);
            if (r_n != 1) {
                m_r += (r_n - ts_n);
                count_r++;
            }
            
        }

        m_ts = !same_sheet_neighbors.empty() ? m_ts / same_sheet_neighbors.size() : 0;
        bool valid_mts = same_sheet_neighbors.size() > 0;
        m_r = count_r > 0 ? m_r / count_r : 0;
        bool valid_mr = count_r > 0;
        m_l = count_l > 0 ? m_l / count_l : 0;
        bool valid_ml = count_l > 0;

        return {r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts};
    }

    void compute_section(int start, int end, float& thread_error_val) {
        for (int i = start; i < end; ++i) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); ++j) {
                if (!fixed_points[i][j]) {
                    auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                    float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                    assert(t <= 0.0);
                    float d_t = t - input_ordered_pointset[i][j];
                    d_t = respect_non_overlapping(i, j, d_t);
                    thread_error_val += std::abs(d_t);
                    new_interpolated_ts[i][j] = input_ordered_pointset[i][j] + learning_rate * d_t; // Apply learning rate
                }
            }
        }
    }

    float compute_interpolated_adjacent() {
        float error_val = 0.0;
        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        std::vector<float> errors(num_threads, 0.0);
        int n = input_ordered_pointset.size();
        int chunk_size = (n + num_threads - 1) / num_threads; // Calculate chunk size for each thread

        for (int i = 0; i < num_threads; ++i) {
            int start = i * chunk_size;
            int end = std::min(start + chunk_size, n);
            threads.emplace_back([this, start, end, &errors, i]() {
                this->compute_section(start, end, errors[i]);
            });
        }

        // Join threads and combine errors
        for (int i = 0; i < num_threads; ++i) {
            threads[i].join();
            error_val += errors[i];
        }

        // Copy the new interpolated ts to the input ordered pointset
        input_ordered_pointset = new_interpolated_ts;

        return error_val;
    }

    float compute_interpolated_adjacent_st() {
        float error_val = 0.0;
        new_interpolated_ts = input_ordered_pointset; // Copy the original ts

        // Calculate neighbour values for each vertex
        std::vector<std::vector<std::tuple<float, float, float, bool, float, bool, float, bool>>> neighbour_values;
        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            std::vector<std::tuple<float, float, float, bool, float, bool, float, bool>> row;
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                if (fixed_points[i][j]) { // optimization only for fixed points
                    row.push_back({1, 1, 0, false, 0, false, 0, false});
                    continue;
                }
                auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                row.push_back({r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts});
            }
            neighbour_values.push_back(row);
        }

        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                if (!fixed_points[i][j]) {
                    // Fetch the necessary neighbor values
                    auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = neighbour_values[i][j];

                    // Now solve for t_individual using dynamically computed m values
                    float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                    assert(t <= 0.0);
                    float d_t = t - input_ordered_pointset[i][j];
                    
                    // Adjust d_t respecting the non-overlapping constraints
                    d_t = respect_non_overlapping(i, j, d_t);
                    
                    error_val += std::abs(d_t);
                    new_interpolated_ts[i][j] = input_ordered_pointset[i][j] + learning_rate * d_t; // Apply learning rate
                }
            }
        }

        // Copy the new interpolated ts to the input ordered pointset
        input_ordered_pointset = new_interpolated_ts;

        return error_val;
    }

    std::vector<std::vector<float>> compute_interpolated_adjacent_errors() {
        std::vector<std::vector<float>> errors(input_ordered_pointset.size(), std::vector<float>(input_ordered_pointset[0].size(), 0.0f));

        for (size_t i = 0; i < input_ordered_pointset.size(); i++) {
            for (size_t j = 0; j < input_ordered_pointset[i].size(); j++) {
                auto [r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts] = calculate_neighbors_values(i, j);
                float t = solve_for_t_individual(r, l, m_r, valid_mr, m_l, valid_ml, m_ts, valid_mts, 1.0);
                assert(t <= 0.0);
                float d_t = t - input_ordered_pointset[i][j];
                d_t = respect_non_overlapping(i, j, d_t);
                errors[i][j] = std::abs(d_t);
            }
        }
        return errors;
    }

    void detect_and_unfix_wrong_fixed_adjacent() {
        auto errors = compute_interpolated_adjacent_errors();
        float sum_errors = 0.0;
        size_t count_fixed = 0;
        float sum_error_unfixed = 0.0;
        size_t count_unfixed = 0;

        // Calculate mean error of fixed points
        for (size_t i = 0; i < fixed_points.size(); i++) {
            for (size_t j = 0; j < fixed_points[i].size(); j++) {
                if (fixed_points[i][j]) {
                    sum_errors += errors[i][j];
                    count_fixed++;
                }
                else {
                    sum_error_unfixed += errors[i][j];
                    count_unfixed++;
                }
            }
        }

        float error_mean_fixed = count_fixed > 0 ? sum_errors / count_fixed : 0.0;
        float error_threshold = unfix_factor * error_mean_fixed;

        float error_mean_unfixed = count_unfixed > 0 ? sum_error_unfixed / count_unfixed : 0.0;
        float fixing_threshold = 0.2 * error_mean_unfixed;

        // Unfix points exceeding the error threshold
        for (size_t i = 0; i < fixed_points.size(); i++) {
            for (size_t j = 0; j < fixed_points[i].size(); j++) {
                if (fixed_points[i][j] && (errors[i][j] > error_threshold)) {
                    fixed_points[i][j] = false;
                }
                if (!fixed_points[i][j] && (errors[i][j] < fixing_threshold)) {
                    fixed_points[i][j] = true;
                }
            }
        }
    }

    std::vector<std::vector<float>> input_ordered_pointset;
    std::vector<std::vector<float>> new_interpolated_ts;
    std::vector<std::vector<bool>> fixed_points;
    std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices;
    float learning_rate = 0.1;
    size_t iterations = 3;
    mutable std::mutex mutex_;
    size_t progress = 0;
    size_t problem_size = -1;
    float error_val_d = 0.01;  // Delta for error value convergence
    float unfix_factor = 3.0;
    bool verbose;
};

std::vector<std::vector<float>> optimize_ordered_pointset(
    std::vector<std::vector<float>> input_ordered_pointset, 
    std::vector<std::vector<bool>> fixed_points, 
    std::vector<std::vector<std::vector<std::vector<int>>>> neighbours_indices,
    float learning_rate = 0.1,
    int iterations = 3,
    float error_val_d = 0.01,  // Delta for error value convergence
    float unfix_factor = 3.0,
    bool verbose = true
    )
{
    OrderedPointsetOptimizer optimizer(input_ordered_pointset, fixed_points, neighbours_indices, learning_rate, iterations, error_val_d, unfix_factor, verbose);
    if (verbose) {
        std::cout << "Optimizing ordered pointset" << std::endl;
    }
    return std::move(optimizer.optimize_ordered_pointset_processor());
}


PYBIND11_MODULE(pointcloud_processing, m) {
    m.doc() = "pybind11 module for parallel point cloud processing";

    m.def("load_pointclouds", &load_pointclouds, "Function to load point clouds and return points, normals, and colors.");

    m.def("upsample_pointclouds", &upsample_pointclouds, "Function to load point clouds and return points, normals, and colors.");

    m.def("create_ordered_pointset", &create_ordered_pointset, 
          "Function to create an ordered point set from a point cloud.",
          py::arg("original_points"),
          py::arg("original_normals"),
          py::arg("umbilicus_points"),
          py::arg("angleStep") = 6,
          py::arg("z_spacing") = 10,
          py::arg("max_eucledian_distance") = 10,
          py::arg("verbose") = true);

    m.def("optimize_adjacent", &optimize_ordered_pointset, 
            "Function to optimize an ordered pointset.",
            py::arg("input_ordered_pointset"),
            py::arg("fixed_points"),
            py::arg("neighbours_indices"),
            py::arg("learning_rate") = 0.1,
            py::arg("iterations") = 3,
            py::arg("error_val_d") = 0.01,
            py::arg("unfix_factor") = 3.0,
            py::arg("verbose") = true);
}
