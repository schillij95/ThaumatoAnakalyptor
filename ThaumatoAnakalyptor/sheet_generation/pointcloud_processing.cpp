#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <archive.h>
#include <archive_entry.h>

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
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
    double x, y, z, w;   // Coordinates and winding angle
    double nx, ny, nz;   // Normal vector components
    unsigned char r, g, b; // Color components
    bool marked_for_deletion = false;  // Flag to mark points for deletion
};

class PointCloud {
public:
    std::vector<Point> pts;

    // Default constructor
    PointCloud() = default;

    // Initialize with a vector of points
    explicit PointCloud(const std::vector<Point>& points) : pts(points) {}

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
    PointCloudLoader(const std::vector<std::tuple<std::vector<int>, int, double>>& node_data, const std::string& base_path)
        : node_data_(node_data), base_path_(base_path) {}

    bool extract_ply_from_tar(const std::string& tar_path, const std::string& ply_filename, std::string& out_ply_content) {
        struct archive *a;
        struct archive_entry *entry;
        int r;

        a = archive_read_new();
        archive_read_support_filter_all(a);
        archive_read_support_format_all(a);
        r = archive_read_open_filename(a, tar_path.c_str(), 10240); // Note: 10240 is the buffer size
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
            int numpy_offset = offset_per_node[index];
            auto& node = node_data_[index];
            const auto& xyz = std::get<0>(node);
            int patch_nr = std::get<1>(node);
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

                // print_progress();
                std::vector<Point> points;
                for (size_t i = 0; i < x.size(); ++i) {
                    Point point;
                    point.x = x[i];
                    point.y = y[i];
                    point.z = z[i];
                    point.w = winding_angle; // Assuming 'w' is the same for all points in this patch
                    point.nx = nx[i];
                    point.ny = ny[i];
                    point.nz = nz[i];
                    point.r = r[i];
                    point.g = g[i];
                    point.b = b[i];

                    points.push_back(point);
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
        progress++;
        // print on one line
        std::cout << "Progress: " << progress << "/" << problem_size << "\r";
        std::cout.flush();
    }

    void find_vertex_counts(size_t start, size_t end) {
        for (size_t index = start; index < end; ++index) {
            const auto& xyz = std::get<0>(node_data_[index]);
            int patch_nr = std::get<1>(node_data_[index]);

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
        size_t chunk_size = std::ceil(total_nodes / static_cast<double>(num_threads));

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::find_vertex_counts, this, start, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Reset progress
        std::cout << std::endl;
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
        all_points.reserve(total_nodes);
        problem_size = total_nodes;
        offset_per_node = new int[total_nodes];
        std::cout << "Loading all nodes..." << std::endl;
        long int total_points = find_total_points();
        std::cout << "Total points: " << total_points << std::endl;
        // points = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)4});
        // normals = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});
        // colors = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});

        size_t num_threads = std::thread::hardware_concurrency(); // Number of threads
        std::vector<std::thread> threads;
        size_t chunk_size = std::ceil(total_nodes / static_cast<double>(num_threads));

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * chunk_size;
            size_t end = std::min(start + chunk_size, total_nodes);
            threads.emplace_back(&PointCloudLoader::process_node, this, start, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }
        std::cout << std::endl;
        std::cout << "All nodes have been processed." << std::endl;
    }

    PointCloud get_results() const {
        return PointCloud(all_points);
    }

private:
    std::vector<std::tuple<std::vector<int>, int, double>> node_data_;
    // Preallocated NumPy arrays
    // py::array_t<float> points, normals, colors;
    std::vector<Point> all_points;
    int* offset_per_node;
    std::string base_path_;
    mutable std::mutex mutex_;
    int progress = 0;
    int problem_size = -1;
};

class PointCloudProcessor {
public:
    PointCloudProcessor(PointCloud& cloud) : cloud_(cloud) {}

    void deleteMarkedPoints() {
        cloud_.pts.erase(std::remove_if(cloud_.pts.begin(), cloud_.pts.end(), [](const Point& p) {
            return p.marked_for_deletion;
        }), cloud_.pts.end());
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
        // Create a KD-tree for 3D points
        MyKDTree index(3 /*dim*/, cloud_, nf::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        index.buildIndex();

        const size_t num_threads = std::thread::hardware_concurrency(); // Number of concurrent threads supported
        std::vector<std::thread> threads(num_threads);
        size_t part_length = cloud_.pts.size() / num_threads;
        progress = 0;
        problem_size = cloud_.pts.size() / 1000;

        for (size_t i = 0; i < num_threads; ++i) {
            size_t start = i * part_length;
            size_t end = (i == num_threads - 1) ? cloud_.pts.size() : start + part_length;
            threads[i] = std::thread(&PointCloudProcessor::processSubset, this, std::ref(index), start, end, spatial_threshold, angle_threshold);
        }

        // Join all threads
        for (auto& thread : threads) {
            thread.join();
        }
        std::cout << std::endl;
        progress = 0;

        // Apply deletions
        deleteMarkedPoints();
    }

    // void filterPointsUsingKDTree(double spatial_threshold, double angle_threshold) {
    //     // Create a KD-tree for 3D points
    //     MyKDTree index(3 /*dim*/, cloud_, nf::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    //     index.buildIndex();

    //     for (size_t i = 0; i < cloud_.pts.size(); ++i) {
    //         // One line progress
    //         std::cout << "Progress: " << i << "/" << cloud_.pts.size() << "\r";
    //         std::vector<nf::ResultItem<size_t, double>> ret_matches;
    //         nf::SearchParameters params;
    //         const double query_pt[3] = { cloud_.pts[i].x, cloud_.pts[i].y, cloud_.pts[i].z };

    //         // Radius search
    //         const double radius = spatial_threshold * spatial_threshold;
    //         index.radiusSearch(&query_pt[0], radius, ret_matches, params);

    //         for (auto& match : ret_matches) {
    //             if (i != match.first && std::abs(cloud_.pts[i].w - cloud_.pts[match.first].w) > angle_threshold) {
    //                 cloud_.pts[i].marked_for_deletion = true;
    //                 cloud_.pts[match.first].marked_for_deletion = true;
    //             }
    //         }
    //     }
    //     std::cout << std::endl;

    //     // Apply deletions
    //     deleteMarkedPoints();
    // }

    PointCloud get_results() const {
        return cloud_;
    }

private:
    PointCloud& cloud_;
    int progress = 0;
    int problem_size = -1;
    std::mutex mutex_;

    void print_progress() {
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

        int best_w_index_range = 0;
        double best_w = begin->w;  // Initialize best_w with the first w value in the range
        double sum_selected_w = 0;
        int count = 0;

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

    void processSubset(MyKDTree& index, size_t start, size_t end, double spatial_threshold, double angle_threshold) {
        for (size_t i = start; i < end; ++i) {
            std::vector<nf::ResultItem<size_t, double>> ret_matches;
            nf::SearchParameters params;
            const double query_pt[3] = { cloud_.pts[i].x, cloud_.pts[i].y, cloud_.pts[i].z };

            // Perform the radius search
            const double radius = spatial_threshold * spatial_threshold;
            index.radiusSearch(&query_pt[0], radius, ret_matches, params);

            for (auto& match : ret_matches) {
                if (i != match.first && std::abs(cloud_.pts[i].w - cloud_.pts[match.first].w) > angle_threshold) {
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
    }
};

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> to_array(PointCloud& cloud) {
    // Create NumPy arrays for points, normals, and colors
    long int total_points = cloud.size();
    py::array_t<float> points, normals, colors;
    points = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)4});
    normals = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});
    colors = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});

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

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> load_pointclouds(const std::vector<std::tuple<std::vector<int>, int, double>>& nodes, const std::string& path) {
    PointCloudLoader loader(nodes, path);
    loader.load_all();
    PointCloud vector_points = loader.get_results();
    PointCloudProcessor processor(vector_points);
    processor.sortPointsXYZW();
    std::cout << "Sorted points by XYZW" << std::endl;
    processor.processDuplicates();
    std::cout << "Processed duplicates" << std::endl;
    processor.deleteMarkedPoints();
    std::cout << "Deleted marked points" << std::endl;
    processor.filterPointsUsingKDTree(2.0, 90.0);
    PointCloud processed_points = processor.get_results();
    std::cout << "Filtered points using KDTree" << std::endl;
    return to_array(processed_points);
}

PYBIND11_MODULE(pointcloud_processing, m) {
    m.doc() = "pybind11 module for parallel point cloud processing";

    m.def("load_pointclouds", &load_pointclouds, "Function to load point clouds and return points, normals, and colors.");
}
