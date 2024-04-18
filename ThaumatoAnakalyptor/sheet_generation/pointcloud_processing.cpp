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
#include <thread>
#include <iomanip>
#include <cstdio>    // For mkstemp
#include <unistd.h>  // For unlink
#include <filesystem>
#include "happly.h"

namespace fs = std::filesystem;
namespace py = pybind11;

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

                auto pts = points.mutable_unchecked<2>();  // for direct access without bounds checking
                auto nrm = normals.mutable_unchecked<2>();
                auto clr = colors.mutable_unchecked<2>();

                // add the data to the numpy arrays
                for (size_t i = 0; i < std::get<0>(std::get<0>(vertices)).size(); ++i) {
                    // add points x y z and winding angle to points
                    pts(numpy_offset + i, 0) = x[i];
                    pts(numpy_offset + i, 1) = y[i];
                    pts(numpy_offset + i, 2) = z[i];
                    pts(numpy_offset + i, 3) = winding_angle;

                    // add normals nx ny nz to normals
                    nrm(numpy_offset + i, 0) = nx[i];
                    nrm(numpy_offset + i, 1) = ny[i];
                    nrm(numpy_offset + i, 2) = nz[i];

                    // add colors r g b to colors
                    clr(numpy_offset + i, 0) = static_cast<float>(r[i]) / 255.0;
                    clr(numpy_offset + i, 1) = static_cast<float>(g[i]) / 255.0;
                    clr(numpy_offset + i, 2) = static_cast<float>(b[i]) / 255.0;
                }

                std::lock_guard<std::mutex> lock(mutex_);
                print_progress();

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
        problem_size = total_nodes;
        offset_per_node = new int[total_nodes];
        std::cout << "Loading all nodes..." << std::endl;
        long int total_points = find_total_points();
        std::cout << "Total points: " << total_points << std::endl;
        points = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)4});
        normals = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});
        colors = py::array_t<double>(py::array::ShapeContainer{total_points, (long int)3});

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

    std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>> get_results() const {
        return std::make_tuple(points, normals, colors);
    }

private:
    std::vector<std::tuple<std::vector<int>, int, double>> node_data_;
    // Preallocated NumPy arrays
    py::array_t<float> points, normals, colors;
    int* offset_per_node;
    std::string base_path_;
    mutable std::mutex mutex_;
    int progress = 0;
    int problem_size = -1;
};

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>> load_pointclouds(const std::vector<std::tuple<std::vector<int>, int, double>>& nodes, const std::string& path) {
    PointCloudLoader loader(nodes, path);
    loader.load_all();
    return loader.get_results();
}

PYBIND11_MODULE(pointcloud_processing, m) {
    m.doc() = "pybind11 module for parallel point cloud processing";

    m.def("load_pointclouds", &load_pointclouds, "Function to load point clouds and return points, normals, and colors.");
}
