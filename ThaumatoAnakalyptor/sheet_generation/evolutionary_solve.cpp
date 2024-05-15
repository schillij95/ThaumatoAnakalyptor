#include "evolutionary_solve_utils.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// int main() {
//     int population_size = 500;
//     int generations = 20000;
//     int nr_nodes = 100;
//     int nr_edges = (nr_nodes * (nr_nodes - 1)) / 2;
//     int* graph = new int[4 * nr_edges];
//     int gene_length = nr_edges;
//     int legth_initial_component = 0;
//     int* initial_component = new int[0];

//     double factor_0 = 1.0;
//     double factor_not_0 = 1.0;

//     // Construct fully connected graph
//     int index = 0;
//     for (int i = 0; i < nr_nodes; ++i) {
//         for (int j = i + 1; j < nr_nodes; ++j) {
//             graph[4 * index] = i; // Node i
//             graph[4 * index + 1] = j; // Node j
//             graph[4 * index + 2] = (rand() % 3) - 1; // k
//             graph[4 * index + 3] = 1; // Certainty
//             index++;
//         }
//     }

//     auto solution = evolution_solve_k_assignment(population_size, generations, nr_edges, graph, factor_0, factor_not_0, legth_initial_component, initial_component);

//     return 0;
// }


// Input is; Nodes Length, Initial DP (shape: (nodes_length, nodes_length, 64), type: bool)
std::tuple<double, py::array_t<int>, py::array_t<float>> evolution_solve_k_assignment_init(
    int population_size,
    int generations,
    int length_edges,
    py::array_t<int> edges,
    py::array_t<bool> same_block,
    int length_bad_edges,
    py::array_t<int> bad_edges,
    double factor_0,
    double factor_not_0,
    double factor_bad,
    int legth_initial_component,
    py::array_t<int> initial_component,
    bool use_ignoring = true
    )
{
    if (use_ignoring) {
        std::cout << "Using edge-ignoring to evolve a solution." << std::endl;
    }
    else {
        std::cout << "Not using edge-ignoring to evolve a solution." << std::endl;
    }
    // Directly use the pointer to the data in the edges array
    int* edges_cpp = static_cast<int*>(edges.request().ptr);
    bool* same_block_cpp = static_cast<bool*>(same_block.request().ptr);
    int* bad_edges_cpp = static_cast<int*>(bad_edges.request().ptr);

    // Directly use the pointer to the data in the initial_component array
    int* initial_component_cpp = static_cast<int*>(initial_component.request().ptr);

    auto res = evolution_solve_k_assignment(population_size, generations, length_edges, edges_cpp, same_block_cpp, length_bad_edges, bad_edges_cpp, factor_0, factor_not_0, factor_bad, legth_initial_component, initial_component_cpp, use_ignoring);

    double valid_edges_count = std::get<0>(res);
    int* valid_edges = std::get<1>(res);
    float* edge_weights = std::get<2>(res);

    // Convert the valid_edges_python to a NumPy array 1D
    py::array_t<int64_t> valid_edges_python(py::array::ShapeContainer({static_cast<long int>(length_edges)}));
    auto valid_edges_res = valid_edges_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (size_t i = 0; i < length_edges; i++) {
        valid_edges_res(i) = valid_edges[i];
    }

    // Convert the edge_weights to a NumPy array 1D
    py::array_t<float> edge_weights_python(py::array::ShapeContainer({static_cast<long int>(length_edges)}));
    auto edge_weights_res = edge_weights_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (size_t i = 0; i < length_edges; i++) {
        edge_weights_res(i) = edge_weights[i];
    }

    return {valid_edges_count, valid_edges_python, edge_weights_python};
}

// Input is; Nodes Length, Initial DP (shape: (nodes_length, nodes_length, 64), type: bool)
std::tuple<double, py::array_t<int>, py::array_t<float>> evolution_solve_patches_init(
    int population_size,
    int generations,
    int length_edges,
    py::array_t<int> edges,
    py::array_t<bool> same_block,   
    int length_bad_edges,
    py::array_t<int> bad_edges,
    double factor_0,
    double factor_not_0,
    double factor_bad
    )
{
    // Directly use the pointer to the data in the edges array
    int* edges_cpp = static_cast<int*>(edges.request().ptr);
    bool* same_block_cpp = static_cast<bool*>(same_block.request().ptr);
    int* bad_edges_cpp = static_cast<int*>(bad_edges.request().ptr);

    auto res = evolution_solve_patches(population_size, generations, length_edges, edges_cpp, same_block_cpp, length_bad_edges, bad_edges_cpp, factor_0, factor_not_0, factor_bad);

    double valid_edges_count = std::get<0>(res);
    int* valid_edges = std::get<1>(res);
    float* edge_weights = std::get<2>(res);

    // Convert the valid_edges_python to a NumPy array 1D
    py::array_t<int64_t> valid_edges_python(py::array::ShapeContainer({static_cast<long int>(length_edges)}));
    auto valid_edges_res = valid_edges_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (size_t i = 0; i < length_edges; i++) {
        valid_edges_res(i) = valid_edges[i];
    }

    // Convert the edge_weights to a NumPy array 1D
    py::array_t<float> edge_weights_python(py::array::ShapeContainer({static_cast<long int>(length_edges)}));
    auto edge_weights_res = edge_weights_python.mutable_unchecked<1>(); // Now correctly a 3D array

    for (size_t i = 0; i < length_edges; i++) {
        edge_weights_res(i) = edge_weights[i];
    }

    return {valid_edges_count, valid_edges_python, edge_weights_python};
}

PYBIND11_MODULE(evolve_graph, m) {
    m.doc() = "pybind11 graph evolution solver for ThaumatoAnakalyptor"; // Module docstring

    m.def("evolution_solve_k_assignment", &evolution_solve_k_assignment_init, "Function to build graph from individual in C++");

    m.def("evolution_solve_patches", &evolution_solve_patches_init, "Function to build patches graph from individual in C++");
}
