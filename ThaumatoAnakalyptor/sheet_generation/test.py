import numpy as np
# test
import unittest
import sys
sys.path.append('build')
import sheet_generation as sg
import random

def check_valid(component, node1, node2, k):
    k_node1 = component[node1]
    k_node2 = component[node2]
    return k_node1 + k == k_node2

def merge_components(component1, component2, node1, node2, k):
    k1_node1 = component1[node1]
    k2_node2 = component2[node2]
    k_diff = k1_node1 + k - k2_node2
    for component2_key in component2.keys():
        component1[component2_key] = component2[component2_key] + k_diff

def add_node_to_component(component, node1, node2, k):
    k_node1 = component[node1]
    k_node2 = k_node1 + k
    component[node2] = k_node2

def build_graph_from_individual(individual, graph_raw, factor_0, factor_not_0, return_valid_mask=False, initial_component=None):
    # Build a valid graph based on the weights for each edge represented by the individual
    # Sort the edges based on the weights
    sorted_edges_indices = np.argsort(individual)
    # Initialize the graph
    graph_components = []
    if initial_component is not None:
        graph_components.append(initial_component)
    valid_edges_count = 0

    if return_valid_mask:
        valid_mask = np.ones(len(sorted_edges_indices), dtype=bool)

    for edge_index in sorted_edges_indices:
        edge = graph_raw[edge_index]
        node1, node2, k, certainty = edge
        # Check if the nodes are already in the graph
        node1_component, node2_component = None, None
        for i, component in enumerate(graph_components):
            if node1 in component:
                node1_component = (component, i)
            if node2 in component:
                node2_component = (component, i)
            if node1_component is not None and node2_component is not None:
                break
        # Calculate first part of fitness, subject to change in checks
        k_factor = factor_0 if k == 0 else factor_not_0

        score_edge = k_factor * certainty
        valid_edges_count += score_edge
        # If both nodes are not in the graph, add a new component
        if node1_component is None and node2_component is None:
            new_component = {node1: 0, node2: k}
            graph_components.append(new_component)
        # If one node is in a component, add the other node to the component
        elif (node1_component is not None) and (node2_component is None):
            add_node_to_component(node1_component[0], node1, node2, k)
        elif (node2_component is not None) and (node1_component is None):
            add_node_to_component(node2_component[0], node2, node1, -k)
        # If both nodes are in the same component, check if the edge is valid
        elif node1_component[1] == node2_component[1]:
            if not check_valid(node1_component[0], node1, node2, k):
                valid_edges_count -= score_edge # Remove the bad edge from the positive score
                if return_valid_mask:
                    valid_mask[edge_index] = False
        # If both nodes are in different components, merge the components
        elif node1_component is not None and node2_component is not None:
            merge_components(node1_component[0], node2_component[0], node1, node2, k)
            # Remove the second component
            graph_components.pop(node2_component[1])
        # Error case
        else:
            print("Error: This case should not happen")
            raise ValueError("Invalid graph")
        
    if return_valid_mask:
        return valid_mask, valid_edges_count
    else:
        return valid_edges_count
    
def bfs_ks(edges_indices, valid_mask_int):
    valid_mask = valid_mask_int > 0

    edges = {}
    valid_edges = edges_indices[valid_mask]
    for edge in valid_edges:
        if edge[0] not in edges:
            edges[edge[0]] = set()
        edges[edge[0]].add((edge[0], edge[1], edge[2], edge[3]))
        if edge[1] not in edges:
            edges[edge[1]] = set()
        edges[edge[1]].add((edge[0], edge[1], edge[2], edge[3]))
    
    # Use BFS to traverse the graph and compute the ks
    start_node = valid_edges[0, 0]
    visited = {start_node: True}
    queue = [start_node]
    ks = {start_node: 0}

    while queue:
        node = queue.pop(0)
        node_k = ks[node]
        for edge in edges[node]:
            node1, node2, k, certainty = edge
            if node1 == node:
                other_node = node2
            else:
                other_node = node1
                k = -k # flip k if edge direction is flipped
            if other_node in visited:
                # Assert for correct k
                condition= ks[other_node] == node_k + k
                # assert condition, f"Invalid k: {ks[other_node]} != {node_k + k}, edges_indices: {edges_indices}, valid_mask: {valid_mask}, valid_mask_int: {valid_mask_int}"
                if not condition:
                    return False
                continue
            visited[other_node] = True
            ks[other_node] = node_k + k
            queue.append(other_node)
    return True

class TestGraphK(unittest.TestCase):
    def test_random_graph(self):
        runs = 10000
        nr_nodes = random.randint(3, 100)
        # nr_nodes = 3
        nr_edges = random.randint(1, 300)
        # nr_edges = 3
        edges = np.random.randint(0, nr_nodes, size=(nr_edges, 2))
        k = np.random.randint(-1, 2, size=(nr_edges))
        certainty = np.random.randint(1, 1000, size=(nr_edges))
        edges_indices  = np.concatenate((edges, k.reshape(-1, 1), certainty.reshape(-1, 1)), axis=1).astype(np.int32)
        initial_component = np.zeros((0,2), dtype=np.int32)
        for i in range(runs):
            individual = np.random.randint(0, 1000, size=(nr_edges)).astype(np.int32)
            valid_mask, valid_edges_count = sg.build_graph_from_individual_cpp(int(individual.shape[0]), individual, int(edges_indices.shape[0]), edges_indices, 1.0, 2.5, int(initial_component.shape[0]), initial_component, True)
            self.assertTrue(bfs_ks(edges_indices, valid_mask))

    def test_fully_connected_graph(self):
        runs = 10000
        nr_nodes = random.randint(3, 100)
        # build edges
        edges = []
        k = []
        certainty = []
        for i in range(nr_nodes):
            for j in range(i+1, nr_nodes):
                edges.append([i, j])
                k.append(random.randint(-1, 2))
                certainty.append(random.randint(1, 1000))
        edges = np.array(edges)
        k = np.array(k)
        certainty = np.array(certainty)
        edges_indices  = np.concatenate((edges, k.reshape(-1, 1), certainty.reshape(-1, 1)), axis=1).astype(np.int32)
        initial_component = np.zeros((0,2), dtype=np.int32)
        for i in range(runs):
            individual = np.random.randint(0, 1000, size=(edges.shape[0])).astype(np.int32)
            valid_mask, valid_edges_count = sg.build_graph_from_individual_cpp(int(individual.shape[0]), individual, int(edges_indices.shape[0]), edges_indices, 1.0, 2.5, int(initial_component.shape[0]), initial_component, True)
            self.assertTrue(bfs_ks(edges_indices, valid_mask))

class TestGraphKPython(unittest.TestCase):
    def test_random_graph_python(self):
        runs = 10000
        nr_nodes = random.randint(3, 100)
        # nr_nodes = 3
        nr_edges = random.randint(1, 300)
        # nr_edges = 3
        edges = np.random.randint(0, nr_nodes, size=(nr_edges, 2))
        k = np.random.randint(-1, 2, size=(nr_edges))
        certainty = np.random.randint(1, 1000, size=(nr_edges))
        edges_indices  = np.concatenate((edges, k.reshape(-1, 1), certainty.reshape(-1, 1)), axis=1).astype(np.int32)
        initial_component = np.zeros((0,2), dtype=np.int32)
        for i in range(runs):
            individual = np.random.randint(0, 1000, size=(nr_edges)).astype(np.int32)
            valid_mask, valid_edges_count = build_graph_from_individual(individual, edges_indices, 1.0, 2.5, initial_component=initial_component, return_valid_mask=True)
            self.assertTrue(bfs_ks(edges_indices, valid_mask))

    def test_fully_connected_graph_python(self):
        runs = 10000
        nr_nodes = random.randint(3, 100)
        # build edges
        edges = []
        k = []
        certainty = []
        for i in range(nr_nodes):
            for j in range(i+1, nr_nodes):
                edges.append([i, j])
                k.append(random.randint(-1, 2))
                certainty.append(random.randint(1, 1000))
        edges = np.array(edges)
        k = np.array(k)
        certainty = np.array(certainty)
        edges_indices  = np.concatenate((edges, k.reshape(-1, 1), certainty.reshape(-1, 1)), axis=1).astype(np.int32)
        initial_component = np.zeros((0,2), dtype=np.int32)
        for i in range(runs):
            individual = np.random.randint(0, 1000, size=(edges.shape[0])).astype(np.int32)
            valid_mask, valid_edges_count = build_graph_from_individual(individual, edges_indices, 1.0, 2.5, initial_component=initial_component, return_valid_mask=True)
            # print(f"Shape: {valid_mask.shape}, Shape individual: {individual.shape}, Shape edges_indices: {edges_indices.shape}")
            self.assertTrue(bfs_ks(edges_indices, valid_mask))
        
if __name__ == '__main__':
    unittest.main()