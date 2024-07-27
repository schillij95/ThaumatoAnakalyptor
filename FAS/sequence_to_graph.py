import networkx as nx
import subprocess
import os
import pickle
import argparse
from tqdm import tqdm
import numpy as np

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

    def remove_nodes_edges(self, nodes):
        """
        Remove nodes and their edges from the graph.
        """
        for node in tqdm(nodes, desc="Removing nodes"):
            node = tuple(int(node[i]) for i in range(4))
            # Delete Node Edges
            node_edges = list(self.nodes[node]['edges'])
            for edge in node_edges:
                node_ = edge[0] if edge[0] != node else edge[1]
                if node_ in self.nodes:
                    self.nodes[node_]['edges'].remove(edge)
                # Delete Edges
                if edge in self.edges:
                    del self.edges[edge]
            # Delete Node
            del self.nodes[node]
        # set length of nodes and edges
        self.nodes_length = len(self.nodes)
        self.edges_length = len(self.edges)

    def compute_node_edges(self, verbose=True):
        """
        Compute the edges for each node and store them in the node dictionary.
        """
        if verbose:
            print("Computing node edges...")
        for node in tqdm(self.nodes, desc="Adding nodes") if verbose else self.nodes:
            self.nodes[node]['edges'] = []
        for edge in tqdm(self.edges, desc="Computing node edges") if verbose else self.edges:
            for k in self.edges[edge]:
                self.nodes[edge[0]]['edges'].append(edge)
                self.nodes[edge[1]]['edges'].append(edge)
    
    def update_winding_angles(self, nodes, ks, update_winding_angles=False):
        nodes = [tuple(int(node[i]) for i in range(4)) for node in nodes]
        nodes_ks_dict = {}
        ks_min = np.min(ks)
        ks = np.array(ks) - ks_min
        for i, node in enumerate(nodes):
            nodes_ks_dict[node] = ks[i]
        # Update winding angles
        for node in nodes_ks_dict:
            k = nodes_ks_dict[node]
            node = tuple(int(node[i]) for i in range(4))
            self.nodes[node]['assigned_k'] = k
            if update_winding_angles:
                self.nodes[node]['winding_angle'] = - k*360 + self.nodes[node]['winding_angle']

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()
        self.umbilicus_path = umbilicus_path

    def largest_connected_component(self, delete_nodes=True, min_size=None):
        print("Finding largest connected component...")        
        # walk the graph from the start node
        visited = set()
        # tqdm object showing process of visited nodes untill all nodes are visited
        tqdm_object = tqdm(total=len(self.nodes))
        components = []
        starting_index = 0
        def build_nodes(edges):
            nodes = set()
            for edge in edges:
                nodes.add(edge[0])
                nodes.add(edge[1])
            return list(nodes)
        nodes = build_nodes(self.edges)
        max_component_length = 0
        nodes_length = len(nodes)
        nodes_remaining = nodes_length
        print(f"Number of active nodes: {nodes_length}, number of total nodes: {len(self.nodes)}")
        while True:
            start_node = None
            # Pick first available unvisited node
            for node_idx in range(starting_index, nodes_length):
                node = nodes[node_idx]
                if node not in visited:
                    start_node = node
                    break
            if start_node is None:
                break    

            queue = [start_node]
            component = set()
            while queue:
                node = queue.pop(0)
                component.add(node)
                if node not in visited:
                    tqdm_object.update(1)
                    visited.add(node)
                    edges = self.nodes[node]['edges']
                    for edge in edges:
                        queue.append(edge[0] if edge[0] != node else edge[1])

            components.append(component)
            # update component length tracking
            component_length = len(component)
            max_component_length = max(max_component_length, component_length)
            nodes_remaining -= component_length
            if (min_size is None) and (nodes_remaining < max_component_length): # early stopping, already found the largest component
                print(f"breaking in early stopping")
                print()
                break

        nodes_total = len(self.nodes)
        edges_total = len(self.edges)
        print(f"number of nodes: {nodes_total}, number of edges: {edges_total}")

        # largest component
        largest_component = list(max(components, key=len))
        if min_size is not None:
            components = [component for component in components if len(component) >= min_size]
            for component in components:
                largest_component.extend(list(component))
        largest_component = set(largest_component)
        
        # Remove all other Nodes, Edges and Node Edges
        if delete_nodes:
            other_nodes = [node for node in self.nodes if node not in largest_component]
            self.remove_nodes_edges(other_nodes)
        
        print(f"Pruned {nodes_total - len(self.nodes)} nodes. Of {nodes_total} nodes.")
        print(f"Pruned {edges_total - len(self.edges)} edges. Of {edges_total} edges.")

        result = list(largest_component)
        print(f"Found largest connected component with {len(result)} nodes.")
        return result
    
    def bfs_ks(self, start_node):
        """
        Breadth-first search from start_node.
        """
        nodes_k_dict = {}
        nodes = []
        ks = []
        min_k = None
        max_k = None
        queue = [(start_node, 0)]
        tqdm_object = tqdm(total=100*len(self.nodes))
        while queue:
            node, k = queue.pop(0)
            if min_k is None or k < min_k:
                min_k = k
            if max_k is None or k > max_k:
                max_k = k
            # tqdm_object.update(1)
            print(len(nodes_k_dict), min_k, max_k)
            if node not in nodes_k_dict or (nodes_k_dict[node] <= 0 and k < nodes_k_dict[node] or (nodes_k_dict[node] >= 0 and k > nodes_k_dict[node])):
                nodes_k_dict[node] = k
                edges = self.nodes[node]['edges']
                for edge in edges:
                    flip_edge = edge[0] != node
                    try:
                        for k_delta in self.edges[edge]:
                            if self.edges[edge][k_delta]['bad_edge']:
                                continue
                            if self.edges[edge][k_delta]['same_block']:
                                continue
                            k_delta_flip = -k_delta if flip_edge else k_delta
                            queue.append((edge[0] if flip_edge else edge[1], k + k_delta_flip))
                    except:
                        continue
        tqdm_object.close()
        for node in nodes_k_dict:
            k = nodes_k_dict[node]
            nodes.append(node)
            ks.append(k)
        return nodes, ks

def load_thaumato_graph(graph_path):
    with open(graph_path, 'rb') as file:
        thaumato_graph = pickle.load(file)
    thaumato_graph.remove_nodes_edges([list(thaumato_graph.nodes)[-1]]) # indexing error somewhere, last node is missing after ArrayFAS
    # Convert to "standart graph format". nodes numbered from 0 to n-1, edges are stored in a dictionary
    graph_dict = {}
    translation_dict = {}
    for i, node in enumerate(tqdm(thaumato_graph.nodes)):
        graph_dict[i] = []
        translation_dict[node] = i
    added_edges = 0
    not_added = 0
    for edge in tqdm(thaumato_graph.edges, desc="Converting edges"):
        winding_angle0 = thaumato_graph.nodes[edge[0]]['winding_angle']
        winding_angle1 = thaumato_graph.nodes[edge[1]]['winding_angle']
        i0 = translation_dict[edge[0]]
        i1 = translation_dict[edge[1]]
        for k in thaumato_graph.edges[edge]:
            try:
                if thaumato_graph.edges[edge][k]['bad_edge']:
                    continue
                if k < 0:
                    graph_dict[i1].append(i0)
                elif k == 0:
                    if winding_angle0 < winding_angle1:
                        graph_dict[i0].append(i1)
                    else:
                        graph_dict[i1].append(i0)
                else:
                    graph_dict[i0].append(i1)
                added_edges += 1
            except:
                not_added += 1
    print(f"Added {added_edges} edges, not added {not_added} edges")
    return thaumato_graph, graph_dict, translation_dict

def load_sequence(sequence_path):
    # one integer per line
    with open(sequence_path, 'r') as file:
        sequence = [int(line.strip()) for line in file]
    print(f"Loaded sequence of length {len(sequence)}")
    sequence_dict = {}
    for i, node in enumerate(sequence):
        sequence_dict[node] = i
    return sequence, sequence_dict

def calculate_ks(sequence, sequence_dict, translation_dict, thaumato_graph):
    visited_nodes = [False for _ in range(len(sequence))]
    ks = [0 for _ in range(len(sequence))]
    nodes_names = list(thaumato_graph.nodes)

    nr_fresh_starts = 0
    for i in range(len(sequence)):
        if not visited_nodes[i]:
            nr_fresh_starts += 1
            visited_nodes[i] = True
            print(f"Fresh start at {i}, sum of visited nodes: {sum(visited_nodes)}")
            for i, node in enumerate(sequence):
                if not visited_nodes[i]:
                    continue
                try:
                    node_name = nodes_names[node]
                except:
                    print(f"Node not in graph {node}")
                node_graph = thaumato_graph.nodes[node_name]
                edges = node_graph['edges']
                for edge in edges:
                    flip_edge = edge[0] != node_name
                    next_node_name = edge[0] if flip_edge else edge[1]
                    next_node = translation_dict[next_node_name]
                    next_node_sequence_position = sequence_dict[next_node]
                    if next_node_sequence_position < i:
                        continue
                    for k in thaumato_graph.edges[edge]:
                        if thaumato_graph.edges[edge][k]['bad_edge']:
                            continue
                        if thaumato_graph.edges[edge][k]['same_block']:
                            continue
                        k_flip = -k if flip_edge else k
                        if k_flip < 0:
                            print(f"Negative k: {k_flip}")
                        else:
                            ks[next_node_sequence_position] = ks[i] + k_flip
                            visited_nodes[next_node_sequence_position] = True
    print(f"{nr_fresh_starts} fresh starts to Calculated ks, min: {min(ks)}, max: {max(ks)}, nr of visited nodes: {sum(visited_nodes)}")
    return ks
                

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser(description='Convert a ThaumatoAnakalyptor graph to a cyclic, directed WebGraph format')
    parser.add_argument('graph', type=str, help='Path to the thaumato graph file')
    parser.add_argument('graph_name', type=str, help='Path to the sequence file')
    args = parser.parse_args()

    # Load the graph
    thaumato_graph, graph_dict, translation_dict = load_thaumato_graph(args.graph)
    # Load the sequence
    sequence, sequence_dict = load_sequence(args.graph_name+"_sequence.txt")
    calculate_ks(sequence, sequence_dict, translation_dict, thaumato_graph)
    exit()
    # Delete Bad nodes
    edges = thaumato_graph.edges
    deletion_edges = []
    for edge in edges:
        i0 = translation_dict[edge[0]]
        i1 = translation_dict[edge[1]]
        try:
            seq_i0 = sequence_dict[i0]
            seq_i1 = sequence_dict[i1]
        except:
            print(f"Node not in sequence {i0} or {i1}")
            deletion_edges.append(edge)
            continue
        winding_angle0 = thaumato_graph.nodes[edge[0]]['winding_angle']
        winding_angle1 = thaumato_graph.nodes[edge[1]]['winding_angle']
        edge_ = edges[edge]
        deletion_ks = []
        for k in edge_:
            if edges[edge][k]['bad_edge']:
                deletion_ks.append(k)
                continue
            if edges[edge][k]['same_block']:
                deletion_ks.append(k)
                continue
            if k < 0 and seq_i1 >= seq_i0:
                deletion_ks.append(k)
            elif k == 0:
                if winding_angle0 <= winding_angle1 and seq_i0 >= seq_i1:
                    deletion_ks.append(k)
                elif winding_angle1 <= winding_angle0 and seq_i1 >= seq_i0:
                    deletion_ks.append(k)
            elif k > 0 and seq_i0 >= seq_i1:
                deletion_ks.append(k)
        for k in deletion_ks:
            del thaumato_graph.edges[edge][k]
            if len(thaumato_graph.edges[edge]) == 0:
                print(f"Deleting edge {edge}")
                deletion_edges.append(edge)
    for edge in deletion_edges:
        del thaumato_graph.edges[edge]
        for n in edge:
            if n in thaumato_graph.nodes:
                thaumato_graph.nodes[n]['edges'].remove(edge)
    
    print(f"Deleted bad edges")
    print(f"Remaining edges: {len(thaumato_graph.edges)}")
    
    thaumato_graph.compute_node_edges()

    # Prune the graph
    thaumato_graph.largest_connected_component(delete_nodes=True)
    thaumato_graph.compute_node_edges()
    print(f"Remaining nodes: {len(thaumato_graph.nodes)}")


    # Update winding angles
    start_node = None
    seq_start_node = len(thaumato_graph.nodes)
    for node in thaumato_graph.nodes:
        i0 = translation_dict[tuple(node)]
        seq_i0 = sequence_dict[i0]
        if start_node is None or seq_i0 < seq_start_node:
            start_node = tuple(node)
            seq_start_node = seq_i0

    nodes, ks = thaumato_graph.bfs_ks(start_node, translation_dict, sequence_dict)
    thaumato_graph.update_winding_angles(nodes, ks, update_winding_angles=True)

    # Save the graph
    with open(args.graph_name+".pkl", 'wb') as file:
        pickle.dump(thaumato_graph, file)
