import networkx as nx
import subprocess
import os
import pickle
import argparse
from tqdm import tqdm

class Graph:
    def __init__(self):
        self.edges = {}  # Stores edges with update matrices and certainty factors
        self.nodes = {}  # Stores node beliefs and fixed status

class ScrollGraph(Graph):
    def __init__(self, overlapp_threshold, umbilicus_path):
        super().__init__()
        self.umbilicus_path = umbilicus_path

def load_thaumato_graph(graph_path):
    with open(graph_path, 'rb') as file:
        thaumato_graph = pickle.load(file)
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
    return graph_dict

def transpose_graph_dict(graph_dict):
    transpose_graph_dict = {}
    for u in graph_dict:
        for v in graph_dict[u]:
            if v not in transpose_graph_dict:
                transpose_graph_dict[v] = []
            transpose_graph_dict[v].append(u)
    return transpose_graph_dict
            
def create_edge_list(graph_dict, output_file):
    G = nx.DiGraph(graph_dict)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for u, v in G.edges():
            f.write(f"{u}\t{v}\n")

def sort_edgelist(edge_list_file, basename):
    # Sort the edge list and remove duplicates
    cmd = f'sort -nk 1 {edge_list_file} | uniq > {basename}'
    subprocess.run(cmd, shell=True, check=True)

def create_webgraph(edge_list_file, basename, lib_dir):
    # Use the WebGraph BVGraph class to create the .graph, .properties, and .offsets files
    cmd = [
        'java', '-cp', f"{lib_dir}/*",
        'it.unimi.dsi.webgraph.BVGraph', '-1', '-g',
        'ArcListASCIIGraph', 'dummy', basename
    ]
    with open(edge_list_file, 'r') as f:
        subprocess.run(cmd, stdin=f)

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser(description='Convert a ThaumatoAnakalyptor graph to a cyclic, directed WebGraph format')
    parser.add_argument('graph', type=str, help='Path to the thaumato graph file')
    parser.add_argument('output_file', type=str, help='Output file name')
    args = parser.parse_args()

    # Load the graph
    graph_dict = load_thaumato_graph(args.graph)
    
    edge_list_file = os.path.join(os.path.dirname(args.output_file), 'graph.edgelist')
    sorted_edge_list_file = os.path.join(os.path.dirname(args.output_file), 'sorted_graph.edgelist')
    basename = args.output_file  # Basename for the WebGraph files
    lib_dir = 'lib'  # Path to the directory containing the WebGraph .jar files

    create_edge_list(graph_dict, edge_list_file)
    sort_edgelist(edge_list_file, sorted_edge_list_file)
    create_webgraph(sorted_edge_list_file, basename, lib_dir)

    transpose_edge_list_file = os.path.join(os.path.dirname(args.output_file), 'transpose_graph.edgelist')
    transpose_sorted_edge_list_file = os.path.join(os.path.dirname(args.output_file), 'transpose_sorted_graph.edgelist')
    transpose_basename = basename + '-t'

    transpose_graph_dict = transpose_graph_dict(graph_dict)
    create_edge_list(transpose_graph_dict, transpose_edge_list_file)
    sort_edgelist(transpose_edge_list_file, transpose_sorted_edge_list_file)
    create_webgraph(transpose_sorted_edge_list_file, transpose_basename, lib_dir)

    print(f"WebGraph files created with basename: {basename}")
