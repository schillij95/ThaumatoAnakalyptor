import numpy as np
# test
import unittest
import build.sheet_generation as sg
import random

def bfs_ks(edges_indices, valid_mask):
        # Use BFS to traverse the graph and compute the ks
        if start_node is None:
            start_node = list(graph.nodes)[0]
        visited = {start_node: True}
        queue = [start_node]
        ks = {start_node: 0}
        while queue:
            node = queue.pop(0)
            node_k = ks[node]
            for edge in graph.nodes[node]['edges']:
                if edge[0] == node:
                    other_node = edge[1]
                else:
                    other_node = edge[0]
                if other_node in visited:
                    # Assert for correct k
                    k = graph.get_edge_k(node, other_node)
                    assert ks[other_node] == node_k + k, f"Invalid k: {ks[other_node]} != {node_k + k}"
                    continue
                visited[other_node] = True
                k = graph.get_edge_k(node, other_node)
                ks[other_node] = node_k + k
                queue.append(other_node)

        nodes = [node for node in visited]
        ks = np.array([ks[node] for node in nodes]) # to numpy
        ks = ks - np.min(ks) # 0 to max

        return nodes, ks

class TestGraphK(unittest.TestCase):
    def random_graph(self):
        runs = 100
        nr_nodes = random.randint(4, 100)
        nr_edges = random.randint(5, 500)
        edges = np.random.randint(0, nr_nodes, size=(nr_edges, 2))
        k = np.random.randint(-1, 2, size=(nr_edges))
        certainty = np.random.randint(1, 1000, size=(nr_edges))
        edges_indices  = np.concatenate((edges, k.reshape(-1, 1), certainty.reshape(-1, 1)), axis=1)
        initial_component = np.zeros((0,2), dtype=np.int32)
        for i in range(runs):
            individual = np.random.randint(0, 1000, size=(nr_edges))
            valid_mask, valid_edges_count = sg.build_graph_from_individual_cpp(int(individual.shape[0]), individual, int(edges_indices.shape[0]), edges_indices, 1.0, 2.5, int(initial_component.shape[0]), initial_component, True)

















class TestVectorProjection(unittest.TestCase):
    def test_vector_projection(self):
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([0.0, 1.0, 0.0])
        self.assertTrue(torch.allclose(vector_projection(v1, v2), torch.tensor([0.0, 0.0, 0.0])))
        
    def test_vector_projection2(self):
        v1 = torch.tensor([1.0, 0.0, 0.0])
        v2 = torch.tensor([1.0, 0.0, 0.0])
        self.assertTrue(torch.allclose(vector_projection(v1, v2), torch.tensor([1.0, 0.0, 0.0])))
        
if __name__ == '__main__':
    unittest.main()