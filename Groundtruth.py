import math
import numpy as np
import scipy as sp
import scipy.sparse
import networkit as nk
import random
import sys
import time
import gc
from tqdm import tqdm

def GlobalPageRankWithNetworkit(graph_path, output_path):
    node_set = set()
    with open(graph_path, 'r') as file:
        for line in tqdm(file, desc="Reading edges for node_set"):
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            data = line.split()
            from_node = int(data[0])
            to_node = int(data[1])
            node_set.add(from_node)
            node_set.add(to_node)

    node_dict = {element: index for index, element in enumerate(node_set)}     
    node_dict_t = {index: element for element, index in node_dict.items()} 

    G = nk.Graph(directed=True, n=len(node_dict))
    with open(graph_path, 'r') as file:
        for line in tqdm(file, desc="Reading edges for graph build"):
            line = line.strip()
            if not line or line.startswith('%') or line.startswith('#'):
                continue
            data = line.split()
            from_node = int(data[0])
            to_node = int(data[1])
            from_node_index = node_dict[from_node]
            to_node_index = node_dict[to_node]
            G.addEdge(from_node_index, to_node_index)

    start_time = time.time()
    pr = nk.centrality.PageRank(G, tol=1 / len(node_dict) / 100, damp=0.85)
    pr.run()
    pagerank_scores = pr.scores()
    execution_time = time.time() - start_time
    print(f"execution_time：{execution_time:.4f} s")
    with open(output_path, 'w+') as file:
        for node, pagerank in enumerate(pagerank_scores):
            file.write(f"{node_dict_t[node]}\t{pagerank:.17f}\n")

args = sys.argv
graph_path = args[1]
output_path = args[2]
GlobalPageRankWithNetworkit(graph_path, output_path)
