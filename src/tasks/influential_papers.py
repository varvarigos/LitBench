"""
Influential Papers Task

This module provides functionality to identify the most influential papers in a citation graph.

Functions:
    influential_papers(K, graph):
        Given an integer K and a citation graph, returns the K most influential papers based on the number of citations.
        The function returns the title and abstract of each of the K most influential papers in a formatted string.

Usage:
    The script reads configuration from a YAML file, loads a citation graph from a GEXF file, and prints the K most influential papers.
"""

def influential_papers(message, graph):
    K = int(message.split("Number of papers to consider: ")[1])
    
    in_degree = dict(graph.in_degree())
    sorted_in_degree = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)

    most_cited_papers = []
    for i in range(K):
        node = sorted_in_degree[i]
        paper = graph.nodes[node[0]]
        most_cited_papers.append(paper)

    resp = "Here are the most influential papers:\n"
    for i, paper in enumerate(most_cited_papers):
        resp += f"{i+1}. Title: {paper['title']}\nAbstract: {paper['abstract']}\n"
    return resp
