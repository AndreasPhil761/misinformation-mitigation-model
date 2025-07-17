import networkx as nx
import matplotlib.pyplot as plt

def create_graph(n_users, W_users, W_rec):

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    G = nx.DiGraph()
    for i in range(n_users):
        G.add_node(i, label=f'{i + 1}')
    for i in range(n_users):
        for j in range(n_users):
            if W_users[i, j] > 0:
                G.add_edge(j, i, weight=round(W_users[i, j], 2))
    G.add_node(n_users, label='R')
    for i in range(n_users):
        if W_rec[i] > 0:
            G.add_edge(n_users, i, weight=round(W_rec[i, 0], 2), color='k')
    pos = nx.spring_layout(G, k=0.9, iterations=50)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    nx.draw_networkx_nodes(G, pos, node_color='white', node_size=300, edgecolors='k', ax=ax)
    nx.draw_networkx_labels(G, pos, {i: G.nodes[i]['label'] for i in G.nodes()}, font_size=8, ax=ax)
    edge_colors = ['k' if G[u][v].get('color') == 'k' else 'k' for u, v in G.edges()]

    # Add edge labels with weights
    edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        horizontalalignment='left',
        verticalalignment='top',
        clip_on=False 
    )
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, arrowsize=10,
                           connectionstyle="arc3,rad=0.1", ax=ax)
    ax.set_title("Adjacency Graph", fontsize=10, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('adjacency_graph.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.show()