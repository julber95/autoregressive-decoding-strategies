import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

os.makedirs('utils', exist_ok=True)

def visualize_greedy_vs_topk(probabilities, k_values=[5, 10, 20], save_path="utils/greedy_vs_topk.png"):
    """
    Visualizes the comparison between Greedy and Top-k sampling strategies.
    
    Args:
        probabilities: Tensor of probabilities
        k_values: Values of k for Top-k sampling
        save_path: Path to save the figure
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = probabilities
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    ax = axs[0]
    greedy_colors = ['red' if i == 0 else 'lightgray' for i in range(min(30, len(sorted_probs)))]
    ax.bar(range(min(30, len(sorted_probs))), sorted_probs[:30], color=greedy_colors)
    ax.set_title("Greedy Search Strategy")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Token Rank")
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 0.9, "Selects only the\nmost probable token", 
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    ax = axs[1]
    
    for i, k in enumerate(k_values):
        color = plt.cm.Blues(0.5 + 0.5*i/len(k_values))
        
        mask = np.zeros_like(sorted_probs[:30])
        mask[:k] = sorted_probs[:k]
        
        ax.bar(range(min(30, len(sorted_probs))), mask, 
               color=color, alpha=0.6,
               label=f'Top-k (k={k})')

        ax.axvline(x=k-0.5, color=color, linestyle='--', label=f'k={k} cutoff')
    
    ax.bar(range(min(30, len(sorted_probs))), sorted_probs[:30], 
           color='lightgray', alpha=0.3, zorder=-1)
            
    ax.set_title("Top-k Sampling Strategy")
    ax.set_xlabel("Token Rank")
    ax.set_ylabel("Probability")
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0.5, 0.9, "Samples from the k\nmost probable tokens", 
            transform=ax.transAxes, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Figure saved: {save_path}")
    plt.close()

def visualize_beam_search_tree(model, dataset, prompt="The President of the United States", 
                              beam_width=3, max_steps=3, save_path="utils/beam_tree.png"):
    """
    Visualizes beam search as a tree diagram for the given prompt.
    
    Args:
        model: The language model
        dataset: The dataset containing the dictionary for encoding/decoding
        prompt: Starting text prompt
        beam_width: Width of the beam
        max_steps: Maximum number of steps to show
        save_path: Path to save the visualization
    """
    G = nx.DiGraph()
    start_node = "START"
    G.add_node(start_node, level=0)
    
    words = prompt.strip().split()
    tokens = [dataset.dictionary.word2idx.get(w, dataset.dictionary.word2idx['<UNK>']) 
             for w in words]
    
    device = next(model.parameters()).device
    input_tensor = torch.LongTensor([tokens]).to(device)
    
    beams = [(start_node, tokens, 0.0)]
    
    node_sequences = {start_node: tokens}
    
    for step in range(max_steps):
        new_beams = []
        for parent_node, beam_tokens, beam_score in beams:
            with torch.no_grad():
                input_ids = torch.LongTensor([beam_tokens]).to(device)
                outputs, _ = model(input_ids)
                next_token_logits = outputs[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                topk_probs, topk_indices = torch.topk(next_token_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = topk_indices[i].item()
                    token_prob = topk_probs[i].item()
                    new_word = dataset.dictionary.idx2word[token_id]
                    if new_word in ['<EOS>', '<PAD>', '<UNK>']:
                        continue
                    node_id = f"{step+1}_{i}_{new_word}"
            
                    G.add_node(node_id, level=step+1, label=new_word)
                    G.add_edge(parent_node, node_id, 
                              probability=f"{token_prob:.3f}",
                              weight=token_prob)
                    
                    new_score = beam_score - np.log(token_prob)
                    new_tokens = beam_tokens + [token_id]
                    new_beams.append((node_id, new_tokens, new_score))
                    node_sequences[node_id] = new_tokens
    
        new_beams.sort(key=lambda x: x[2])
        beams = new_beams[:beam_width]

    if beams:
        best_beam_node = beams[0][0]
        best_score = beams[0][2]
    else:
        best_beam_node = None

    pos = {}
    nodes_by_level = {}

    for node, data in G.nodes(data=True):
        level = data['level']
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)

    max_level = max(nodes_by_level.keys()) if nodes_by_level else 0
    for level, nodes in nodes_by_level.items():
        n_nodes = len(nodes)
        for i, node in enumerate(sorted(nodes)):
            pos[node] = (level, (i - (n_nodes-1)/2))

    plt.figure(figsize=(15, 10))
    
    best_path = []
    if best_beam_node:
        current = best_beam_node
        while current != start_node:
            best_path.append(current)
            for pred in G.predecessors(current):
                current = pred
                break
        best_path.append(start_node)
        best_path.reverse()
    
    best_path_edges = []
    regular_edges = []
    
    for u, v in G.edges():
        if u in best_path and v in best_path:
            best_path_edges.append((u, v))
        else:
            regular_edges.append((u, v))
    
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        level = G.nodes[node]['level']
        if node in best_path:
            node_colors.append('lightgreen')
        else:
            node_colors.append(plt.cm.Blues(0.5 + 0.5 * level / max(max_level, 1)))
        
        if level == max_level:
            node_sizes.append(3000)
        else:
            node_sizes.append(2500)
    
    labels = {}
    for node in G.nodes():
        if node == start_node:
            labels[node] = "START"
        else:
            labels[node] = G.nodes[node].get('label', node)
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_weight='bold')
    
    nx.draw_networkx_edges(G, pos, edgelist=regular_edges,
                          arrows=True, arrowsize=15, arrowstyle='-|>')
    
    nx.draw_networkx_edges(G, pos, edgelist=best_path_edges, 
                          arrows=True, arrowsize=20, arrowstyle='-|>',
                          width=2.5, edge_color='green')

    edge_labels = {(u, v): G.edges[u, v]['probability'] for u, v in G.edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)
    
    plt.plot([0], [0], color='green', linewidth=2.5, label='Best path')
    plt.legend(loc='upper left')
    
    plt.title(f"Beam Search Tree for \"{prompt}\" (width={beam_width}, steps={max_steps})\nFinal Score: {best_score:.4f}")
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300)
    print(f"Beam search tree visualization saved: {save_path}")
    plt.close()

def visualize_nucleus_sampling(probabilities, p=0.4, save_path="utils/nucleus_sampling_p04.png"):
    """
    Visualizes Nucleus (Top-p) sampling with p=0.4
    
    Args:
        probabilities: Tensor of probabilities
        p: Value of p for Nucleus sampling
        save_path: Path to save the visualization
    """
    if isinstance(probabilities, torch.Tensor):
        probs = probabilities.detach().cpu().numpy()
    else:
        probs = probabilities
    
    sorted_indices = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_indices]

    cumulative_probs = np.cumsum(sorted_probs)
    
    p_cutoff = np.searchsorted(cumulative_probs, p) + 1
    
    plt.figure(figsize=(12, 9))
    
    plt.subplot(2, 1, 1)
    bars = plt.bar(range(min(30, len(sorted_probs))), 
                  sorted_probs[:30], 
                  color=['green' if i < p_cutoff else 'lightgray' for i in range(30)])
    
    plt.axvline(x=p_cutoff-0.5, color='red', linestyle='--', label=f'Nucleus cutoff (p={p})')
    plt.title(f"Nucleus (Top-p) Sampling with p={p}")
    plt.xlabel("Token Rank")
    plt.ylabel("Probability")
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    plt.annotate(f'Nucleus contains\n{p_cutoff} tokens', 
                xy=(p_cutoff-1, sorted_probs[p_cutoff-1]), 
                xytext=(p_cutoff+5, sorted_probs[0]*0.7),
                arrowprops=dict(arrowstyle='->', color='red'),
                bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    plt.subplot(2, 1, 2)
    plt.plot(range(min(30, len(sorted_probs))), cumulative_probs[:30], 'r-o', label='Cumulative probability')
    plt.axhline(y=p, color='green', linestyle='--', label=f'p={p}')
    plt.axvline(x=p_cutoff-0.5, color='green', linestyle='--')
    
    plt.fill_between(range(p_cutoff), cumulative_probs[:p_cutoff], alpha=0.3, color='green')
    
    plt.title(f"Cumulative Probability Distribution")
    plt.xlabel("Number of tokens in nucleus")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.annotate(f'p={p} threshold', 
                xy=(p_cutoff-1, p), 
                xytext=(p_cutoff+5, p+0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Nucleus sampling visualization saved: {save_path}")
    plt.close()

def run_all_visualizations(model, dataset, prompt, device="cuda", output_dir="utils"):
    """
    Runs all visualization functions with the given model and prompt.
    
    Args:
        model: The language model
        dataset: Dataset containing vocabulary and decoding methods
        prompt: Starting text prompt
        device: Device to run the model on ('cuda' or 'cpu')
        output_dir: Directory to save visualizations
    """
    print(f"Generating visualizations for prompt: '{prompt}'")
    
    os.makedirs(output_dir, exist_ok=True)
    
    words = prompt.strip().split()
    tokens = [dataset.dictionary.word2idx.get(w, dataset.dictionary.word2idx['<UNK>']) 
             for w in words]
    
    with torch.no_grad():
        input_tensor = torch.LongTensor([tokens]).to(device)
        output, _ = model(input_tensor)
        logits = output[0, -1, :]
        probs = F.softmax(logits, dim=0)
        
        print("\n1. Creating greedy vs top-k comparison...")
        visualize_greedy_vs_topk(probs, save_path=f"{output_dir}/greedy_vs_topk.png")
        
        print("\n2. Creating beam search tree visualization...")
        visualize_beam_search_tree(
            model, dataset, prompt,
            beam_width=3, max_steps=3,
            save_path=f"{output_dir}/beam_search_tree.png"
        )
        
        print("\n3. Creating nucleus sampling visualization with p=0.4...")
        visualize_nucleus_sampling(
            probs, p=0.4,
            save_path=f"{output_dir}/nucleus_sampling_p04.png"
        )
    
    print(f"\nAll visualizations saved to {output_dir}/ directory.")
