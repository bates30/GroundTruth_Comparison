import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque, defaultdict
import numpy as np

# Paths
base_dir = Path(__file__).resolve().parent.parent
new_tree_json = base_dir / "new_tree_output.json"
consensus_tree_json = base_dir / "consenus_tree.json"  # Note: filename has typo 'consenus'
output_dir = Path(__file__).resolve().parent

def load_tree(json_path):
    """Load tree structure from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_all_leaves(tree_data):
    """Extract all leaf node names from tree data."""
    leaves = []
    for node in tree_data['nodes']:
        if node['kind'] == 'leaf':
            leaves.append(node['name'])
    return sorted(leaves)

def build_newick_from_json(tree_data, new_root):
    """
    Build Newick string from JSON tree data, rooted at new_root.
    Re-roots the tree to new_root using BFS traversal.
    """
    # Build adjacency map (undirected)
    adjacency = defaultdict(list)
    for node in tree_data['nodes']:
        for child in node['children']:
            adjacency[node['name']].append(child)
            adjacency[child].append(node['name'])

    # BFS from new_root to build parent-child relationships
    children_map = defaultdict(list)
    visited = {new_root}
    queue = deque([new_root])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                children_map[current].append(neighbor)
                queue.append(neighbor)

    # Recursively build Newick string
    def build_newick(node):
        if not children_map[node]:
            return node
        child_newicks = [build_newick(child) for child in sorted(children_map[node])]
        return f"({','.join(child_newicks)})"

    return build_newick(new_root) + ";"

def get_bipartitions_from_graph(G, root, all_tips):
    """
    Extract bipartitions from a directed tree graph.
    Each internal node defines a bipartition: the leaves in its subtree vs the rest.
    Only includes leaves that are in all_tips (common leaves).
    """
    partitions = []
    all_tips_set = set(all_tips)

    # For each internal node (junction), find all leaves in its subtree
    for node in G.nodes():
        if G.nodes[node]['kind'] == 'junction':
            # Get all descendants
            descendants = nx.descendants(G, node)
            # Filter to only leaf nodes that are in common leaves
            leaves_in_subtree = {n for n in descendants
                               if G.nodes[n]['kind'] == 'leaf' and n in all_tips_set}

            # Only include non-trivial partitions (not all leaves, not empty)
            if 0 < len(leaves_in_subtree) < len(all_tips):
                partition_str = ','.join(sorted(leaves_in_subtree))
                # Avoid duplicates
                if partition_str not in partitions:
                    partitions.append(partition_str)

    return partitions

def calculate_partition_metrics(bible_partitions, test_partitions):
    """Calculate partition agreement metrics."""
    bible_set = set(bible_partitions)
    test_set = set(test_partitions)

    common = bible_set & test_set
    bible_only = bible_set - test_set
    test_only = test_set - bible_set

    max_partitions = max(len(bible_set), len(test_set))
    agreement = (len(common) / max_partitions * 100) if max_partitions > 0 else 0

    rf_distance = len(bible_only) + len(test_only)

    return {
        'common': sorted(common),
        'bible_only': sorted(bible_only),
        'test_only': sorted(test_only),
        'agreement': agreement,
        'rf_distance': rf_distance
    }

def display_split(split_str, all_tips, index):
    """Display a split in readable format."""
    leaves_in_split = split_str.split(',')
    complement = sorted(set(all_tips) - set(leaves_in_split))

    print(f"  [{index:2d}] Side A ({len(leaves_in_split)} leafs): {{{', '.join(leaves_in_split)}}}")
    print(f"       Side B ({len(complement)} leafs): {{{', '.join(complement)}}}")
    print()

def print_detailed_partitions(metrics, all_tips):
    """Print detailed partition listings."""
    print("\n===============================================")
    print("DETAILED PARTITION LISTINGS FOR VALIDATION")
    print("===============================================\n")

    # Common splits
    print(f"===== 1. COMMON SPLITS ({len(metrics['common'])} total) =====")
    print("These bipartitions exist in BOTH Bible and Test trees\n")
    if metrics['common']:
        for i, split in enumerate(metrics['common'], 1):
            display_split(split, all_tips, i)
    else:
        print("  (None)\n")

    # Bible-only splits
    print(f"\n===== 2. BIBLE-ONLY SPLITS ({len(metrics['bible_only'])} total) =====")
    print("These bipartitions exist ONLY in the Bible tree\n")
    if metrics['bible_only']:
        for i, split in enumerate(metrics['bible_only'], 1):
            display_split(split, all_tips, i)
    else:
        print("  (None)\n")

    # Test-only splits
    print(f"\n===== 3. TEST-ONLY SPLITS ({len(metrics['test_only'])} total) =====")
    print("These bipartitions exist ONLY in the Test tree\n")
    if metrics['test_only']:
        for i, split in enumerate(metrics['test_only'], 1):
            display_split(split, all_tips, i)
    else:
        print("  (None)\n")

def compute_partition_scores(metrics, all_tips):
    """
    Compute partition disagreement score for each leaf.
    Higher score = leaf appears in more disagreeing partitions.
    """
    scores = {tip: 0 for tip in all_tips}

    # Count how many differing partitions each leaf appears in
    for split_str in metrics['bible_only']:
        for leaf in split_str.split(','):
            if leaf in scores:
                scores[leaf] += 1

    for split_str in metrics['test_only']:
        for leaf in split_str.split(','):
            if leaf in scores:
                scores[leaf] += 1

    # Normalize to 0-1
    max_score = max(scores.values()) if scores.values() else 1
    min_score = min(scores.values()) if scores.values() else 0

    if max_score > min_score:
        normalized = {tip: (scores[tip] - min_score) / (max_score - min_score)
                     for tip in scores}
    else:
        normalized = {tip: 0.0 for tip in scores}

    return normalized

def get_node_color_from_score(score):
    """Map partition disagreement score to color."""
    # Color gradient: lightblue -> yellow -> orange -> red
    if score < 0.33:
        # Blue to yellow
        ratio = score / 0.33
        r = int(173 + (255 - 173) * ratio)
        g = int(216 + (255 - 216) * ratio)
        b = int(230 + (0 - 230) * ratio)
    elif score < 0.67:
        # Yellow to orange
        ratio = (score - 0.33) / 0.34
        r = 255
        g = int(255 - (255 - 165) * ratio)
        b = 0
    else:
        # Orange to red
        ratio = (score - 0.67) / 0.33
        r = 255
        g = int(165 - 165 * ratio)
        b = 0

    return f'#{r:02x}{g:02x}{b:02x}'

def build_graph_from_json_with_root(tree_data, new_root):
    """Build a directed NetworkX graph from tree JSON data, rooted at new_root."""
    # Build adjacency map (undirected)
    adjacency = defaultdict(list)
    node_kinds = {}

    for node in tree_data['nodes']:
        node_kinds[node['name']] = node['kind']
        for child in node['children']:
            adjacency[node['name']].append(child)
            adjacency[child].append(node['name'])

    # BFS from new_root to build directed tree
    G = nx.DiGraph()
    G.add_node(new_root, kind=node_kinds[new_root])

    visited = {new_root}
    queue = deque([new_root])

    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                G.add_node(neighbor, kind=node_kinds[neighbor])
                G.add_edge(current, neighbor)
                queue.append(neighbor)

    return G

def hierarchical_layout(G, root, horizontal_spacing=1.8, vertical_spacing=2.5):
    """Create a top-down hierarchical layout for tree visualization."""
    if root not in G.nodes():
        return {}

    # BFS to determine levels and parent-child relationships
    levels = {root: 0}
    children = defaultdict(list)
    queue = deque([root])

    while queue:
        node = queue.popleft()
        for child in G.successors(node):
            levels[child] = levels[node] + 1
            children[node].append(child)
            queue.append(child)

    # Sort children for consistent layout
    for parent in children:
        children[parent] = sorted(children[parent])

    # Calculate subtree widths
    subtree_widths = {}
    def calc_widths(node):
        if node in subtree_widths:
            return subtree_widths[node]
        if not children[node]:
            subtree_widths[node] = 1
        else:
            subtree_widths[node] = sum(calc_widths(child) for child in children[node])
        return subtree_widths[node]

    calc_widths(root)

    # Assign positions
    pos = {}
    def assign_positions(node, x_start, level):
        y = -level * vertical_spacing

        if not children[node]:
            # Leaf node
            pos[node] = (x_start, y)
        else:
            # Internal node - center over children
            x_current = x_start
            for child in children[node]:
                assign_positions(child, x_current, level + 1)
                x_current += subtree_widths[child] * horizontal_spacing

            # Center parent over children
            child_positions = [pos[child][0] for child in children[node]]
            parent_x = (min(child_positions) + max(child_positions)) / 2
            pos[node] = (parent_x, y)

    assign_positions(root, 0, 0)
    return pos

def visualize_trees_with_partition_coloring(bible_tree, test_tree, bible_root, test_root,
                                            partition_scores, common_leaves,
                                            bible_only_leaves, test_only_leaves,
                                            output_path):
    """Visualize both trees side by side with partition disagreement coloring."""
    # Build directed graphs using their respective roots
    bible_G = build_graph_from_json_with_root(bible_tree, bible_root)
    test_G = build_graph_from_json_with_root(test_tree, test_root)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))

    # Calculate layouts
    bible_pos = hierarchical_layout(bible_G, bible_root)
    test_pos = hierarchical_layout(test_G, test_root)

    # Assign colors based on partition scores - colors must match between trees for same leaves
    def get_node_colors_and_sizes(G, tree_root, only_leaves_this_tree):
        colors = []
        sizes = []
        for node in G.nodes():
            if node == tree_root:
                colors.append('green')
                sizes.append(900)
            elif G.nodes[node]['kind'] == 'leaf':
                if node in partition_scores:
                    # Use the SAME color for this leaf in both trees
                    colors.append(get_node_color_from_score(partition_scores[node]))
                    sizes.append(700)
                elif node in only_leaves_this_tree:
                    # Purple for Bible-only, dark green for Test-only
                    colors.append('purple' if only_leaves_this_tree == bible_only_leaves else 'darkgreen')
                    sizes.append(800)
                else:
                    colors.append('lightblue')
                    sizes.append(700)
            else:  # Junction
                colors.append('lightgray')
                sizes.append(500)
        return colors, sizes

    bible_colors, bible_sizes = get_node_colors_and_sizes(bible_G, bible_root, bible_only_leaves)
    test_colors, test_sizes = get_node_colors_and_sizes(test_G, test_root, test_only_leaves)

    # Draw Bible tree
    nx.draw_networkx_edges(bible_G, pos=bible_pos, ax=ax1, edge_color="gray",
                           width=2, arrows=False)
    nx.draw_networkx_nodes(bible_G, pos=bible_pos, ax=ax1, node_color=bible_colors,
                          node_size=bible_sizes)
    nx.draw_networkx_labels(bible_G, pos=bible_pos, ax=ax1, font_color="black",
                           font_size=9)

    ax1.set_title(f"Bible Tree (rooted at {bible_root})\n(Leafs colored by partition disagreement)",
                  fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Draw Test tree
    nx.draw_networkx_edges(test_G, pos=test_pos, ax=ax2, edge_color="gray",
                           width=2, arrows=False)
    nx.draw_networkx_nodes(test_G, pos=test_pos, ax=ax2, node_color=test_colors,
                          node_size=test_sizes)
    nx.draw_networkx_labels(test_G, pos=test_pos, ax=ax2, font_color="black",
                           font_size=9)

    ax2.set_title(f"Test Tree (rooted at {test_root})\n(Leafs colored by partition disagreement)",
                  fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor='lightblue', label='Low partition disagreement'),
        Patch(facecolor='yellow', label='Medium partition disagreement'),
        Patch(facecolor='red', label='High partition disagreement'),
        Patch(facecolor='purple', label='Bible-only leaf'),
        Patch(facecolor='darkgreen', label='Test-only leaf'),
        Patch(facecolor='lightgray', label='Junction'),
        Patch(facecolor='green', label=f'Root nodes')
    ]
    fig.legend(handles=legend_items, loc='lower center', fontsize=12, ncol=4,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    bible_tree = load_tree(new_tree_json)
    test_tree = load_tree(consensus_tree_json)

    # Get leaves
    bible_leaves = get_all_leaves(bible_tree)
    test_leaves = get_all_leaves(test_tree)
    common_leaves = sorted(set(bible_leaves) & set(test_leaves))
    bible_only_leaves = sorted(set(bible_leaves) - set(test_leaves))
    test_only_leaves = sorted(set(test_leaves) - set(bible_leaves))

    # Use roots from JSON files
    bible_root = bible_tree['root']
    test_root = test_tree['root']

    print("\n=== LEAFS ===")
    print(f"Bible leafs:        {len(bible_leaves)}")
    print(f"Test leafs:         {len(test_leaves)}")
    print(f"Common leafs:       {len(common_leaves)}")
    print(f"Bible-only leafs:   {len(bible_only_leaves)} -> {', '.join(bible_only_leaves) if bible_only_leaves else 'None'}")
    print(f"Test-only leafs:    {len(test_only_leaves)} -> {', '.join(test_only_leaves) if test_only_leaves else 'None'}")
    print(f"\nBible tree root: {bible_root}")
    print(f"Test tree root:  {test_root}")
    print("(Using original roots from JSON files)\n")

    # Build graphs for partition extraction
    bible_graph = build_graph_from_json_with_root(bible_tree, bible_root)
    test_graph = build_graph_from_json_with_root(test_tree, test_root)

    # Get bipartitions
    bible_partitions = get_bipartitions_from_graph(bible_graph, bible_root, common_leaves)
    test_partitions = get_bipartitions_from_graph(test_graph, test_root, common_leaves)

    # Calculate metrics
    metrics = calculate_partition_metrics(bible_partitions, test_partitions)

    print("\n=== PARTITIONS (SPLITS) ===")
    print(f"Bible partitions:   {len(bible_partitions)}")
    print(f"Test partitions:    {len(test_partitions)}")
    print(f"Common partitions:  {len(metrics['common'])}")
    print(f"Agreement:          {metrics['agreement']:.1f}%")

    # Print detailed partition listings
    print_detailed_partitions(metrics, common_leaves)

    # Compute partition disagreement scores for visualization
    partition_scores = compute_partition_scores(metrics, common_leaves)

    # Visualize trees with partition coloring
    output_path = output_dir / "RF_output.png"
    visualize_trees_with_partition_coloring(
        bible_tree, test_tree, bible_root, test_root, partition_scores,
        common_leaves, bible_only_leaves, test_only_leaves, output_path
    )

    print("\n=== SUMMARY ===")
    print(f"Partition Agreement: {metrics['agreement']:.1f}%")
    print(f"RF Distance: {metrics['rf_distance']}")

if __name__ == "__main__":
    main()
