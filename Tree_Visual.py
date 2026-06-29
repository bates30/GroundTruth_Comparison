import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from input_data.Structure_for_Bus123 import get_positions
import input_data.Structure_for_Bus123 as Structure_for_Bus123
from pathlib import Path

base_dir = Path(__file__).resolve().parent
SAVE_DIR = base_dir / "output_data"


input_data = r"input_data\input_123_Bus.json"
bus123_no_switch = r"input_data\line_data.xls"
bus123_only_switch = r"input_data\switch_data.xls"

source_node = 150
changable_node_size = 750


B = nx.Graph()

# Global memory structures to persist across lists
yellow_node_connections = {}  # Maps yellow_node -> connection_node
all_pink_nodes = set()  # All pink nodes ever created
all_yellow_nodes_history = set()  # All yellow nodes from previous lists
current_group_processed = set()  # Yellow nodes from current group that have been processed as red

#Sibling ordering memory - preserves left-to-right order of nodes with same parent
sibling_order_memory = {}


def path_distance_from_graph(G, path):
    return sum(float(G[u][v]["length"]) for u, v in zip(path[:-1], path[1:]))


def load_data():
    with open(input_data, "r") as f:
        data = json.load(f)

    bus_df = pd.read_excel(bus123_no_switch)
    switch_df = pd.read_excel(bus123_only_switch)

    return data, bus_df, switch_df


def add_nodes(bus_df, switch_df):
    G = nx.Graph()

    col1 = bus_df.iloc[:, 0].dropna().tolist()[1:]
    col2 = bus_df.iloc[:, 1].dropna().tolist()[1:]
    col3 = bus_df.iloc[:, 2].dropna().tolist()[1:]

    unique_nodes = set(col1).union(col2)
    G.add_nodes_from(unique_nodes)

    for u, v, length in zip(col1, col2, col3):
        G.add_edge(u, v, length=length)

    switch1 = switch_df.iloc[:, 0].dropna().tolist()[1:]
    switch2 = switch_df.iloc[:, 1].dropna().tolist()[1:]

    switch3 = switch_df.iloc[:, 2].dropna().tolist()[1:]

    for u, v, z in zip(switch1, switch2, switch3):
        if z=='closed':
            G.add_edge(u, v, length=1)
        else:
            print(f'Connection between {u}, {v} is not possible')

    return G


def get_edge_labels(G):
    edge_labels = {}
    for edge, length in nx.get_edge_attributes(G, "length").items():
        edge_labels[edge] = "S\n" if length == 1 else f"{length}\n"
    return edge_labels

#Positions for tree
def improved_hierarchical_layout_with_memory(tree_graph, source_node, horizontal_spacing=2.0, vertical_spacing=1.5):
    """
    Create a top-down hierarchical layout that centers children under their parent.
    PRESERVES sibling ordering when they stay attached to the same parent.

    Args:
        tree_graph: NetworkX graph (should be a tree)
        source_node: The root node (displayed at top)
        horizontal_spacing: Minimum space between nodes
        vertical_spacing: Space between levels

    Returns:
        Dictionary mapping node -> (x, y) position
    """
    global sibling_order_memory

    if source_node not in tree_graph.nodes():
        return {}

    from collections import defaultdict

    # Build parent-child relationships via BFS
    parent = {}
    children = defaultdict(list)
    visited = {source_node}
    queue = [source_node]

    while queue:
        node = queue.pop(0)
        for neighbor in tree_graph.neighbors(node):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                children[node].append(neighbor)
                queue.append(neighbor)

    # Apply stable ordering to children based on memory
    for parent_node, child_list in children.items():
        if not child_list:
            continue

        # Create a key from parent and set of children
        child_set = frozenset(child_list)
        memory_key = (parent_node, child_set)

        # Check if we have a previous ordering for this exact parent-children combination
        if memory_key in sibling_order_memory:
            old_order = sibling_order_memory[memory_key]
            # Reorder children to match old order as much as possible
            new_children = []
            # First, add children that were in the old order (in that order)
            for old_child in old_order:
                if old_child in child_list:
                    new_children.append(old_child)
            # Then add any new children not in old order
            for child in child_list:
                if child not in new_children:
                    new_children.append(child)
            children[parent_node] = new_children
        else:
            # No previous ordering, sort consistently by node value
            children[parent_node] = sorted(child_list)

        # Store/update the ordering in memory
        sibling_order_memory[memory_key] = children[parent_node]

    # Clean up stale memory entries (parent-children combos that no longer exist)
    current_keys = set()
    for parent_node, child_list in children.items():
        current_keys.add((parent_node, frozenset(child_list)))

    keys_to_remove = [k for k in sibling_order_memory.keys() if k not in current_keys]
    for k in keys_to_remove:
        del sibling_order_memory[k]

    # Calculate subtree sizes (number of leaf descendants)
    subtree_widths = {}
    def calc_widths(node):
        if node in subtree_widths:
            return subtree_widths[node]
        if node not in children or not children[node]:
            subtree_widths[node] = 1
        else:
            subtree_widths[node] = sum(calc_widths(child) for child in children[node])
        return subtree_widths[node]

    calc_widths(source_node)

    # Assign positions recursively
    pos = {}

    def assign_positions(node, x_start, level):
        y = -level * vertical_spacing  # Negative so source is at top (0) and tree grows downward

        if node not in children or not children[node]:
            # Leaf node
            pos[node] = (x_start, y)
        else:
            # Internal node - center it over its children
            child_list = children[node]
            x_current = x_start

            # Position children first (using stable ordering)
            for child in child_list:
                assign_positions(child, x_current, level + 1)
                x_current += subtree_widths[child] * horizontal_spacing

            # Center parent over children
            child_positions = [pos[child][0] for child in child_list]
            parent_x = (min(child_positions) + max(child_positions)) / 2
            pos[node] = (parent_x, y)

    assign_positions(source_node, 0, 0)

    return pos


def main(red_node, yellow_nodes, source, G, group_idx=0, count=0):
    global B, yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed
    B.clear()

    B.add_node(red_node)

    fig, ax = plt.subplots(figsize=(16, 12))  # Increased from (10, 8)
    node_size = changable_node_size

    path = nx.shortest_path(G, source=red_node, target=source)
    between_nodes_probe = path

    probe_nodes = set(path[1:-1])
    path_nodes = set()

    yellow_nodes_first_instance = []
    red_nodes_first_instance = []
    new_intersection_nodes = []

    # Find the nodes on shortest path of probing/listening sensors for this iteration
    for y in yellow_nodes:
        try:
            path = nx.shortest_path(G, source=y, target=source)
            path_nodes.update(path[1:-1])

            intersection_node = None

            if path[0] in between_nodes_probe:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)

                #If the first node on the path of probing sensor is a listening probe
                if intersection_node is not None:
                    yellow_nodes_first_instance.append(y)
            #If any listening sensor on the path of probing sensor
            elif y in between_nodes_probe:
                pass

            #if red_node on shortest path of listening sensor
            elif between_nodes_probe[0] in path:
                intersection_node = between_nodes_probe[0]
                red_nodes_first_instance.append(intersection_node)

            #Checks all nodes on listening sensor's shortest path, add nodes to list if true, None otherwise
            else:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)

            #Adding nodes to new_intersection_nodes (list of all pink nodes)
            if intersection_node is not None:
                new_intersection_nodes.append(intersection_node)

        except nx.NetworkXNoPath:
            print(f"No path found between {y} and {source}")

    #Adds listening nodes as impedence points (pink nodes)
    if yellow_nodes_first_instance:
        new_intersection_nodes += yellow_nodes_first_instance


    #Adds probing node as impedence point (pink nodes)
    if red_nodes_first_instance:
        new_intersection_nodes += red_nodes_first_instance


    # Add all previous pink nodes from memory to keep them
    for i in all_pink_nodes:
        if i not in new_intersection_nodes:
            new_intersection_nodes.append(i)

    # Update global pink nodes set
    all_pink_nodes.update(new_intersection_nodes)

    # Get previous yellow nodes, explicitly excluding the current red_node
    # This ensures a node that was yellow before but is now red doesn't get treated as both
    previous_yellows_only = all_yellow_nodes_history - {red_node}

    # Combine current yellow nodes with previous yellow nodes (excluding the red_node)
    all_yellow_to_process = list(set(yellow_nodes) | previous_yellows_only)

    # Separate processing for current yellow nodes vs previous (orange) nodes
    # Handle CURRENT yellow node connections - find first overlapping node with red_node path
    for i in yellow_nodes:
        # Skip if this is the current red_node
        if i == red_node:
            continue
        try:
            yellow_path = nx.shortest_path(G, source=i, target=source)
            red_path = nx.shortest_path(G, source=red_node, target=source)

            # Find first overlapping node between yellow and red paths
            best_node = None
            best_index = len(yellow_path)

            for idx, node in enumerate(yellow_path):
                if node in red_path and node in new_intersection_nodes and node != i:
                    if idx < best_index:
                        best_index = idx
                        best_node = node

            # If no overlap with red path, find first pink node on yellow path
            if best_node is None:
                for idx, node in enumerate(yellow_path):
                    if node in new_intersection_nodes and node != i:
                        if idx < best_index:
                            best_index = idx
                            best_node = node

            if best_node is not None:
                # Check if this yellow node has a previous connection
                if i in yellow_node_connections:
                    old_connection = yellow_node_connections[i]
                    # Check if the new best_node is closer than the old one
                    try:
                        old_idx = yellow_path.index(old_connection) if old_connection in yellow_path else len(yellow_path)

                        # Update to closer connection
                        if best_index < old_idx:
                            yellow_node_connections[i] = best_node
                            B.add_edge(i, best_node)
                        else:
                            # Keep old connection
                            B.add_edge(i, old_connection)
                    except ValueError:
                        # If old connection is no longer on path, use new one
                        yellow_node_connections[i] = best_node
                        B.add_edge(i, best_node)
                else:
                    # New yellow node, record connection
                    yellow_node_connections[i] = best_node
                    B.add_edge(i, best_node)

                Structure_for_Bus123.master_dict[i] = yellow_node_connections.get(i, best_node)

        except nx.NetworkXNoPath:
            print(f"No path found between {i} and {source}")

    # Handle previous (ORANGE) yellow node connections - do not add edge if at same node as pink node
    for i in previous_yellows_only:
        # Skip if this orange node is at the same location as a pink node
        if i in new_intersection_nodes:
            continue

        # Orange nodes should keep their stagnant connection from when they were yellow
        if i in yellow_node_connections:
            stored_connection = yellow_node_connections[i]
            # Always use the stored connection, never update it
            B.add_edge(i, stored_connection)
        # If somehow an orange node doesn't have a stored connection, skip it
        # (this shouldn't happen in normal operation)

    # Connect pink nodes to each other
    for i in new_intersection_nodes:
        try:
            path = nx.shortest_path(G, source=i, target=source)

            new_node = None
            for node in path[1:]:
                if node in new_intersection_nodes:
                    new_node = node
                    break

            if new_node is not None:
                B.add_edge(i, new_node)

        except nx.NetworkXNoPath:
            print(f"No path found between {i} and {source}")

    # Connect closest pink node to source
    total_dist = float("inf")
    new_node = None
    for i in new_intersection_nodes:
        try:
            path = nx.shortest_path(G, source=i, target=source)
            new_dist = path_distance_from_graph(G, path)
            if new_dist < total_dist:
                total_dist = new_dist
                new_node = i
        except nx.NetworkXNoPath:
            print(f"No path found between {i} and {source}")

    if new_node is not None:
        B.add_edge(source, new_node)


    # MODIFIED: Handle red node connection - always connect to closest pink node
    # Even for the first iteration
    try:
        red_path = nx.shortest_path(G, source=red_node, target=source)

        # Find the closest pink node to red_node
        closest_pink = None
        closest_index = len(red_path)

        for idx, node in enumerate(red_path):
            if node in new_intersection_nodes and node != red_node:
                if idx < closest_index:
                    closest_index = idx
                    closest_pink = node

        # Connect red_node to closest pink node
        if closest_pink is not None:
            B.add_edge(red_node, closest_pink)

    except nx.NetworkXNoPath:
        print(f"No path found between {red_node} and {source}")

    # Previous yellow nodes: exclude current list AND red_node (needed for later logic)
    previous_yellow_nodes = all_yellow_nodes_history - set(yellow_nodes) - {red_node}

    # Identify nodes that will need to be split (before layout calculation)
    # This helps us adjust spacing appropriately
    nodes_will_split = []
    for n in B.nodes():
        if n in new_intersection_nodes:
            if (n in yellow_nodes and n != red_node) or (n in previous_yellow_nodes):
                nodes_will_split.append(n)

    # Calculate hierarchical tree layout with increased horizontal spacing to accommodate splits
    # Use larger spacing if we have nodes that will be split
    h_spacing = 5.0 if nodes_will_split else 2.5
    tree_pos = improved_hierarchical_layout_with_memory(B, source, horizontal_spacing=h_spacing, vertical_spacing=3.0)

    # Handle disconnected nodes - position them separately to the left of the tree
    disconnected_nodes = [n for n in B.nodes() if n not in tree_pos]
    if disconnected_nodes:
        # Find the leftmost position in the tree
        if tree_pos:
            min_x = min(pos[0] for pos in tree_pos.values())
            max_y = max(pos[1] for pos in tree_pos.values())
        else:
            min_x = 0
            max_y = 0

        # Place disconnected nodes to the left of the tree, stacked vertically
        for idx, node in enumerate(disconnected_nodes):
            tree_pos[node] = (min_x - 5.0, max_y - idx * 3.0)

    # MAJOR CHANGE: Split nodes that are both pink (intersection) and yellow/orange (listening sensors)
    # into two separate nodes
    nodes_to_split = []

    # Identify nodes that need to be split (pink nodes that are also current or previous yellow)
    for n in B.nodes():
        if n in new_intersection_nodes:
            # Check if this pink node is also a current yellow node
            if n in yellow_nodes and n != red_node:
                nodes_to_split.append((n, 'yellow'))
            # Check if this pink node is also a previous yellow node
            elif n in previous_yellow_nodes:
                nodes_to_split.append((n, 'orange'))

    # Create new nodes for the split (use string labels to distinguish)
    split_node_mapping = {}  # Maps original_node -> (pink_node, yellow/orange_node)

    # Helper function to check if a position is too close to existing nodes
    def is_position_clear(pos_x, pos_y, existing_positions, min_distance=2.5):
        """Check if a position is at least min_distance away from all existing positions"""
        for ex_x, ex_y in existing_positions:
            distance = ((pos_x - ex_x)**2 + (pos_y - ex_y)**2)**0.5
            if distance < min_distance:
                return False
        return True

    for original_node, color_type in nodes_to_split:
        # Create a new node identifier for the yellow/orange version
        yellow_orange_node = f"{original_node}_listening"

        # Add the new yellow/orange node to graph B
        B.add_node(yellow_orange_node)

        # Position the new node with collision avoidance - always to the LEFT
        if original_node in tree_pos:
            orig_x, orig_y = tree_pos[original_node]

            # Get all existing positions (excluding the current original_node)
            existing_positions = [pos for node, pos in tree_pos.items() if node != original_node]

            # Try different LEFT offsets to avoid overlaps with other nodes
            # Prioritize same level, then vertical adjustments (up/down levels), then further left
            attempts = [
                (-4.0, 0.0),    # Left at same level (preferred)
                (-4.0, 1.0),    # Left and down one level
                (-4.0, -1.0),   # Left and up one level
                (-4.0, 2.0),    # Left and down two levels
                (-4.0, -2.0),   # Left and up two levels
                (-5.0, 0.0),    # Further left at same level
                (-5.0, 1.0),    # Further left and down
                (-5.0, -1.0),   # Further left and up
                (-6.0, 0.0),    # Even further left
                (-6.0, 1.5),    # Further left and down 1.5 levels
                (-6.0, -1.5),   # Further left and up 1.5 levels
                (-4.0, 3.0),    # Left and down three levels
                (-4.0, -3.0),   # Left and up three levels
                (-7.0, 0.0),    # Much further left
                (-8.0, 0.0),    # Very far left
            ]

            # Find first clear position
            new_pos = None
            for offset_x, offset_y in attempts:
                test_x = orig_x + offset_x
                test_y = orig_y + offset_y
                if is_position_clear(test_x, test_y, existing_positions):
                    new_pos = (test_x, test_y)
                    break

            # If no clear position found, use far left position anyway
            if new_pos is None:
                new_pos = (orig_x - 8.0, orig_y)

            tree_pos[yellow_orange_node] = new_pos

        # Connect the new yellow/orange node to the original (now pink-only) node
        B.add_edge(yellow_orange_node, original_node)

        # Store the mapping
        split_node_mapping[original_node] = (original_node, yellow_orange_node, color_type)

    # Determine node colors AFTER split nodes have been created
    node_colors = []
    node_color_map = {}  # Map each node to its color for easier lookup

    for n in B.nodes():
        color = None

        # Check if this is a split listening node (yellow/orange)
        if isinstance(n, str) and "_listening" in str(n):
            original_node = int(str(n).replace("_listening", ""))
            # Check if it was split as yellow or orange
            if original_node in split_node_mapping:
                _, _, color_type = split_node_mapping[original_node]
                color = color_type  # 'yellow' or 'orange'
        elif n == source:
            color = "green"
        elif n in new_intersection_nodes:
            # Pink nodes - but only if they weren't split
            # If they were split, the original node stays pure pink
            color = "pink"
        elif n == red_node:  # Red_node only colored red if NOT in intersection
            color = "red"
        elif n in previous_yellow_nodes:  # Previous yellow nodes stay orange even if on probe path
            color = "orange"
        elif n in probe_nodes:  # Other nodes on red path (not pink, not previous yellow, not red_node itself)
            color = "red"
        else:
            color = "yellow"

        node_colors.append(color)
        node_color_map[n] = color

    # Draw edges - all solid red edges (no dashed edge in this version)
    edges_to_draw_solid = list(B.edges())

    # Draw regular solid edges (RED)
    if edges_to_draw_solid:
        nx.draw_networkx_edges(
            B,
            pos=tree_pos,
            ax=ax,
            edgelist=edges_to_draw_solid,
            edge_color="red",
            width=2,
            style="solid",
        )

    # STEP 1: Draw red_node with larger size first if it's in intersection (will show as outline)
    # But only if it wasn't split (if split, it's now pure pink)
    if red_node in new_intersection_nodes and red_node not in [n for n, _ in nodes_to_split]:
        nx.draw_networkx_nodes(
            B,
            pos=tree_pos,
            nodelist=[red_node],
            node_size=changable_node_size * 1.5,
            node_color="red",
            ax=ax,
        )

    # STEP 2: Draw yellow nodes that are on intersection with larger size (will show as outline)
    # But only if they weren't split
    for n in B.nodes():
        if n in new_intersection_nodes and n in yellow_nodes and n != red_node:
            if n not in [node for node, _ in nodes_to_split]:
                nx.draw_networkx_nodes(
                    B,
                    pos=tree_pos,
                    ax=ax,
                    nodelist=[n],
                    node_size=changable_node_size * 1.5,
                    node_color='yellow',
                )

    # STEP 3: Draw orange nodes (previous yellow) that are on intersection with larger size (will show as outline)
    # But only if they weren't split
    for n in B.nodes():
        if n in new_intersection_nodes and n in previous_yellow_nodes:
            if n not in [node for node, _ in nodes_to_split]:
                nx.draw_networkx_nodes(
                    B,
                    pos=tree_pos,
                    ax=ax,
                    nodelist=[n],
                    node_size=changable_node_size * 1.5,
                    node_color='orange',
                )

    # STEP 4: Draw all nodes at normal size (this creates the center color on top of outlines)
    nx.draw_networkx_nodes(
        B,
        pos=tree_pos,
        ax=ax,
        nodelist=list(B.nodes()),
        node_color=node_colors,
        node_size=changable_node_size,
    )

    # Draw labels - need custom labels for split nodes
    labels = {}
    for n in B.nodes():
        if isinstance(n, str) and "_listening" in str(n):
            # This is a split listening node - use the original node number as label
            original_node = int(str(n).replace("_listening", ""))
            labels[n] = str(original_node)
        else:
            labels[n] = str(n)

    nx.draw_networkx_labels(
        B,
        pos=tree_pos,
        labels=labels,
        ax=ax,
        font_color="black",
    )

    legend_items = [
            Patch(facecolor="green", label="Source"),
            Patch(facecolor="yellow", label="Listening Sensor (Current)"),
            Patch(facecolor="orange", label="Listening Sensor (Previous)"),
            Patch(facecolor="red", label="Probing Sensor Location"),
            Patch(facecolor="pink", label="First Shared Node"),
            Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='Connections (Solid)'),
        ]
    ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(-0.15, 1.15))

    ax.set_title(f"Probe location: {red_node} | Group: {group_idx}")


    SAVE_DIR.mkdir(exist_ok=True)
    fig.savefig(
        SAVE_DIR / f"group_{group_idx}_no_{count}_probe_{red_node}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)  # Close figure to free memory and prevent blurring

    results = {
        "new_intersection_nodes": list(new_intersection_nodes)
    }

    return results


data, bus_df, switch_df = load_data()
G = add_nodes(bus_df, switch_df)
pos = nx.spring_layout(G, seed=42)

yellow_node_groups = data["sensor_locations"]
probe_depth = data.get("probe_depth", 0)  # Default to 0 if not specified




results = []
all_yellow_nodes = []

for group_idx, yellow_nodes in enumerate(yellow_node_groups):
    count = 0
    # Reset current_group_processed for each new group
    current_group_processed.clear()

    # Determine how many nodes to iterate through as red_nodes (probing sensors)
    if probe_depth == 0:
        # Use all sensor locations as probing sensors (current behavior)
        red_node_list = yellow_nodes
    else:
        # Only use the first probe_depth sensor locations as probing sensors
        red_node_list = yellow_nodes[:probe_depth]


    for idx, red_node in enumerate(red_node_list):

        all_yellow_nodes.append(red_node)
        count += 1
        # All nodes in the group are listening sensors except the current red_node
        other_yellow_nodes = [node for node in yellow_nodes if node != red_node]
        result = main(
            red_node,
            other_yellow_nodes,
            source_node,
            G,
            group_idx=group_idx,
            count=count,
        )
        results.append(result)

        # Add this red_node to current_group_processed after processing
        current_group_processed.add(red_node)

    # After processing all nodes in current group, update history with ALL yellow nodes from the group
    # (not just the ones that were probing sensors)
    all_yellow_nodes_history.update(yellow_nodes)
