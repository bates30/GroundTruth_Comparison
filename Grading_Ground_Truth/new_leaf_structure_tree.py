import json
import pandas as pd
import networkx as nx
from pathlib import Path
import sys

base_dir = Path(__file__).resolve().parent
OUTPUT_JSON = base_dir / "new_tree_output.json"

# Get parent directory to access input_data folder
parent_dir = base_dir.parent

# Add parent directory to Python path to import from input_data
sys.path.insert(0, str(parent_dir))

from input_data.Structure_for_Bus123 import get_positions
import input_data.Structure_for_Bus123 as Structure_for_Bus123
input_data = parent_dir / "input_data" / "input_123_Bus.json"
bus123_no_switch = parent_dir / "input_data" / "line_data.xls"
bus123_only_switch = parent_dir / "input_data" / "switch_data.xls"
source_node = 150
changable_node_size = 750


B = nx.Graph()

# Global memory structures to persist across lists
yellow_node_connections = {}  # Maps yellow_node -> connection_node
all_pink_nodes = set()  # All pink nodes ever created
all_yellow_nodes_history = set()  # All yellow nodes from previous lists
current_group_processed = set()  # Yellow nodes from current group that have been processed as red

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


def main(red_node, yellow_nodes, source, G, group_idx=0, count=0):
    global B, yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed
    B.clear()

    B.add_node(red_node)

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


    # Handle red node connection - reverted to parent sibling relationship
    # Only keep the previous yellow connection as dashed edge
    red_node_dashed_edge = None

    try:
        red_path = nx.shortest_path(G, source=red_node, target=source)

        # If red_node was previously a yellow_node, keep its old connection as dashed (BLACK)
        if red_node in yellow_node_connections:
            stored_connection = yellow_node_connections[red_node]
            if stored_connection != red_node:
                # Keep the previous yellow connection as dashed edge (BLACK)
                B.add_edge(red_node, stored_connection)
                red_node_dashed_edge = (red_node, stored_connection)
        # Otherwise, connect red_node to its parent in the tree (first pink node on path to source)
        else:
            # Find first pink node on red path
            first_pink_on_red_path = None
            for node in red_path[1:]:
                if node in new_intersection_nodes:
                    first_pink_on_red_path = node
                    break

            if first_pink_on_red_path is not None:
                # Add solid edge to parent (first pink node)
                B.add_edge(red_node, first_pink_on_red_path)
            else:
                # Last resort: connect directly to source
                B.add_edge(red_node, source)

    except nx.NetworkXNoPath:
        print(f"No path found between {red_node} and {source}")
        # Fallback: connect to source
        B.add_edge(red_node, source)

    # Previous yellow nodes: exclude current list AND red_node (needed for later logic)
    previous_yellow_nodes = all_yellow_nodes_history - set(yellow_nodes) - {red_node}

    # Split nodes that are both pink (intersection) and yellow/orange (listening sensors)
    # into two separate nodes for the JSON output structure
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

    for original_node, color_type in nodes_to_split:
        # Create a new node identifier for the yellow/orange version
        yellow_orange_node = f"{original_node}_listening"

        # Add the new yellow/orange node to graph B
        B.add_node(yellow_orange_node)

        # Connect the new yellow/orange node to the original (now pink-only) node
        B.add_edge(yellow_orange_node, original_node)

        # Store the mapping
        split_node_mapping[original_node] = (original_node, yellow_orange_node, color_type)

    results = {
        "new_intersection_nodes": list(new_intersection_nodes),
        "split_node_mapping": split_node_mapping
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

# Create consensus_tree format output with split node handling
# J# = junction nodes (internal nodes - all pink/shared nodes)
# L# = leaf nodes (all sensors - both split and pure leaf)

# Get all unique sensor nodes (all nodes that were ever yellow)
all_sensor_nodes = []
for group in yellow_node_groups:
    all_sensor_nodes.extend(group)
all_sensor_nodes = sorted(set(all_sensor_nodes))

# Get all unique shared/pink nodes (intersection nodes)
all_shared_nodes = sorted(all_pink_nodes)

# Identify sensors that are ALSO at shared positions (these get split)
sensors_at_shared_positions = set(all_sensor_nodes) & set(all_shared_nodes)

# Pure leaf sensors (not at shared positions)
pure_leaf_sensors = set(all_sensor_nodes) - sensors_at_shared_positions

# ALL pink nodes become junction nodes (internal nodes)
all_junction_nodes = set(all_shared_nodes)

# Create labels: J# for junction nodes (using actual node numbers)
node_to_j_label = {}
for node in sorted(all_junction_nodes):
    node_to_j_label[node] = f"J{node}"

# Create L# labels for ALL sensor nodes (both split and pure leaf)
node_to_l_label = {}
for node in sorted(all_sensor_nodes):
    node_to_l_label[node] = f"L{node}"

# Build the tree structure with proper parent-child relationships using BFS
nodes_list = []

# Find the root (source node should be in the tree)
# Always use the source_node as the root
root_node = source_node

# Check if source_node is a junction node, if not, add it as one
if root_node not in all_junction_nodes:
    all_junction_nodes.add(root_node)

# Create or update the label for the root node
node_to_j_label[root_node] = f"J{root_node}"
root_label = node_to_j_label[root_node]

# Build parent-child relationships using BFS from root on the actual graph B (with splits)
from collections import defaultdict, deque

parent_map = {}  # Maps node -> parent node
children_map = defaultdict(list)  # Maps node -> list of children nodes

# BFS to establish parent-child relationships on actual graph B
visited = {root_node}
queue = deque([root_node])

while queue:
    current = queue.popleft()

    if current in B.nodes():
        for neighbor in B.neighbors(current):
            # Skip source node and already visited nodes
            if neighbor == source_node or neighbor in visited:
                continue

            visited.add(neighbor)
            parent_map[neighbor] = current
            children_map[current].append(neighbor)
            queue.append(neighbor)

# Helper function to convert actual node to label
def node_to_label(node):
    """Convert a node (either int or string like '25_listening') to its J# or L# label"""
    if isinstance(node, str) and "_listening" in node:
        # This is a split listening node - extract original node number
        original_node = int(node.replace("_listening", ""))
        return node_to_l_label.get(original_node)
    elif node in all_junction_nodes:
        # This is a junction node (pink/shared)
        return node_to_j_label.get(node)
    elif node in all_sensor_nodes:
        # This is a pure leaf sensor (not at shared position)
        return node_to_l_label.get(node)
    else:
        # Unknown node type (shouldn't happen)
        return None

# Create entries for J# nodes (junction nodes - internal nodes)
for j_node in sorted(all_junction_nodes):
    j_label = node_to_j_label[j_node]
    children = []

    # Get all children from the actual graph B for this j_node
    if j_node in children_map:
        for child_node in children_map[j_node]:
            child_label = node_to_label(child_node)
            if child_label:
                children.append(child_label)

    # Sort children for consistent output
    children.sort()

    node_entry = {
        "name": j_label,
        "kind": "junction",
        "level": 0.0,  # Placeholder - would need distance calculation
        "children": children
    }
    nodes_list.append(node_entry)

# Create entries for L# nodes (ALL sensor nodes - both split and pure leaf)
for sensor_node in sorted(all_sensor_nodes):
    sensor_label = node_to_l_label[sensor_node]

    node_entry = {
        "name": sensor_label,
        "kind": "leaf",
        "level": 0.0,
        "children": []
    }
    nodes_list.append(node_entry)

# Create the final output structure
tree_output = {
    "root": root_label,
    "node_count": len(nodes_list),
    "edge_count": len(nodes_list) - 1,  # In a tree, edges = nodes - 1
    "nodes": nodes_list,
    "metadata": {
        "source": "new_leaf_structure_tree.py",
        "source_node": source_node,
        "split_nodes": sorted(list(sensors_at_shared_positions))
    }
}

# Save the consensus tree format output
with open(OUTPUT_JSON, 'w') as f:
    json.dump(tree_output, f, indent=2)

# print(f"Root: {root_label}")
# print(f"Total J# nodes (junctions/internal): {len(all_junction_nodes)}")
# print(f"Total L# nodes (all sensors): {len(all_sensor_nodes)}")
# print(f"  - Split sensors (at shared positions): {len(sensors_at_shared_positions)}")
# print(f"  - Pure leaf sensors: {len(pure_leaf_sensors)}")
# print(f"Total nodes: {len(nodes_list)}")
# print(f"Split nodes (sensors at shared positions): {sorted(list(sensors_at_shared_positions))}")
