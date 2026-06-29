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
all_yellow_nodes_history = set()  # All yellow nodes (all sensors) from previous lists
current_group_processed = set()  # Yellow nodes (listening sensors) from current group that have been processed as red (probing sensor)


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
        try:
            if z == 'closed':
                G.add_edge(u, v, length=1)
        except:
            print(f'{u},{v} has a open switch statee')

    return G

#was previously used but not currently
def get_edge_labels(G):
    edge_labels = {}
    for edge, length in nx.get_edge_attributes(G, "length").items():
        edge_labels[edge] = "S\n" if length == 1 else f"{length}\n"
    return edge_labels



def main(red_node, yellow_nodes, source, G, pos, group_idx=0, count=0):
    global B, yellow_node_connections, all_pink_nodes, all_yellow_nodes_history, current_group_processed

    #removes previously defined nodes/edges from the last group
    B.clear()

    B.add_node(red_node)

    #size of output (can change if needed)
    fig, ax = plt.subplots(figsize=(16, 12))
    #Defined in beggining
    node_size = changable_node_size

    #Background grid system
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[source],
        node_size=node_size,
        node_shape="o",
        node_color="grey",
        ax=ax,
    )

    #initializing some variables for iterating through yellow node
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
            #shortest path from each sensor and the power source
            path = nx.shortest_path(G, source=y, target=source)
            path_nodes.update(path[1:-1])

            intersection_node = None

            #If the first node on the path of probing sensor is a listening probe
            if path[0] in between_nodes_probe:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)


                if intersection_node is not None:
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        nodelist=[intersection_node],
                        node_size=node_size * 1.5,
                        node_shape="o",
                        node_color="yellow",
                        ax=ax,
                    )
                    yellow_nodes_first_instance.append(y)
            #If any listening sensor on the path of probing sensor
            elif y in between_nodes_probe:
                nx.draw_networkx_nodes(
                        G,
                        pos,
                        nodelist=[intersection_node],
                        node_size=node_size * 1.5,
                        node_shape="o",
                        node_color="yellow",
                )

            #if red_node on shortest path of listening sensor
            elif between_nodes_probe[0] in path:
                intersection_node = between_nodes_probe[0]
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[intersection_node],
                    node_size=node_size * 1.5,
                    node_shape="o",
                    node_color="red",
                    ax=ax,
                )
                red_nodes_first_instance.append(intersection_node)

            #Checks all nodes on listening sensor's shortest path, add nodes to list if true, None otherwise
            else:
                intersection_node = next((n for n in path if n in between_nodes_probe), None)

            #Adding nodes to new_intersection_nodes (list of all pink nodes)
            if intersection_node is not None:
                new_intersection_nodes.append(intersection_node)


        #current error for impossible switch state
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
    # Handle current yellow node connections - find first overlapping node with red_node path
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

        # Orange nodes should keep their FROZEN connection from when they were yellow
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
    # No more dashed edge feature
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
        B.add_edge(red_node, source)

    # Determine node colors
    # Previous yellow nodes: all yellow nodes from history that are not the current red_node
    # This includes nodes from previous groups AND nodes from current group that have already been processed
    previous_yellow_nodes = all_yellow_nodes_history - {red_node}
    node_colors = []
    for n in B.nodes():
        if n == source:
            node_colors.append("green")
        elif n in new_intersection_nodes:  # Pink nodes take priority - even if it's red_node
            node_colors.append("pink")
        elif n == red_node:  # Red_node only colored red if NOT in intersection
            node_colors.append("red")
        elif n in yellow_nodes:  # Current yellow nodes (listening sensors in current iteration)
            node_colors.append("yellow")
        elif n in previous_yellow_nodes:  # Previous yellow nodes stay orange (includes current group's already-processed nodes)
            node_colors.append("orange")
        elif n in probe_nodes:  # Other nodes on red path (not pink, not previous yellow, not current yellow, not red_node itself)
            node_colors.append("red")
        else:
            node_colors.append("yellow")



    nx.draw(
        G,
        pos=pos,
        ax=ax,
        with_labels=False,
        node_color="lightgrey",
        font_color="black",
        node_size=50,
    )

    # Draw all edges as solid red (no more dashed edge)
    nx.draw_networkx_edges(
        B,
        pos=pos,
        ax=ax,
        edgelist=list(B.edges()),
        edge_color="red",
        width=2,
        style="solid",
    )

    # STEP 1: Draw red_node with larger size first if it's in intersection (will show as outline)
    if red_node in new_intersection_nodes:
        nx.draw_networkx_nodes(
            B,
            pos=pos,
            nodelist=[red_node],
            node_size=changable_node_size * 1.5,
            node_color="red",
            ax=ax,
        )

    # STEP 2: Draw yellow nodes that are on intersection with larger size (will show as outline)
    for n in B.nodes():
        if n in new_intersection_nodes and n in yellow_nodes and n != red_node:
            nx.draw_networkx_nodes(
                B,
                pos=pos,
                ax=ax,
                nodelist=[n],
                node_size=changable_node_size * 1.5,
                node_color='yellow',
            )

    # STEP 3: Draw orange nodes (previous yellow) that are on intersection with larger size (will show as outline)
    for n in B.nodes():
        if n in new_intersection_nodes and n in previous_yellow_nodes:
            nx.draw_networkx_nodes(
                B,
                pos=pos,
                ax=ax,
                nodelist=[n],
                node_size=changable_node_size * 1.5,
                node_color='orange',
            )

    # STEP 4: Draw all nodes at normal size (this creates the center color on top of outlines)
    nx.draw_networkx_nodes(
        B,
        pos=pos,
        ax=ax,
        nodelist=list(B.nodes()),
        node_color=node_colors,
        node_size=changable_node_size,
    )

    # Draw labels
    nx.draw_networkx_labels(
        B,
        pos=pos,
        ax=ax,
        font_color="black",
    )

    # legend_items = [
    #         Patch(facecolor="green", label="Source"),
    #         Patch(facecolor="yellow", label="Listening Sensor (Current)"),
    #         Patch(facecolor="orange", label="Listening Sensor (Previous)"),
    #         Patch(facecolor="red", label="Probing Sensor Location"),
    #         Patch(facecolor="pink", label="First Shared Node"),
    #         Patch(facecolor="pink", edgecolor="yellow", linewidth=2, label="Listening Sensor is on Intersection"),
    #         Patch(facecolor="pink", edgecolor="red", linewidth=2, label="Probe Sensor is on Intersection"),
    #         Line2D([0], [0], color='red', linewidth=2, linestyle='-', label='Connections (Solid)'),
    #     ]
    # ax.legend(handles=legend_items, loc="upper left", bbox_to_anchor=(-0.15, 1.15))

    ax.invert_yaxis()
    ax.set_title(f"Probe location: {red_node} | Group: {group_idx}")


    SAVE_DIR.mkdir(exist_ok=True)
    fig.savefig(
    SAVE_DIR / f"group_{group_idx}_no_{count}_probe_{red_node}.png",
    dpi=300,
    bbox_inches="tight",
)



    results = {
        "new_intersection_nodes": list(new_intersection_nodes)
    }

    return results


data, bus_df, switch_df = load_data()
G = add_nodes(bus_df, switch_df)
pos = get_positions()

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
            pos,
            group_idx=group_idx,
            count=count,
        )
        results.append(result)

        # Add this red_node to current_group_processed after processing
        current_group_processed.add(red_node)

    # After processing all nodes in current group, update history with ALL yellow nodes from the group
    # (not just the ones that were probing sensors)
    all_yellow_nodes_history.update(yellow_nodes)
