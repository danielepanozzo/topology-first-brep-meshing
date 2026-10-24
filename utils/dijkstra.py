from copy import deepcopy
import heapq
import numpy as np
from utils.mesh_utils import create_edge
from collections import deque
from utils.print import plot_mesh_with_trace, print_if_verbose
def bfs(adj_list, source_vid, target_vid, prohibited_vids: set = None):
    prohibited_contains_source = False
    prohibited_contains_target = False
    if prohibited_vids is not None:
        if source_vid in prohibited_vids:
            prohibited_contains_source = True
            prohibited_vids.remove(source_vid)
        if target_vid in prohibited_vids:
            prohibited_contains_target = True
            prohibited_vids.remove(target_vid)
    visited = set()
    parent = {}
    queue = deque([source_vid])
    visited.add(source_vid)

    while queue:
        current = queue.popleft()
        if current == target_vid:
            break
        for neighbor in adj_list[current]:
            if neighbor in visited or (prohibited_vids is not None and neighbor in prohibited_vids):
                continue
            visited.add(neighbor)
            parent[neighbor] = current
            queue.append(neighbor)

    # Step 3: Reconstruct the path
    if target_vid not in parent:
        print("No path found between source_vid and target_vid in the prohibited region.")
        if prohibited_contains_source:
            prohibited_vids.add(source_vid)
        if prohibited_contains_target:
            prohibited_vids.add(target_vid)
        return []

    path = [target_vid]
    while path[-1] != source_vid:
        path.append(parent[path[-1]])
    path.reverse()
    if prohibited_contains_source:
        prohibited_vids.add(source_vid)
    if prohibited_contains_target:
        prohibited_vids.add(target_vid)

    return path

def dijkstra_construct_network(edge_lengths, adj_list, source_vid, target_vid, prohibited_vids):
    """
    Finds the shortest path from source_vid to target_vid using Dijkstra's algorithm,
    skipping prohibited vertices.

    Parameters:
    - edge_lengths: (M, 3) array of edge lengths, where each row corresponds to edges [1,2], [2,0], [0,1] of F.
    - adj_list: List of lists, where adj_list[i] contains the adjacent vertices of vertex i.
    - source_vid: Start vertex ID.
    - target_vid: End vertex ID.
    - prohibited_vids: Set of vertex IDs that should not be visited.

    Returns:
    - List of vertices representing the shortest path, or an empty list if unreachable.
    """
    pq = []  # Min-heap priority queue
    distance = {v: np.inf for v in range(len(adj_list))}
    parent = {v: -1 for v in range(len(adj_list))}
    visited = set()

    # Initialize source
    distance[source_vid] = 0.0
    heapq.heappush(pq, (0.0, source_vid))

    while pq:
        current_distance, current_vid = heapq.heappop(pq)
        if current_vid in visited:
            continue
        # if current_vid != source_vid and current_vid in prohibited_vids:
        #     continue
        visited.add(current_vid)
        # Stop if we reached the target if target_vid is not -1
        if target_vid != -1 and current_vid == target_vid:
            break

        # Explore neighbors
        for neighbor_vid in adj_list[current_vid]:
            if neighbor_vid in visited or neighbor_vid in prohibited_vids:
                continue

            # Find edge length from F and edge_lengths
            # edge = frozenset([current_vid, neighbor_vid])
            edge = create_edge(current_vid, neighbor_vid)
            edge_length = edge_lengths.get(edge)

            if edge_length is None:
                print_if_verbose(f"adj[{current_vid}]:", adj_list[current_vid])
                print_if_verbose(f"adj[{neighbor_vid}]:", adj_list[neighbor_vid])
                print_if_verbose(f"     dijkstra_construct_network: Edge {edge} not found in edge_lengths")
                raise Exception(f"Edge {edge} not found in edge_lengths")
                continue  # Edge not found, skip

            # Relaxation step
            new_distance = current_distance + edge_length
            if new_distance < distance[neighbor_vid]:
                distance[neighbor_vid] = new_distance
                parent[neighbor_vid] = current_vid
                heapq.heappush(pq, (new_distance, neighbor_vid))
    return distance, parent
def dijkstra_shortest_path_to_single_target(edge_lengths, adj_list, source_vid, target_vid, prohibited_vids):
    # keep track of prohibited vids so that we add the source and target vids back if removed 
    prohibited_contains_source = False
    prohibited_contains_target = False
    if source_vid in prohibited_vids:
        prohibited_contains_source = True
        prohibited_vids.remove(source_vid)
    if target_vid in prohibited_vids:
        prohibited_contains_target = True
        prohibited_vids.remove(target_vid)
    distance, parent = dijkstra_construct_network(edge_lengths, adj_list, source_vid, target_vid, prohibited_vids)
    # If target is unreachable
    if distance[target_vid] == np.inf:
        print(f"No path found from {source_vid} to {target_vid}")
        # Add back the source and target vids to prohibited vids if they were originally present
        if prohibited_contains_source:
            prohibited_vids.add(source_vid)
        if prohibited_contains_target:
            prohibited_vids.add(target_vid)
        return [], np.inf
    # Reconstruct the path
    path = reconstruct_shortest_path(parent, target_vid)
    # Add back the source and target vids to prohibited vids if they were originally present
    if prohibited_contains_source:
        prohibited_vids.add(source_vid)
    if prohibited_contains_target:
        prohibited_vids.add(target_vid)
    return path, distance[target_vid]

def reconstruct_shortest_path(parent, target_vid):
    # Reconstruct the path
    path = []
    v = target_vid
    while v != -1:
        path.append(v)
        v = parent[v]
    path.reverse()
    return path


def bfs_reachable(adj_list, start_vertices, prohibited_vids):
    """
    adj_list: List[List[int]]
        Vertex adjacency list.
    start_vertices: set[int]
        BFS sources (e.g. loop_A).
    prohibited_vids: set[int]
        Vertices that cannot be traversed.
    Returns:
        set[int]: all reachable vertices.
    """
    visited = set()
    q = deque()

    for v in start_vertices:
        if v in prohibited_vids:
            continue
        visited.add(v)
        q.append(v)

    while q:
        u = q.popleft()
        for n in adj_list[u]:
            if n in visited or n in prohibited_vids:
                continue
            visited.add(n)
            q.append(n)

    return visited


def dijkstra_shortest_path_between_loops(edge_lengths, adj_list, loop_A, loop_B, prohibited_vids=None, frozen_edge_set=None):
    if prohibited_vids is None:
        prohibited_vids = set()
    else:
        prohibited_vids = set(prohibited_vids)
    local_prohibited_vids_copy = deepcopy(prohibited_vids)
    local_prohibited_vids_copy.difference_update(loop_B)
    local_prohibited_vids_copy.difference_update(loop_A)
    # Initialize distances and parent tracking
    distance = {v: np.inf for v in range(len(adj_list))}
    parent = {v: None for v in range(len(adj_list))}    
    visited = set()

    # Priority queue: (distance, vid)
    heap = []

    for vid in loop_A:
        if vid not in local_prohibited_vids_copy:
            distance[vid] = 0
            heapq.heappush(heap, (0, vid))

    while heap:
        dist_u, u = heapq.heappop(heap)
        if u in visited:
            continue
        visited.add(u)

        # Early exit if we reach a vertex in loop_B
        if u in loop_B:
            path = []
            current = u
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return path

        for v in adj_list[u]:
            if v in visited or v in local_prohibited_vids_copy:
                continue
            edge_key = create_edge(u, v)
            edge_len = edge_lengths.get(edge_key)
            if edge_len is None:
                print_if_verbose(f"dijkstra_shortest_path_between_loops: Edge {edge_key} not found in edge_lengths")
                raise Exception(f"Edge {edge_key} not found in edge_lengths")
                continue  # Edge not found, skip
            alt = dist_u + edge_len
            if alt < distance[v]:
                distance[v] = alt
                parent[v] = u
                heapq.heappush(heap, (alt, v))

    # If we reach here, no path found
    print("No path found between loops")
    # DEBUG


    reachable = bfs_reachable(
        adj_list,
        set(loop_A),
        prohibited_vids - set(loop_A) - set(loop_B)
    )

    print("reachable size:", len(reachable))
    print("blocked neighbors:")
    set_unblocking = set()
    for v in reachable:
        for n in adj_list[v]:
            if n not in reachable:
                if frozen_edge_set is not None and create_edge(v, n) not in frozen_edge_set:
                    set_unblocking.add(create_edge(v, n))
    print("unblocking edges number:", len(set_unblocking))
    # print("try BFS instead")
    # return find_any_path_between_loops(adj_list,
    #                                               loop_A,
    #                                               loop_B,
    #                                               prohibited_vids,
    #                                               frozen_edge_set)
    return []
