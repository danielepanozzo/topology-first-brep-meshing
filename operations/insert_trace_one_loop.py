import igl
import numpy as np
import utils.dijkstra as dijkstra
from operations.internal_operations import embed_point_in_triangle, split_edge_in_mesh
from utils.mesh_utils import adjacency_list, compute_VF, create_edge
from utils.print import plot_mesh_with_trace, print_with_time, print_if_verbose
from collections import defaultdict
def project_feature_edge(V, F, feature_edge_polyline_V, feature_edge_orientation=True, myk=8):
    '''Project the feature edge to the mesh.
    Input: feature_edge_to_polyline_V, feature_edge_to_orientations_current_loop
    Output:  #points by k(8) matrix of the closest face ids (assuming regular mesh, this is one ring of triangles)
            note: when k = 1, the return matrix is a 1d array
            Edit 2025.09.19
            Always return a matrix. When there is only one vertex in polyline_V, the return a 1 row matrix
            Edit 2025.12.20
            Return a dictionary to allow varying number of triangles for different points. 
            This change is to facilitate keeping track of the closest triangle candidates with local operations. 
    '''
    feature_V_oriented = feature_edge_polyline_V
    if feature_edge_orientation:
        pass
    else:
        feature_V_oriented = feature_edge_polyline_V[::-1]
    V_np = np.array(V)
    F_np = np.array(F)
    tree, _ = build_barycenter_kdtree(V_np, F_np)
    actual_k = min(myk, len(F_np))
    _, face_indices = tree.query(feature_V_oriented, k=actual_k)

    if isinstance(face_indices, np.ndarray) and face_indices.ndim == 1:
        face_indices = face_indices.reshape(1, -1)
    return face_indices 

def project_feature_edge_in_same_component(V,
                                        F_np,
                                        c,
                                        feature_edge_id_to_polyline_V,
                                        traced_id_set):
    '''Project the feature edge to the mesh, but only within the same connected component.
    We will do a majority vote among the closest faces for all the points of the feature edge. And pick the faces that are in the same component as the majority.
    '''
    # stack all the polyline V together
    curve_poly_V = []
    for feature_edge_id, polyline_V in feature_edge_id_to_polyline_V.items():
        if feature_edge_id in traced_id_set:
            continue
        curve_poly_V.extend(polyline_V)
    if len(curve_poly_V) == 0:
        # only degenerate (empty-polyline) feature edges are left: nothing to project
        return None, None
    V_np = np.array(V)
    tree, _ = build_barycenter_kdtree(V_np, F_np)
    actual_k = min(8, len(F_np))
    _, face_indices = tree.query(curve_poly_V, k=actual_k)
    if isinstance(face_indices, np.ndarray) and face_indices.ndim == 1:
        face_indices = face_indices.reshape(1, -1)
    f_flatten = []
    for face_row in face_indices:
        f_flatten.extend(face_row)

    # components 'c' is a list of components, each component is a list of face indices
    # build a set for each candidate component for quick membership test
    comp_sets = [set(comp) for comp in c]

    # Count how many times faces from each candidate component appear among the k nearest for all points
    comp_counts = [0] * len(comp_sets)
    for row in face_indices:
        for fid in np.atleast_1d(row):
            for ci, comp_set in enumerate(comp_sets):
                if fid in comp_set:
                    comp_counts[ci] += 1

    # choose the component with the maximum count (majority)
    if len(comp_counts) == 0:
        # nothing to do
        return face_indices, None
    chosen_idx = int(np.argmax(comp_counts))
    chosen_set_fids = comp_sets[chosen_idx]
    chosen_fids_list = list(chosen_set_fids)
    tree, _ = build_barycenter_kdtree(V_np, F_np[chosen_fids_list])
    actual_k2 = min(8, len(chosen_fids_list))
    _, face_indices_filtered = tree.query(curve_poly_V, k=actual_k2)
    if isinstance(face_indices_filtered, np.ndarray) and face_indices_filtered.ndim == 1:
        face_indices_filtered = face_indices_filtered.reshape(1, -1)
    return face_indices_filtered, chosen_set_fids

from scipy.spatial import KDTree
def build_barycenter_kdtree(V, F):
    V_np = np.array(V)
    F_np = np.array(F)
    barycenters = np.mean(V_np[F_np], axis=1)  # shape: (n_faces, 3)
    tree = KDTree(barycenters)
    return tree, barycenters

def trace_two_vertices(V, F, 
                       start_vid, 
                       end_vid,
                       tracing_prohibited_vids_set, 
                       adj_list=None, 
                       edge_lengths=None):
    # Trace a path between two vertices
    # Input:
    # V: vertices of the mesh
    # F: faces of the mesh
    # start_vid: the starting vertex id
    # end_vid: the ending vertex id
    # prohibited_vids: the list of vertex ids that are can be traversed
    # Output:
    # trace_vids: the list of vertex ids that are traced
    if edge_lengths is None:
        V_np = np.ascontiguousarray(V, dtype=np.float64)
        F_np = np.ascontiguousarray(F, dtype=np.int64)
        edge_lengths = igl.edge_lengths(V_np, F_np)
    if adj_list is None:
        adj_list = adjacency_list(F)
    trace_vids, distance = dijkstra.dijkstra_shortest_path_to_single_target(edge_lengths, adj_list, start_vid, end_vid, tracing_prohibited_vids_set)
    return trace_vids, distance



def collapse_nonmanifold_prevention(V, F, boundary_vids_set, adj_list, edge_set_to_length, adj_modified_copy_dict, modified_edge_set):
    ''' checking a condition more strict than the link condition.
    Specifically, it will check if any the will edit triangles share the edge that is opposite to the apex'''
    # will_delete_edges = defaultdict(list)
    will_edit_edges = defaultdict(list)
    for i in range(len(F)):
        triangle = F[i]
        
        if len(set(triangle) & boundary_vids_set) == 1:
            for j in range(3):
                if triangle[j] in boundary_vids_set:
                    will_edit_edges[i]= create_edge(triangle[(j -1)%3], triangle[(j + 1) % 3])
                    break
    vertex_to_face = compute_VF(F)
    # check link condition for the will delete edges. If fails, split one of the neighboring edge
    edges_to_split = []
    # for f, edge in will_delete_edges.items():  # Updated to use .items() for dictionary iteration
    #     if link_condition(edge, adj_list, is_boundary_edge=True):
    #         continue
    #     # we will split edges that are in the link that's not in the current triangle
    #     neighbors0 = adj_list[edge[0]]
    #     neighbors1 = adj_list[edge[1]]
    #     neighbors_intersect = set(neighbors0) & set(neighbors1) 
    #     # remove the current triangle from the intersection
    #     neighbors_intersect -= F[f]
    #     print(f" for edge {edge} neighbors_intersect: {neighbors_intersect}")
    #     for neighbor in neighbors_intersect:
    #         edges_to_split.extend([[edge[0], neighbor], [edge[1], neighbor]]) 

    # we also need to check the will edit triangles share edges that are opposite to the apex
    opposite_edges_set = set()
    for f, edge in will_edit_edges.items():
        # edge_set = frozenset(edge)
        if edge not in opposite_edges_set:
            opposite_edges_set.add(edge)
            continue
        face = F[f]
        for j in range(3):
            if face[j] in boundary_vids_set:
                edges_to_split.append([face[j], edge[0]])
                edges_to_split.append([face[j], edge[1]])
    for edge in edges_to_split:
        add_to_bnd_vids = False
        v0 = edge[0]
        v1 = edge[1]
        if v1 in adj_list[v0]:
            if len(set(vertex_to_face[v0]) & set(vertex_to_face[v1])) == 1:
                # the edge is a boundary edge, we need to add the new vid to the boundary_vid list
                add_to_bnd_vids = True
            split_edge_in_mesh(V, F, edge, vertex_to_face, new_point= None, faces=None, 
                                adjacency_list=adj_list, 
                                edge_mapping_to_length=edge_set_to_length, 
                                adj_list_copy_dict=adj_modified_copy_dict, 
                                modified_edge_set=modified_edge_set)
        if add_to_bnd_vids:
            boundary_vids_set.add(len(V) - 1)
    
def trace_to_find_best_candidate(V,
                                 connectivity,
                                 prev_vid,
                                 point,
                                 patch_regular_space_data,
                                 candidate_fids,
                                 off_boundary):
    current_vid, _ = embed_point_in_triangle(V,
                                             connectivity,
                                             point,
                                             patch_regular_space_data.frozen_edges_set,
                                             patch_regular_space_data.prohibited_vids_set,
                                             candidate_fids=candidate_fids,
                                             off_boundary=off_boundary,
                                             prev_vid=prev_vid,
                                             traced_edge_set=patch_regular_space_data.traced_edge_set)
    path_vids, dist = trace_two_vertices(V,
                                         connectivity.F,
                                         prev_vid,
                                         current_vid,
                                         patch_regular_space_data.prohibited_vids_set,
                                         connectivity.adj_list,
                                         connectivity.edge_set_to_length_mapping)
    if dist < float('inf'):
        return current_vid, path_vids, dist
    print_with_time("No valid path found among closest candidates ")
    raise ValueError("No valid path found among closest candidates")
