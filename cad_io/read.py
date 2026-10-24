import h5py
from utils.print import plot_mesh_with_trace
import numpy as np
from collections import defaultdict
from utils.data_structures import LoopGeometryData, PatchInputData
from utils.print import print_with_time, print_if_verbose
import pickle
def part_index_to_part_id_mapping(topology_file_path, random_part_index):
    """Maps part indices to part IDs."""
    with h5py.File(topology_file_path, "r") as file:
        if "parts" not in file:
            print("No 'parts' group found in the file.")
            return
        parts_group = file["parts"]
        part_ids = list(parts_group.keys())
    return part_ids[random_part_index]

def load_topology_hdf5_patches(topology_file_path, random_part=None):
    """Loads the patches from the HDF5 file."""
    with h5py.File(topology_file_path, "r") as file:
        if "parts" not in file:
            print("No 'parts' group found in the file.")
            return
        
        parts_group = file["parts"]
        parts_patches = []
        for part_id in parts_group:
            if random_part is not None:
                part_id = random_part
            part_group = parts_group[part_id]
            if "topology" not in part_group:
                print(f"No 'topology' group found in part {part_id}.")
                continue
            face_group = part_group["topology/faces"]
            for face_id in face_group:
                parts_patches.append(int(face_id))
            break
    return parts_patches
# ==================================================
# ==================================================

def load_topology_hdf5_outer_loop_id(topology_file_path, random_part=None):
    with h5py.File(topology_file_path, "r") as file:
        if "parts" not in file:
            print("No 'parts' group found in the file.")
            return
        
        parts_group = file["parts"]
        patch_id_to_outer_loop_id = defaultdict(int)
        for part_id in parts_group:
            if random_part is not None:
                part_id = random_part
            part_group = parts_group[part_id]
            if "topology" not in part_group:
                print(f"No 'topology' group found in part {part_id}.")
                continue
            topology = part_group["topology"]
            face_group = topology["faces"]
            for face_id in face_group:
                face = face_group[f"{face_id}"]
                outer_loop_id = np.array(face["outer_loop"])
                patch_id_to_outer_loop_id[int(face_id)] = int(outer_loop_id)
            break
    return patch_id_to_outer_loop_id

def load_patch_loop_geometry_data(feature_polylines, loop_halfedges):
    '''
    For each loop, load a unique edge_feature_id corresponds to the polyline_V of the upsampled edge oriented in True.
    Degenerate edges are also loaded, with polyline_V size = 0. 
    '''
    loop_geometry_data = LoopGeometryData()
    for halfedge in loop_halfedges:
        feature_id = halfedge.feature_id
        if feature_id in loop_geometry_data.feature_id_poly_V_True_orientation.keys():
            continue
        polyline_V = np.array(feature_polylines[feature_id])
        loop_geometry_data.feature_id_poly_V_True_orientation[feature_id] = polyline_V
   
    return loop_geometry_data

def load_patch_input_data(patch_group, feature_polylines, global_loop_topology_data):
    V = np.array(patch_group["V"]).tolist()
    F = np.array(patch_group["F"]).tolist()
    initial_loop_id = patch_group.get("initial_loop_id", None)

    # Load loop geometry and topology data for each loop

    if "loops" not in patch_group:
        raise ValueError("No 'loops' found in the patch group.")
    patch_loop_geometry_data = {}
    patch_loop_topology_data = {}
    for loop_id in patch_group["loops"]:
        patch_loop_topology_data[int(loop_id)] = global_loop_topology_data[int(loop_id)]
        patch_loop_geometry_data[int(loop_id)] = load_patch_loop_geometry_data(feature_polylines, patch_loop_topology_data[int(loop_id)].halfedges)    

    # Extract periodic edges
    u_periodic_edges = ([], [])  # (side0, side1)
    v_periodic_edges = ([], [])  # (side0, side1)
    if "u_periodic_edges" in patch_group:
        read_u_periodic_edges = np.array(patch_group["u_periodic_edges"])
        for i in range(len(read_u_periodic_edges)):
            vv = read_u_periodic_edges[i]
            u_periodic_edges[0].append(vv[0])
            u_periodic_edges[1].append(vv[1])

    if "v_periodic_edges" in patch_group:
        read_v_periodic_edges = np.array(patch_group["v_periodic_edges"])
        for i in range(len(read_v_periodic_edges)):
            vv = read_v_periodic_edges[i]
            v_periodic_edges[0].append(vv[0])
            v_periodic_edges[1].append(vv[1])

    # Create and return PatchInputData instance
    return PatchInputData(
        V=V,
        F=F,
        initial_loop_id=initial_loop_id,
        patch_loop_geometry_data=patch_loop_geometry_data,
        patch_loop_topology_data=patch_loop_topology_data,
        u_periodic_edges=u_periodic_edges,
        v_periodic_edges=v_periodic_edges
    )

def load_input_data(input_file_path, outer_loop_id, model_loop_topology_data):
    """Loads the input data from the HDF5 file."""
    print_if_verbose(f"Loading input data from {input_file_path}")
    with h5py.File(input_file_path, "r") as file:
        
        if "meshes" not in file:
            print("No 'meshes' group found in the file.")
            return
        patch_to_input_data = {}
        feature_polylines = dict()
        for feature_id in file["feature_polylines"]:
            feature_polylines[int(feature_id)] = np.array(file["feature_polylines"][feature_id])
        # Iterate over all meshes
        for patch_id in file["meshes"]:
            patch_id = int(patch_id)
            mesh_group = file[f"meshes/{patch_id}"]
            try:
                patch_to_input_data[patch_id] = load_patch_input_data(mesh_group, 
                                                                      feature_polylines, 
                                                                      model_loop_topology_data[patch_id])
            except:
                raise ValueError(f"Error loading patch {patch_id} from {input_file_path}.")
            patch_to_input_data[patch_id].initial_loop_id = outer_loop_id[patch_id]
    return patch_to_input_data


def load_fused_output(fused_file_path):
    """Loads the fused output data from the HDF5 file."""
    print_if_verbose(f"Loading fused output data from {fused_file_path}")
    with open(fused_file_path, 'rb') as file:
        data = pickle.load(file)


        V = data[0]
        F = data[1]

        # key: is the face id, value: list of triangle ids
        patch2fid = data[2]

        # key: edge id, value: list of vertex ids forming a polygon
        edge2vidschain = data[3]
        # print(edge2vidschain[588])

        #
        corner2vids = data[4]
        new_corner_id_mapping = data[5] if len(data) > 5 else None
        print_if_verbose(f"Loaded new_corner_id_mapping: {new_corner_id_mapping}")
        return V, F, patch2fid, edge2vidschain, corner2vids, new_corner_id_mapping
    
def load_intermediate_pkl(file_path):
    """Loads intermediate data from a pickle file."""
    print_if_verbose(f"Loading intermediate data from {file_path}")
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

def atomic_load(path):
    with open(path, "rb") as f:
        return pickle.load(f)