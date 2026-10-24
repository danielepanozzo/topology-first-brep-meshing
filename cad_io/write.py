import os
import pickle
import tempfile
def output_fused_mesh(output_folder, V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids):
    file_path_hdf5 = os.path.join(output_folder, f"fused.hdf5")
    file_path_pkl = os.path.join(output_folder, f"fused.pkl")
    if os.path.exists(file_path_hdf5):
        os.remove(file_path_hdf5)
    if os.path.exists(file_path_pkl):
        os.remove(file_path_pkl)

    # automic save pkl
    atomic_save((V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids), file_path_pkl)

    print(f"Data saved at: {file_path_pkl}") # and {file_path_hdf5}")
    # only save pkl for now to save space. 


def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=os.path.dirname(path)) as tmp:
        pickle.dump(obj, tmp)
        tmp_path = tmp.name
    os.replace(tmp_path, path)