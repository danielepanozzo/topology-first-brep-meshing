"""Remeshing stage: remeshes the forward stage's fused.pkl with wildmeshing-toolkit's isotropic remeshing.

usage: python remeshing/remesh.py <output_folder> [--len_rel 1e-3] [--threads 0]

<output_folder> is the folder main.py wrote fused.pkl into. Writes the remeshing input there
(remesh_input.msh: the fused mesh with its patch ids; remesh_features.json: its feature chains and
corners), runs the isotropic remeshing of the wildmeshing package (requirements.txt), which keeps
every patch, chain and corner, and writes remeshed.vtu and remeshed_edges.vtu (see README).
--len_rel is the target edge length relative to the bounding box diagonal; use the --error_rel of the
forward run.
"""
import argparse
import json
import os
import pickle

import numpy as np


def load_fused(path):
    """V, F, patch id per triangle, B-rep edge -> chain of vertices, B-rep vertex -> vertex."""
    with open(path, "rb") as f:
        V, F, patch2fids, edge2chain, corner2vid = pickle.load(f)[:5]
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    patch = np.full(len(F), -1, dtype=np.int64)
    for p, fids in patch2fids.items():
        patch[np.asarray(list(fids), dtype=np.int64)] = int(p)
    if (patch < 0).any():
        raise ValueError(f"{path}: {(patch < 0).sum()} triangles have no patch")
    return V, F, patch, edge2chain, corner2vid


def write_msh(path, V, F, patch):
    """ASCII MSH 4.1 as wildmeshing-toolkit writes a surface: one node block, one triangle block, and
    the patch ids as a per-triangle field."""
    n, m = len(V), len(F)
    with open(path, "w") as f:
        f.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
        f.write(f"$Nodes\n1 {n} 1 {n}\n2 1 0 {n}\n")
        f.write("".join(f"{i + 1}\n" for i in range(n)))
        f.write("".join("%.17g %.17g %.17g\n" % tuple(p) for p in V))
        f.write("$EndNodes\n")
        f.write(f"$Elements\n1 {m} 1 {m}\n2 1 2 {m}\n")
        f.write("".join(f"{i + 1} {t[0] + 1} {t[1] + 1} {t[2] + 1}\n" for i, t in enumerate(F)))
        f.write("$EndElements\n")
        f.write(f'$ElementData\n1\n"patch_id"\n1\n0.0\n5\n0\n1\n{m}\n0\n2\n')
        f.write("".join(f"{i + 1} {p}\n" for i, p in enumerate(patch)))
        f.write("$EndElementData\n")


def features(edge2chain, corner2vid):
    """The remeshing's features file: every chain edge with its B-rep edge id, and every corner as a
    fixed vertex with its B-rep vertex id. Zero-length B-rep edges (one-vertex chains) add nothing."""
    edges, edge_ids = [], []
    for eid, chain in sorted(edge2chain.items()):
        for a, b in zip(chain[:-1], chain[1:]):
            edges.append([int(a), int(b)])
            edge_ids.append(int(eid))
    vertices = [[int(v), int(c)] for c, v in sorted(corner2vid.items())]
    return {"edges": edges, "edge_ids": edge_ids, "vertices": vertices}


def main():
    parser = argparse.ArgumentParser(description="Remesh the forward stage output with wildmeshing-toolkit.")
    parser.add_argument("output_folder", type=str, help="Folder containing fused.pkl from main.py")
    parser.add_argument("--len_rel", type=float, default=1e-3,
                        help="Target edge length relative to the bounding box diagonal (default: 1e-3, "
                             "the default --error_rel of main.py)")
    parser.add_argument("--threads", type=int, default=0,
                        help="Threads of the remeshing (default: 0, serial and deterministic)")
    args = parser.parse_args()

    out = os.path.abspath(args.output_folder)
    fused = os.path.join(out, "fused.pkl")
    if not os.path.exists(fused):
        raise SystemExit(f"No fused.pkl in {out}; run main.py first.")
    V, F, patch, edge2chain, corner2vid = load_fused(fused)
    write_msh(os.path.join(out, "remesh_input.msh"), V, F, patch)
    with open(os.path.join(out, "remesh_features.json"), "w") as f:
        json.dump(features(edge2chain, corner2vid), f)

    # The envelope that the earlier remeshing app used: half the input's average edge length.
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
    eps_rel = 0.5 * float(np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()) / diag

    import wildmeshing

    wildmeshing.wildmeshing({
        "application": "isotropic_remeshing",
        "input": "remesh_input.msh",
        "input_dir": out,
        "output": os.path.join(out, "remeshed.vtu"),
        "features": "remesh_features.json",
        "feature_fields": ["patch_id"],
        "length_rel": args.len_rel,
        "eps_rel": eps_rel,
        "use_sample_envelope": True,
        "max_iterations": 3,
        "check_inversions": True,
        "swap_quality": True,
        "limit_collapse_length": True,  # no collapse leaves an edge over 4/3 of the target
        "cleanup_angle": 40.0,  # repair the triangles under 40 degrees after every iteration
        "freeze_boundary": False,  # an open boundary is B-rep edges, which stay as curves
        "remove_duplicate_eps": -1.0,  # the features file refers to the input's vertex ids
        "num_threads": args.threads,
        "report": os.path.join(out, "remeshed_report.json"),
    })
    print("remeshed mesh written to", os.path.join(out, "remeshed.vtu"))


if __name__ == "__main__":
    main()
