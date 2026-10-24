# Topology-First B-Rep Meshing

Triangle meshing of B-rep CAD models that preserves the model's topology: every B-rep face becomes a
patch of triangles, every B-rep edge a chain of mesh edges, and every B-rep vertex a mesh vertex.
The pipeline has two stages:

1. **Forward stage** (`main.py`, Python): meshes each face in its parameter domain, traces the face's
   boundary edges into the mesh, aligns the shared edges of adjacent faces, and fuses all patches into one
   conforming mesh (`fused.pkl`).
2. **Remeshing stage** (`remeshing/remesh.py`): remeshes the fused mesh to a target edge length with the
   wildmeshing-toolkit remeshing app, keeping patches, feature chains and corners.

## Install

Python 3.11:

```
pip install -r requirements.txt
```

This installs pygeodesic from a fork, https://github.com/yunfanzhou/pygeodesic (pinned commit), which adds
geodesic tracing between points on faces and edges. It is built from source, so a C++ compiler is needed.
The PyPI release of pygeodesic does not work with this pipeline; `main.py` stops with an ImportError if it
is installed.

Prebuilt macOS arm64 binaries of the two C++ tools the pipeline calls are in `bin/macos-arm64/`:
`mmg2d_O3` (forward stage) and `remeshing_app` (remeshing stage). On other platforms, build them and pass
their paths:

- Mmg: https://github.com/MmgTools/mmg, then `main.py --mmg2d_path <path to mmg2d_O3>`
- wildmeshing-toolkit, branch `fpir`: https://github.com/wildmeshing/wildmeshing-toolkit, configured with
  `-DWMTK_APP_SEC=ON`, target `remeshing_app`; then `remeshing/remesh.py --app <path to remeshing_app>`

## Run

Inputs are CAD models in the ABS HDF5 format ([abs-hdf5](https://pypi.org/project/abs-hdf5/)).
`data/simple/` has 15 small test models.

```
python main.py data/simple/Box.hdf5 out/Box --error_rel 0.05
python remeshing/remesh.py out/Box --len_rel 0.05
python check_topology.py data/simple/Box.hdf5 out/Box
```

`--error_rel` is the forward stage's target geometric error and `--len_rel` the remeshing target edge length,
both relative to the bounding box diagonal. Use the same value for both.

`main.py` meshes the faces in parallel, one process per CPU by default; `--workers N` sets the number of
processes (`--workers 1` runs serially). The output is the same for any number of workers.

`check_topology.py` checks that `fused.pkl` and, if present, `remeshed.vtu` match the B-rep's topology:
every B-rep vertex is one mesh vertex, every B-rep edge a chain of mesh edges between its two corners, and
every B-rep face a patch bordered by the chains of its own edges. It exits with status 1 on a mismatch.

## Output

The forward stage writes `out/<model>/fused.pkl`, a pickled tuple
`(V, F, patch_to_fids, feature_edge_id_to_vids, corner_id_to_vids)`:

- `V`, `F`: vertices and triangles of the fused mesh
- `patch_to_fids`: B-rep face id to its triangle ids
- `feature_edge_id_to_vids`: B-rep edge id to its chain of vertex ids
- `corner_id_to_vids`: B-rep vertex id to its mesh vertex id

plus `result.json` (per-face status and timings). The remeshing stage adds
`remeshed.vtu` (triangles with a per-triangle patch id) and `remeshed_edges.vtu` (feature chains and corners).

## Evaluation scripts

- `get_distance.py`: Hausdorff distance from points sampled on the CAD model to `remeshed.vtu`
- `get_metrics.py`: per-triangle quality (minimum angle, radius ratio)
- `plot_metrics.py`: histograms of the CSVs those two write
- `parse_results.py`: success and timing summary of a batch of `result.json` files

## Tests

```
pytest tests
```

## License

MIT, see `LICENSE`. The binaries in `bin/` are third-party software under their own licenses (Mmg and GMP
are LGPL-3.0); see `third_party/README.md`.
