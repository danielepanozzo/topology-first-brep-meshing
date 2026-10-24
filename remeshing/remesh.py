"""Remeshing stage: converts the forward stage's fused.pkl and runs the wildmeshing-toolkit remeshing app.

usage: python remeshing/remesh.py <output_folder> [--len_rel 0.05] [--app PATH]

<output_folder> is the folder main.py wrote fused.pkl into. Writes mesh.msh, features.json and remesh.json
there (convert_pkl_to_wmtk.py), then remeshed.vtu and remeshed_edges.vtu (remeshing_app).
--len_rel is the target edge length relative to the bounding box diagonal; use the --error_rel of the
forward run. The bundled app is macOS arm64; on other platforms build remeshing_app and pass --app.
"""
import argparse
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLED_APP = os.path.join(HERE, "..", "bin", "macos-arm64", "remeshing_app")


def main():
    parser = argparse.ArgumentParser(description="Remesh the forward stage output with wildmeshing-toolkit.")
    parser.add_argument("output_folder", type=str, help="Folder containing fused.pkl from main.py")
    parser.add_argument("--len_rel", type=float, default=0.05,
                        help="Target edge length relative to the bounding box diagonal (default: 0.05)")
    parser.add_argument("--app", type=str, default=None, help="Path to remeshing_app (default: the bundled binary)")
    args = parser.parse_args()

    app = args.app
    if app is None:
        if platform.system() != "Darwin" or platform.machine() != "arm64":
            sys.exit("The bundled remeshing_app is macOS arm64 only; build it (see README) and pass --app.")
        app = BUNDLED_APP
    out = os.path.abspath(args.output_folder)
    if not os.path.exists(os.path.join(out, "fused.pkl")):
        sys.exit(f"No fused.pkl in {out}; run main.py first.")

    subprocess.run([sys.executable, os.path.join(HERE, "convert_pkl_to_wmtk.py"), out,
                    "--len_rel", str(args.len_rel)], check=True)
    subprocess.run([app, "-j", os.path.join(out, "remesh.json")], check=True)
    print("remeshed mesh written to", os.path.join(out, "remeshed.vtu"))


if __name__ == "__main__":
    main()
