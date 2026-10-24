# Third-party software

`bin/macos-arm64/` ships prebuilt macOS arm64 binaries so the pipeline runs without building C++ code.
The MIT license of this repository does not cover them; each component keeps its own license, copied
into this folder.

| Binary | Component | License | Linking | Source |
|---|---|---|---|---|
| `mmg2d_O3` | Mmg (commit 4d8232c8) | LGPL-3.0 | executable | https://github.com/MmgTools/mmg |
| `remeshing_app` | wildmeshing-toolkit, branch `fpir` (commit ed57e12061) | MIT | executable | https://github.com/wildmeshing/wildmeshing-toolkit |
| `libgmp.10.dylib` | GNU MP 6.3.0 | LGPL-3.0 / GPL-2.0 | dynamic, loaded by `remeshing_app` | https://gmplib.org |

Libraries statically linked into `remeshing_app`:

| Library | License | File |
|---|---|---|
| Indirect Predicates (IMATI-GE / CNR) | LGPL-3.0 | `indirect_predicates.LGPL-3.txt` |
| FastEnvelope | MIT | `fast_envelope.LICENSE` |
| Geogram Delaunay PSM | BSD-3-Clause | `geogram_delaunay_psm.LICENSE` |
| Shewchuk predicates | public domain | none |
| GKlib, METIS | Apache-2.0 | `gklib.LICENSE`, `metis.LICENSE` |
| HDF5 | BSD-style | `hdf5.COPYING` |
| Lagrange | Apache-2.0 | `lagrange.LICENSE` |
| oneTBB | Apache-2.0 | `tbb.LICENSE` |
| MshIO | Apache-2.0 | `mshio.LICENSE` |
| libigl (with Eigen, header only) | MPL-2.0 | `libigl.LICENSE.MPL2` |
| jse, paraviewo, simple_bvh | MIT | `jse.LICENSE`, `paraviewo.LICENSE`, `simple_bvh.LICENSE` |
| spdlog | MIT | `spdlog.LICENSE` |
| TinyXML-2 | zlib | `tinyxml2.LICENSE` |

The LGPL components can be rebuilt from the sources above: `remeshing_app` is the `remeshing_app`
target of wildmeshing-toolkit built with `-DWMTK_APP_SEC=ON`, and `libgmp.10.dylib` can be replaced by any
compatible GMP build placed next to it.

`remeshing/convert_pkl_to_wmtk.py` is copied from wildmeshing-toolkit (MIT, `wildmeshing-toolkit.LICENSE`).
