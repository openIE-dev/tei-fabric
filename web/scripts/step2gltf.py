#!/usr/bin/env python3
"""Local STEP → glTF/GLB conversion (no network, no gateway timeout).

The robust "any board" path: Zoo's cloud conversion is synchronous and
Cloudflare-gateway-capped (~100 s), so heavy board assemblies like the
full Raspberry Pi Pico STEP 504 even though the service is healthy. This
converts on-machine with cadquery (OpenCascade), which has no such cap.

    python3 step2gltf.py <input.step> <output.glb>

Requires: pip install cadquery
"""
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: step2gltf.py <input.step> <output.glb|.gltf>", file=sys.stderr)
        return 2
    inp, out = sys.argv[1], sys.argv[2]
    try:
        import cadquery as cq
    except ImportError:
        print("cadquery not installed — run: pip install cadquery", file=sys.stderr)
        return 3
    # Coarse tessellation for WEB delivery. The default 0.1 mm linear
    # deflection produces tens-of-MB meshes from a full board assembly
    # (every fillet at micron detail) — unusable in a browser. 0.4 mm /
    # 0.5 rad keeps the board readable at a fraction of the size. Override
    # via argv[3]/argv[4].
    tol = float(sys.argv[3]) if len(sys.argv) > 3 else 0.4
    ang = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
    shape = cq.importers.importStep(inp)
    assy = cq.Assembly(shape)
    assy.export(out, exportType="GLTF", tolerance=tol, angularTolerance=ang)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
