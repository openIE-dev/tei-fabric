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
    shape = cq.importers.importStep(inp)
    # wrap in an Assembly so the glTF/GLB exporter (format inferred from the
    # extension) has something to serialize.
    assy = cq.Assembly(shape)
    assy.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
