import argparse, os

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--align", type=int, default=16)
ap.add_argument("models", nargs="+")
args = ap.parse_args()

entries = []
blob = bytearray()
for path in args.models:
    # align current position
    pad = (-len(blob)) % args.align
    blob.extend(b"\x00" * pad)
    offset = len(blob)
    data = open(path, "rb").read()
    blob.extend(data)
    entries.append((os.path.basename(path), offset, len(data)))

with open(args.out, "wb") as f:
    f.write(blob)

with open(args.manifest, "w") as f:
    f.write("#pragma once\n")
    f.write("#include <stdint.h>\n\n")
    f.write(f"#define MODEL_COUNT {len(entries)}\n\n")
    f.write("static const uint32_t MODEL_OFFSETS[MODEL_COUNT] = {\n")
    for name, off, sz in entries:
        f.write(f"    {off}, // {name} ({sz} bytes)\n")
    f.write("};\n\n")
    f.write("static const uint32_t MODEL_SIZES[MODEL_COUNT] = {\n")
    for name, off, sz in entries:
        f.write(f"    {sz}, // {name}\n")
    f.write("};\n")