#!/usr/bin/env python3
# ============================================================================
# inspect_artifact.py -- read a .litertlm (or a bare .tflite) and print what
# actually determines on-device RAM.
#
# WHY. Disk size was only ever a proxy. The binding constraint is the in-app
# budget measured on the SHIPPED E4B (32k ring, STADIUM):
#
#     weights 1942 + INT2 lm_head 160 + global KV 372 + local KV 44
#     + arena 219 + staging 20  =  2757 MB   (app RSS 2811 MB)
#
# Three of those terms are decided by the exported artifact and are readable
# from it without a device:
#   * global vs local KV  -- the per-layer kv_cache_* shapes. A SYMMETRIC export
#     gives every sliding layer the full cache_length, which is what pushes
#     local KV from 44 MB to global width. Asymmetry is an export flag
#     (--sliding_window_ring_buffer_size), not graph surgery: cache.py:404-410
#     takes it for `sliding_attention` layers, and metadata_builder.py demotes
#     those layers to 'full_attention' when it is None.
#   * weights + lm_head -- the per-tensor quantised dtypes. Ours are int4 where
#     Google ships int2 for the embedder, which is a ~2x section.
#   * what is present at all -- the section list (drafter, vision, audio).
#
# Runs on the runner, so nothing has to be downloaded to be inspected.
#
# Usage:
#   python inspect_artifact.py model.litertlm [--full]
#   python inspect_artifact.py model_quantized.tflite
# ============================================================================
import argparse
import collections
import os
import sys

SECTION_TYPE = {
    0: "NONE", 1: "GenericBinaryData", 2: "Deprecated", 3: "TFLiteModel",
    4: "SP_Tokenizer", 5: "LlmMetadataProto", 6: "HF_Tokenizer_Zlib",
    7: "TFLiteWeights",
}
# tflite TensorType -> (name, bits). Bits are what the quantiser actually packs.
TT = {0: ("FLOAT32", 32), 1: ("FLOAT16", 16), 2: ("INT32", 32), 3: ("UINT8", 8),
      4: ("INT64", 64), 5: ("STRING", 0), 6: ("BOOL", 8), 7: ("INT16", 16),
      8: ("COMPLEX64", 64), 9: ("INT8", 8), 10: ("FLOAT64", 64),
      11: ("COMPLEX128", 128), 12: ("UINT64", 64), 13: ("RESOURCE", 0),
      14: ("VARIANT", 0), 15: ("UINT32", 32), 16: ("UINT16", 16),
      17: ("INT4", 4), 18: ("BFLOAT16", 16), 19: ("INT2", 2)}

import re
KVRE = re.compile(r"kv_cache_([kv])_(\d+)")
KV_KEYS = ("kv_cache_", "mask", "pos", "logits", "param_tensor",
           "embeddings", "per_layer")


def _tt(code):
    return TT.get(code, (f"T{code}", 0))


def dump_tflite(buf, label, full=False):
    from ai_edge_quantizer.utils import tfl_flatbuffer_utils
    m = tfl_flatbuffer_utils.read_model(buf)
    print(f"\n=== {label}: {len(m.subgraphs)} subgraph(s) ===")

    sig = {}
    for s in (m.signatureDefs or []):
        key = s.signatureKey.decode() if isinstance(s.signatureKey, bytes) else str(s.signatureKey)
        sig.setdefault(s.subgraphIndex, []).append(key)

    # --- 1. the KV geometry, which is the whole point ---------------------
    for i, sg in enumerate(m.subgraphs):
        name = ", ".join(sig.get(i, [])) or (
            sg.name.decode() if getattr(sg, "name", None) else f"subgraph{i}")
        rows = []
        for t in sg.tensors:
            nm = t.name.decode() if isinstance(t.name, bytes) else str(t.name)
            short = nm.split("/")[-1].split(";")[0]
            if any(k in short for k in KV_KEYS):
                rows.append((short, tuple(t.shape or ()), _tt(int(t.type))[0]))
        if not rows:
            continue
        print(f"\n-- signature: {name}")
        # Collapse kv_cache_k_/v_ to one line per distinct shape, with the layer
        # list -- 42 layers of identical shapes is noise, the SHAPE SET is signal.
        kv = collections.OrderedDict()
        other = []
        for short, shape, ty in rows:
            if KVRE.search(short):   # NOT startswith: ours are prefill_128_kv_cache_k_0
                kv.setdefault((shape, ty), []).append(short)
            else:
                other.append((short, shape, ty))
        for (shape, ty), names in kv.items():
            ks = sorted(n for n in names if "_k_" in n)
            vs = sorted(n for n in names if "_v_" in n)
            idx = sorted({n.rsplit("_", 1)[-1] for n in names}, key=lambda x: int(x) if x.isdigit() else -1)
            print(f"   kv_cache  shape={shape} {ty:<8} x{len(names)} "
                  f"(K:{len(ks)} V:{len(vs)}) layers={','.join(idx)}")
        for short, shape, ty in sorted(set(other)):
            print(f"   {short:<28} shape={shape} {ty}")

    # --- 2. the quantisation census ---------------------------------------
    print(f"\n-- weight census (constant tensors with a buffer)")
    by = collections.Counter()
    nbytes = collections.Counter()
    seen_buffers = set()          # buffers are shared between tensors -- count once
    for sg in m.subgraphs:
        for t in sg.tensors:
            bi = int(t.buffer)
            b = m.buffers[bi]
            n = 0
            if getattr(b, "data", None) is not None:
                n = len(b.data)
            elif getattr(b, "size", 0):
                n = int(b.size)
            if n <= 0:
                continue
            ty = _tt(int(t.type))[0]
            by[ty] += 1
            if bi in seen_buffers:
                continue
            seen_buffers.add(bi)
            nbytes[ty] += n
    total = sum(nbytes.values())
    for ty, n in nbytes.most_common():
        print(f"   {ty:<10} {by[ty]:>6} tensors  {n/2**20:>10.2f} MiB  {n/max(total,1)*100:5.1f}%")
    print(f"   {'TOTAL':<10} {sum(by.values()):>6} tensors  {total/2**20:>10.2f} MiB")

    if full:
        print("\n-- largest 15 buffers")
        big = []
        for sg in m.subgraphs:
            for t in sg.tensors:
                b = m.buffers[t.buffer]
                n = len(b.data) if getattr(b, "data", None) is not None else int(getattr(b, "size", 0) or 0)
                if n > 0:
                    nm = t.name.decode() if isinstance(t.name, bytes) else str(t.name)
                    big.append((n, nm.split(";")[0][-90:], _tt(int(t.type))[0]))
        for n, nm, ty in sorted(set(big), reverse=True)[:15]:
            print(f"   {n/2**20:>9.2f} MiB {ty:<9} {nm}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sections-only", action="store_true")
    a = ap.parse_args(argv)

    if a.path.endswith(".tflite"):
        dump_tflite(a.path, os.path.basename(a.path), a.full)
        return 0

    from ai_edge_quantizer.utils import litertlm_utils
    f = litertlm_utils.LiteRTLMFile(a.path)
    total = os.path.getsize(a.path)
    print(f"=== {a.path}  {total} bytes ({total/2**20:.1f} MiB) ===")
    md = f.get_system_metadata()
    for k, v in md.items():
        print(f"   {k}: {v}")
    print(f"\n-- {len(f.sections)} sections")
    tflite_ids = []
    for i, s in enumerate(f.sections):
        beg, end = int(s.beginOffset), int(s.endOffset)
        n = end - beg
        ty = SECTION_TYPE.get(int(s.dataType), f"type{s.dataType}")
        meta = f.get_section_metadata(i)
        nm = meta.get("model_type") or meta.get("name") or ""
        print(f"   [{i}] {ty:<18} {n:>12} B  {n/2**20:>9.2f} MiB  {nm}")
        if int(s.dataType) == 3:
            tflite_ids.append((i, f"{ty}[{i}] {nm}"))
    if a.sections_only:
        return 0
    for i, label in tflite_ids:
        buf = f.get_section_buffer(i)
        if buf is None:
            print(f"\n(section {i}: no buffer)")
            continue
        try:
            dump_tflite(buf, label, a.full)
        except Exception as e:  # a section we cannot parse must not kill the rest
            print(f"\n(section {i}: could not parse -- {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
