#!/usr/bin/env python3
# ============================================================================
# litert_torch_patches.py -- out-of-core patches for `litert-torch`
#
# WRITTEN AGAINST: litert-torch 0.9.4  (pure python, `pip install litert-torch`)
# TARGET PROBLEM:  exporting Gemma 4 E4B (14.9 GiB bf16 safetensors) to
#                  .litertlm on a host with 31 GB of RAM and NO SWAP
#                  (Kaggle CPU notebook).
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS -- the memory anatomy of an `export_hf` run
# ---------------------------------------------------------------------------
# `litert-torch` was tuned on hosts that have swap. Its peak working set for a
# large LLM is roughly THREE simultaneous copies of the weights, all of them
# ANONYMOUS (un-evictable without swap):
#
#   1. `export_lib.load_model` hardcodes `torch_dtype=torch.float32`, so a bf16
#      checkpoint is upcast at load:            15 GiB source -> ~30 GB anon.
#   2. `backend/inline_consts.py` materialises every graph constant with
#      `ir.DenseElementsAttr.get(arr, ...)`, which COPIES the buffer into the
#      MLIR context's storage:                  + another full copy, anon.
#   3. `_convert/runtime_fold.py` runs the constant-only subgraph through a
#      LiteRT interpreter and keeps every folded numpy array alive for the
#      lifetime of the module:                  + more anon.
#
# A GitHub runner survived E2B at "peak RSS 14.66 GiB" only because it had
# 64 GB of swap: `getrusage`'s ru_maxrss reports what stayed RESIDENT, and
# Linux never populates ru_nswap (the "Swaps: 0" line in /usr/bin/time -v is
# always 0 and means nothing).  On a swap-less host the same run needs all of
# it resident at once, and dies.
#
# THE PRINCIPLE BEHIND EVERY PATCH HERE:
#
#     On a host with no swap, mmap IS swap.  Turn anonymous memory into
#     file-backed memory and the kernel can evict it again.
#
# So: the weights are upcast to f32 ONCE, into a file on the big scratch disk,
# and are thereafter referenced by mmap -- by torch (P2), by MLIR (P3/P5, via
# `DenseResourceElementsAttr.get_from_buffer`, which does not copy), and by the
# constant folder (P6).  Peak anonymous memory drops from ~3x weights to ~0.
#
# ---------------------------------------------------------------------------
# SAFETY PROPERTY -- this script is INERT until you opt in
# ---------------------------------------------------------------------------
# Every patch routes through `litert_torch/_lt_arena.py` (installed by this
# script) and reads an environment variable whose DEFAULT reproduces upstream
# behaviour exactly.  A fresh `pip install litert-torch==0.9.4` followed by
# `python litert_torch_patches.py --apply` with no environment variables set
# behaves identically to the unpatched library.  That is what makes an
# unpatched-vs-patched A/B on the same host a fair test.
#
# Knobs (all read at import time of `litert_torch._lt_arena`):
#
#   LT_LOAD_NATIVE_DTYPE=1   P1  load the checkpoint in bf16, not upcast f32
#   LT_ARENA_DIR=<dir>       P2  upcast params/buffers to f32 into mmap files
#                                in <dir> and swap them into the model
#   LT_ARENA_MIN_BYTES=N     P2  only tensors >= N f32 bytes go to disk (1 MiB)
#   LT_NO_INPLACE_CLAMP=1    P4  don't clamp constants in place (defaults ON
#                                whenever LT_ARENA_DIR is set)
#   LT_RESOURCE_ALL_DTYPES=1 P5  allow bf16/f16 through the zero-copy
#                                DenseResourceElementsAttr path
#   LT_RUNTIME_CONST_FOLD=   P7  auto (default, = upstream) | on | off
#
# and one thing that is NOT a patch but a CLI flag you must pass:
#
#   --experimental_lightweight_conversion=True
#       plumbs to `InlineConstsContext(enable_resource_constants=True)`, i.e.
#       it selects the zero-copy branch in inline_consts.py.  Without it P5
#       has nothing to widen and P2 only halves the peak instead of removing
#       it.
#
# ---------------------------------------------------------------------------
# USAGE
# ---------------------------------------------------------------------------
#   python litert_torch_patches.py --apply     # patch the installed package
#   python litert_torch_patches.py --check     # verify without writing
#   python litert_torch_patches.py --print-env # show the resolved knobs
#
# Re-running --apply is a no-op (every patch is guarded by a marker and every
# anchor is asserted to match exactly the expected number of times, so a
# silently-drifted upstream fails loudly instead of half-applying).
# ============================================================================

from __future__ import annotations

import argparse
import dataclasses
import importlib
import os
import py_compile
import sys
import textwrap

EXPECTED_VERSION = "0.9.4"
MARK = "# [litert_torch_patches]"
# Always write LF: the patched files are consumed on Linux, and rewriting a
# whole file with CRLF would make every later diff useless.
_LF = "\n"
# Short alias, used inside the multi-line patch anchors below.
NL = _LF

# The helper module this script installs into the litert_torch package.  It is
# a normal module so that the patched library files can `import` it in any
# process (the export runs as a subprocess of the notebook, so monkeypatching
# from the notebook would not reach it).
ARENA_MODULE_NAME = "_lt_arena.py"
ARENA_MODULE_SRC = r'''
# ============================================================================
# litert_torch/_lt_arena.py -- INSTALLED BY research/edgeparity-cacheblend/
# export/litert_torch_patches.py.  Do not edit here; edit the generator.
#
# Out-of-core helpers that let a >30 GB export run on a swap-less host by
# keeping the weights in mmap'd files instead of anonymous memory.
#
# Everything is inert unless the matching env var is set; the defaults
# reproduce stock litert-torch behaviour byte for byte.
# ============================================================================
import gc
import os

import numpy as np
import torch

__all__ = [
    "LOAD_DTYPE",
    "ARENA_DIR",
    "ARENA_MIN_BYTES",
    "NO_INPLACE_CLAMP",
    "RESOURCE_ALL_DTYPES",
    "clamp_inf_values",
    "drop_float_model",
    "externalize_source_model",
    "resource_attr_dtypes",
    "runtime_const_fold_setting",
    "spill",
    "stats",
]


def _flag(name, default=False):
  v = os.environ.get(name)
  if v is None:
    return default
  return v.strip().lower() in ("1", "y", "yes", "t", "true", "on")


# --- P1 -------------------------------------------------------------------
# export_lib.load_model hardcodes torch.float32.  The checkpoint on disk is
# bf16, so that upcast doubles it at load time (~30 GB for E4B) and the run
# dies before a graph exists.  bf16 -> f32 is an exact widening, so loading in
# the file's own dtype and upcasting later (P2, to disk) is bit-faithful.
LOAD_DTYPE = torch.bfloat16 if _flag("LT_LOAD_NATIVE_DTYPE") else torch.float32

# --- P2 -------------------------------------------------------------------
ARENA_DIR = os.environ.get("LT_ARENA_DIR", "").strip()
ARENA_MIN_BYTES = int(os.environ.get("LT_ARENA_MIN_BYTES", str(1 << 20)))
ARENA_DTYPE = torch.float32

# --- P4 -------------------------------------------------------------------
# Defaults ON when the arena is in use: an in-place clamp_ would dirty every
# page of a 30 GB mmap for a no-op on finite data.
NO_INPLACE_CLAMP = _flag("LT_NO_INPLACE_CLAMP", default=bool(ARENA_DIR))

# --- P5 -------------------------------------------------------------------
RESOURCE_ALL_DTYPES = _flag("LT_RESOURCE_ALL_DTYPES")

# --- P10 ------------------------------------------------------------------
DROP_FLOAT_TFLITE = _flag("LT_DROP_FLOAT_TFLITE")

_KEEP = []          # keeps every mmap alive for the lifetime of the process
_N = [0]            # arena file counter
_BYTES = [0]        # arena bytes written


def _log(msg):
  print("[lt_arena] " + str(msg), flush=True)


def stats():
  return {"files": _N[0], "bytes": _BYTES[0], "dir": ARENA_DIR}


def _new_path(tag):
  _N[0] += 1
  safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(tag))[-90:]
  os.makedirs(ARENA_DIR, exist_ok=True)
  return os.path.join(ARENA_DIR, "a%05d_%s.bin" % (_N[0], safe))


def _to_disk(t, name):
  # Write t (any float dtype) to an mmap'd f32 file and return a torch tensor
  # backed by it.  Chunked along dim 0 so the transient upcast buffer stays
  # bounded regardless of tensor size.
  shape = tuple(t.shape)
  path = _new_path(name)
  mm = np.memmap(path, dtype=np.float32, mode="w+", shape=shape)
  if t.dim() == 0:
    mm[...] = t.to(torch.float32).numpy()
  else:
    rows = shape[0]
    per_row = max(1, t.numel() // max(rows, 1))
    step = max(1, (64 << 20) // (per_row * 4))
    for i in range(0, rows, step):
      mm[i : i + step] = t[i : i + step].to(torch.float32).numpy()
  mm.flush()
  del mm
  # Re-open read/write (MAP_SHARED).  Writable so that any library code that
  # still mutates a constant cannot fault; P4 makes sure nothing does, so the
  # pages stay clean and the kernel can drop them under pressure.
  out = np.memmap(path, dtype=np.float32, mode="r+", shape=shape)
  _KEEP.append(out)
  _BYTES[0] += out.nbytes
  return torch.from_numpy(out)


def _maybe_move(t, seen, name):
  if t is None or not torch.is_floating_point(t):
    return None
  out_bytes = t.numel() * 4
  if out_bytes < ARENA_MIN_BYTES:
    # Small: upcast in RAM so the whole model is uniformly f32, exactly as
    # stock load_model(torch_dtype=torch.float32) would have produced.
    return None if t.dtype == ARENA_DTYPE else t.to(ARENA_DTYPE)
  if not t.is_contiguous():
    t = t.contiguous()
  key = (
      t.untyped_storage().data_ptr(),
      t.storage_offset(),
      tuple(t.shape),
      tuple(t.stride()),
  )
  # Tied weights (Gemma 4 ties lm_head to the input embedding) must stay tied.
  if key in seen:
    return seen[key]
  new = _to_disk(t, name)
  seen[key] = new
  return new


def externalize_model(model):
  # Replace every float parameter/buffer with an f32 tensor backed by a file
  # in ARENA_DIR.  After this call the module is numerically identical to what
  # stock `from_pretrained(torch_dtype=torch.float32)` would have built, but
  # its weights are file-backed page cache instead of anonymous memory.
  if not ARENA_DIR:
    return model
  seen = {}
  moved = 0
  for mod_name, mod in model.named_modules():
    for pname, p in list(mod._parameters.items()):  # pylint: disable=protected-access
      if p is None:
        continue
      new = _maybe_move(p.data, seen, mod_name + "." + pname)
      if new is not None:
        mod._parameters[pname] = torch.nn.Parameter(  # pylint: disable=protected-access
            new, requires_grad=False
        )
        moved += 1
    for bname, b in list(mod._buffers.items()):  # pylint: disable=protected-access
      if b is None:
        continue
      new = _maybe_move(b, seen, mod_name + "." + bname)
      if new is not None:
        mod._buffers[bname] = new  # pylint: disable=protected-access
        moved += 1
  gc.collect()
  _log(
      "externalized %d tensors, %d arena files, %.2f GiB on disk at %s"
      % (moved, _N[0], _BYTES[0] / (1 << 30), ARENA_DIR)
  )
  return model


def externalize_source_model(fn):
  # Decorator for export_lib.load_model.
  import functools

  @functools.wraps(fn)
  def _wrapped(*args, **kwargs):
    artifacts = fn(*args, **kwargs)
    if ARENA_DIR and getattr(artifacts, "model", None) is not None:
      externalize_model(artifacts.model)
    return artifacts

  return _wrapped


# --- P4 -------------------------------------------------------------------
def clamp_inf_values(tensor):
  # Same result as the stock in-place clamp_ (clamping finite values is the
  # identity; +/-inf becomes +/-finfo.max/min either way), but never writes
  # through an mmap.  The inf test is chunked so the boolean temporary stays
  # bounded instead of being 1/4 of the tensor.
  if not torch.is_floating_point(tensor):
    return tensor
  info = torch.finfo(tensor.dtype)
  if not NO_INPLACE_CLAMP:
    tensor.clamp_(min=info.min, max=info.max)
    return tensor
  flat = tensor.reshape(-1)
  n = flat.numel()
  step = 1 << 24
  has_inf = False
  for i in range(0, n, step):
    if bool(torch.isinf(flat[i : i + step]).any()):
      has_inf = True
      break
  if not has_inf:
    return tensor
  return tensor.clamp(min=info.min, max=info.max)


# --- P5 -------------------------------------------------------------------
def resource_attr_dtypes():
  # Upstream restricts the zero-copy DenseResourceElementsAttr path to
  # float32/int32.  With the bf16->uint16 view in
  # _tensor_to_mlir_compatible_array the 2-byte float types are equally valid
  # (the element type on the MLIR side is carried by tensor_type, not by the
  # numpy dtype), so they can be widened in.  Off by default: with the arena
  # enabled there are no large bf16 constants left, so this is a safety net.
  base = [torch.float32, torch.int32]
  if RESOURCE_ALL_DTYPES:
    base += [torch.bfloat16, torch.float16]
  return base


# --- P6 -------------------------------------------------------------------
def spill(arr):
  # runtime_fold materialises every folded constant as a numpy array and the
  # resulting DenseResourceElementsAttr pins it for the module's lifetime.
  # Copy it to the arena so the anonymous original can be collected.
  if not ARENA_DIR:
    return arr
  arr = np.ascontiguousarray(arr)
  path = _new_path("fold")
  mm = np.memmap(path, dtype=arr.dtype, mode="w+", shape=arr.shape)
  mm[...] = arr
  mm.flush()
  del mm
  out = np.memmap(path, dtype=arr.dtype, mode="r+", shape=arr.shape)
  _KEEP.append(out)
  _BYTES[0] += out.nbytes
  return out


# --- P10 ------------------------------------------------------------------
def drop_float_model(src, dst):
  # After a successful quantisation the float .tflite is dead weight. For E4B
  # that is 27.7 GiB across the three sub-models (14.7 + 10.5 + 2.5). Kaggle's
  # 7.9 TB scratch does not care; a GitHub runner with ~40 GB left after the
  # swapfile very much does.
  if not DROP_FLOAT_TFLITE or src == dst:
    return
  try:
    if os.path.exists(dst) and os.path.getsize(dst) > 0 and os.path.exists(src):
      n = os.path.getsize(src)
      os.remove(src)
      _log("dropped float model %s (%.2f GiB)" % (src, n / (1 << 30)))
  except OSError as e:
    _log("could not drop %s: %s" % (src, e))


# --- P7 -------------------------------------------------------------------
def runtime_const_fold_setting():
  # None == upstream default (follows lightweight_conversion).
  v = os.environ.get("LT_RUNTIME_CONST_FOLD", "auto").strip().lower()
  if v in ("on", "1", "true", "yes", "y"):
    return True
  if v in ("off", "0", "false", "no", "n"):
    return False
  return None
'''


@dataclasses.dataclass(frozen=True)
class Patch:
    pid: str
    module: str          # dotted module path inside litert_torch
    anchor: str
    replacement: str
    count: int
    why: str


PATCHES = [
    # ---- P1 -------------------------------------------------------------
    Patch(
        pid="P1",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor="torch_dtype=torch.float32",
        replacement="torch_dtype=_lt_arena.LOAD_DTYPE",
        count=2,
        why=(
            "load_model() hardcodes an fp32 upcast of a bf16 checkpoint, which "
            "doubles 14.9 GiB to ~30 GB of anonymous memory before any graph "
            "exists.  Routed through LOAD_DTYPE so the default is still fp32."
        ),
    ),
    # ---- P2 -------------------------------------------------------------
    Patch(
        pid="P2",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor="@progress.task('Load source model')\ndef load_model(",
        replacement=(
            "@progress.task('Load source model')\n"
            "@_lt_arena.externalize_source_model  " + MARK + "\n"
            "def load_model("
        ),
        count=1,
        why=(
            "After the bf16 load, upcast every large parameter/buffer to f32 "
            "INTO AN MMAP'D FILE and swap it into the module.  The model then "
            "matches what stock fp32 loading would have produced, but lives in "
            "evictable page cache instead of anonymous memory."
        ),
    ),
    # ---- P3 -------------------------------------------------------------
    Patch(
        pid="P3",
        module="litert_torch.backend.inline_consts",
        anchor="arr = tensor.contiguous().detach().cpu().numpy()",
        replacement=(
            "arr = (lambda _t: (_t.view(torch.uint16) if _t.dtype =="
            " torch.bfloat16 else _t).numpy())(tensor.contiguous().detach().cpu())"
        ),
        count=1,
        why=(
            "numpy has no bfloat16: `.numpy()` raises 'Got unsupported "
            "ScalarType BFloat16'.  Hand MLIR the raw 2-byte pattern instead. "
            "A .float() upcast does NOT work here -- tensor_type is derived "
            "separately and stays bf16, so DenseElementsAttr rejects the "
            "buffer.  Guarded on dtype, so every other constant is untouched."
        ),
    ),
    # ---- P4 -------------------------------------------------------------
    Patch(
        pid="P4",
        module="litert_torch.backend.inline_consts",
        anchor=(
            "def _clamp_inf_values(tensor: torch.Tensor):\n"
            '  """Clamps a tensor to the min/max value for float tensors."""\n'
            "  if torch.is_floating_point(tensor):\n"
            "    info = torch.finfo(tensor.dtype)\n"
            "    tensor.clamp_(min=info.min, max=info.max)\n"
            "  return tensor"
        ),
        replacement=(
            "def _clamp_inf_values(tensor: torch.Tensor):\n"
            '  """Clamps a tensor to the min/max value for float tensors."""\n'
            "  return _lt_arena.clamp_inf_values(tensor)  " + MARK
        ),
        count=1,
        why=(
            "The stock clamp_ mutates the constant in place.  For an "
            "arena-backed constant that dirties every page of a 30 GB mmap to "
            "perform a no-op on finite data, forcing 30 GB of writeback.  The "
            "replacement tests for inf first (chunked) and only then clamps, "
            "out of place."
        ),
    ),
    # ---- P5 -------------------------------------------------------------
    Patch(
        pid="P5",
        module="litert_torch.backend.inline_consts",
        anchor="  if x.dtype not in [torch.float32, torch.int32]:",
        replacement="  if x.dtype not in _lt_arena.resource_attr_dtypes():  " + MARK,
        count=1,
        why=(
            "The zero-copy DenseResourceElementsAttr branch is gated to "
            "float32/int32.  With P3 in place the 2-byte float types are "
            "equally valid; widening is opt-in via LT_RESOURCE_ALL_DTYPES and "
            "acts as a safety net for any large bf16 constant the arena misses."
        ),
    ),
    # ---- P6 -------------------------------------------------------------
    Patch(
        pid="P6",
        module="litert_torch._convert.runtime_fold",
        anchor=(
            "      ir_attr = ir.DenseResourceElementsAttr.get_from_buffer(\n"
            "          memoryview(arr),"
        ),
        replacement=(
            "      ir_attr = ir.DenseResourceElementsAttr.get_from_buffer(  "
            + MARK
            + "\n          memoryview(_lt_arena.spill(arr)),"
        ),
        count=1,
        why=(
            "get_from_buffer pins the numpy array for the module's lifetime, "
            "so every runtime-folded constant becomes permanent anonymous "
            "memory.  Spilling to the arena first lets the original be "
            "collected."
        ),
    ),
    # ---- P7 -------------------------------------------------------------
    # Three call sites, two indent levels.  Anchoring on the
    # lightweight_conversion + strict_export PAIR (not the bare token) is what
    # makes the count exact -- see the "guards must match an invocation
    # fragment" rule in E4B_FREE_EXPORT.md.
    Patch(
        pid="P7a",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor=(
            "        lightweight_conversion=export_config.experimental_lightweight_conversion,\n"
            "        strict_export=False,"
        ),
        replacement=(
            "        lightweight_conversion=export_config.experimental_lightweight_conversion,\n"
            "        runtime_constant_folding=_lt_arena.runtime_const_fold_setting(),  "
            + MARK
            + "\n"
            "        strict_export=False,"
        ),
        count=2,
        why=(
            "runtime_constant_folding is auto-enabled whenever lightweight "
            "conversion is on, and it runs the whole constant subgraph through "
            "a LiteRT interpreter -- unbounded anonymous memory.  export_hf "
            "exposes no flag for it, so give it one.  Default None == upstream. "
            "(embedder + ASR call sites, 8-space indent)"
        ),
    ),
    # ---- P8 -------------------------------------------------------------
    # MEASURED, run v15: the prefill/decode and embedder exports both survived
    # at peak anon ~14.5 GiB, and then `Export per_layer_embedder model > Lower
    # to MLIR: decode_per_layer_embedder > Create MLIR Module` walked anon from
    # 8.9 GiB to 30.3 GiB and was OOM-killed. Cause: the additional / auxiliary
    # / vision exports call `converter.convert(strict_export=False)` with NO
    # lightweight_conversion argument, so they always take the COPYING
    # DenseElementsAttr branch -- and for gemma4 the additional model is the
    # 262144 x 10752 per-layer embedding table, 10.8 GiB in f32.
    # The embedder export in that same run is the positive control: a 2.5 GiB
    # f32 table, it DOES pass the flag, and it cost about 1 GiB of anon.
    Patch(
        pid="P8",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor="lrt_model = converter.convert(strict_export=False)",
        replacement=(
            "lrt_model = converter.convert(strict_export=False,"
            " lightweight_conversion=export_config.experimental_lightweight_conversion,"
            " runtime_constant_folding=_lt_arena.runtime_const_fold_setting())  "
            + MARK
        ),
        count=5,
        why=(
            "The additional-model, auxiliary-model and vision-encoder exports "
            "hardcode the copying constant path. For gemma4 the additional "
            "model is the 10.8 GiB per-layer embedding table, which makes this "
            "the largest remaining anonymous copy in the pipeline. Written as "
            "one line so it makes no assumption about the indent at the five "
            "call sites; `export_config` is in scope at all of them."
        ),
    ),
    # ---- P10 ------------------------------------------------------------
    Patch(
        pid="P10",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor=(
            "  qt.quantize().export_model(quantized_model_path, overwrite=True)@@"
            "  return quantized_model_path"
        ).replace("@@", NL),
        replacement=(
            "  qt.quantize().export_model(quantized_model_path, overwrite=True)@@"
            "  _lt_arena.drop_float_model(model_path, quantized_model_path)  " + MARK + "@@"
            "  return quantized_model_path"
        ).replace("@@", NL),
        count=1,
        why=(
            "The float .tflite survives its own quantisation. For E4B that is "
            "27.7 GiB of dead intermediates -- irrelevant on Kaggle's 7.9 TB "
            "scratch, decisive on a GitHub runner. Opt-in via "
            "LT_DROP_FLOAT_TFLITE, and it only unlinks once the quantised file "
            "exists and is non-empty."
        ),
    ),
    # ---- P9 -------------------------------------------------------------
    # Only export_text_prefill_decode_model tears its converter down. The
    # embedder and additional-model exports leave theirs alive, so each
    # sub-model's MLIR context is still resident while the next one is built and
    # peak becomes the SUM of the sub-models instead of the largest one.
    Patch(
        pid="P9a",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor=(
            "  model_path = os.path.join(work_dir, 'embedder.tflite')  # pyrefly: ignore[no-matching-overload]@@"
            "  lrt_model.export(model_path)@@"
        ).replace("@@", NL),
        replacement=(
            "  model_path = os.path.join(work_dir, 'embedder.tflite')  # pyrefly: ignore[no-matching-overload]@@"
            "  lrt_model.export(model_path)@@"
            "  del lrt_model, converter  " + MARK + "@@"
            "  gc.collect()@@"
        ).replace("@@", NL),
        count=1,
        why=(
            "Hard teardown between sub-models, at the same point the main model "
            "already does it."
        ),
    ),
    Patch(
        pid="P9b",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor=(
            "  model_path = os.path.join(work_dir, f'{name}.tflite')  # pyrefly: ignore[no-matching-overload]@@"
            "  lrt_model.export(model_path)@@"
        ).replace("@@", NL),
        replacement=(
            "  model_path = os.path.join(work_dir, f'{name}.tflite')  # pyrefly: ignore[no-matching-overload]@@"
            "  lrt_model.export(model_path)@@"
            "  del lrt_model, converter  " + MARK + "@@"
            "  gc.collect()@@"
        ).replace("@@", NL),
        count=1,
        why="Same as P9a for the additional-model (per_layer_embedder) export.",
    ),
    Patch(
        pid="P7b",
        module="litert_torch.generative.export_hf.core.export_lib",
        anchor=(
            "          lightweight_conversion=export_config.experimental_lightweight_conversion,\n"
            "          strict_export=False,"
        ),
        replacement=(
            "          lightweight_conversion=export_config.experimental_lightweight_conversion,\n"
            "          runtime_constant_folding=_lt_arena.runtime_const_fold_setting(),  "
            + MARK
            + "\n"
            "          strict_export=False,"
        ),
        count=1,
        why="Same as P7a for the prefill/decode call site (10-space indent).",
    ),
]

# Files that need `from litert_torch import _lt_arena` injected.
IMPORT_TARGETS = [
    "litert_torch.generative.export_hf.core.export_lib",
    "litert_torch.backend.inline_consts",
    "litert_torch._convert.runtime_fold",
]
IMPORT_LINE = "from litert_torch import _lt_arena  " + MARK


_ROOT_OVERRIDE = None  # set by --root, for testing against an unpacked wheel


def _package_dir() -> str:
    # Deliberately does NOT import litert_torch: the package pulls in
    # litert_converter, which has no Windows wheel, and this script has to be
    # editable/testable off-Linux.
    if _ROOT_OVERRIDE:
        return _ROOT_OVERRIDE
    spec = importlib.util.find_spec("litert_torch")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("litert_torch is not installed (pip install litert-torch)")
    return list(spec.submodule_search_locations)[0]


def _module_path(dotted: str) -> str:
    rel = dotted.split(".")[1:]  # strip the leading 'litert_torch'
    return os.path.join(_package_dir(), *rel) + ".py"


def _installed_version() -> str:
    p = os.path.join(_package_dir(), "version.py")
    try:
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip().startswith("__version__"):
                    return ln.split("=", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return "?"


def _inject_import(src: str) -> tuple[str, bool]:
    if IMPORT_LINE in src:
        return src, False
    lines = src.split("\n")
    last_import = -1
    for i, ln in enumerate(lines[:200]):
        if ln.startswith("import ") or ln.startswith("from "):
            last_import = i
    if last_import < 0:
        raise SystemExit("no import block found to anchor the injected import")
    lines.insert(last_import + 1, IMPORT_LINE)
    return "\n".join(lines), True


def apply(check_only: bool = False, import_verify: bool = True) -> int:
    ver = _installed_version()
    if ver != EXPECTED_VERSION:
        print(
            "WARNING: written against litert-torch %s, found %s -- anchors may "
            "have drifted; a mismatch will fail loudly below."
            % (EXPECTED_VERSION, ver)
        )

    # 1. install litert_torch/_lt_arena.py
    arena_path = os.path.join(_package_dir(), ARENA_MODULE_NAME)
    want = ARENA_MODULE_SRC.lstrip("\n")
    have = ""
    if os.path.exists(arena_path):
        with open(arena_path, "r", encoding="utf-8") as f:
            have = f.read()
    if have != want:
        if check_only:
            print("MISSING/STALE  litert_torch/%s" % ARENA_MODULE_NAME)
        else:
            with open(arena_path, "w", encoding="utf-8", newline=_LF) as f:
                f.write(want)
            print("installed       litert_torch/%s" % ARENA_MODULE_NAME)
    else:
        print("already present litert_torch/%s" % ARENA_MODULE_NAME)

    # 2. inject the import into every file we touch
    touched = {}
    for dotted in IMPORT_TARGETS:
        p = _module_path(dotted)
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        src, changed = _inject_import(src)
        touched[p] = src
        print("import   %-8s %s" % ("added" if changed else "present", dotted))

    # 3. apply the text patches
    ok = True
    for pt in PATCHES:
        p = _module_path(pt.module)
        src = touched.get(p)
        if src is None:
            with open(p, "r", encoding="utf-8") as f:
                src = f.read()
        if pt.replacement in src:
            print("%-4s already applied  (%s)" % (pt.pid, pt.module.split(".")[-1]))
            touched[p] = src
            continue
        n = src.count(pt.anchor)
        if n != pt.count:
            ok = False
            print(
                "%-4s ANCHOR MISMATCH: expected %d occurrence(s), found %d in %s"
                % (pt.pid, pt.count, n, pt.module)
            )
            print("      anchor was:\n" + textwrap.indent(pt.anchor, "        | "))
            continue
        src = src.replace(pt.anchor, pt.replacement, pt.count)
        touched[p] = src
        print("%-4s applied x%d       (%s)" % (pt.pid, pt.count, pt.module.split(".")[-1]))

    if not ok:
        raise SystemExit("PATCH FAILED -- refusing to write a half-patched tree")

    if check_only:
        print("--check: no files written")
        return 0

    for p, src in touched.items():
        with open(p, "w", encoding="utf-8", newline=_LF) as f:
            f.write(src)

    # 4. verify: byte-compile everything we touched, then re-import it
    for p in list(touched) + [arena_path]:
        py_compile.compile(p, doraise=True)
    print("verify: all patched files byte-compile")
    if import_verify:
        for dotted in IMPORT_TARGETS + ["litert_torch._lt_arena"]:
            m = importlib.import_module(dotted)
            importlib.reload(m)
        print("verify: all patched modules re-import")

    # 5. the residual-anchor assertions that a `grep -c` style guard would miss
    el = _module_path("litert_torch.generative.export_hf.core.export_lib")
    with open(el, "r", encoding="utf-8") as f:
        s = f.read()
    assert "torch_dtype=torch.float32" not in s, "P1 left a raw fp32 upcast behind"
    ic = _module_path("litert_torch.backend.inline_consts")
    with open(ic, "r", encoding="utf-8") as f:
        s = f.read()
    assert "tensor.clamp_(" not in s, "P4 left an in-place clamp behind"
    assert "[torch.float32, torch.int32]" not in s, "P5 left the hard dtype gate behind"
    with open(el, "r", encoding="utf-8") as f:
        s = f.read()
    assert "converter.convert(strict_export=False)" not in s, (
        "P8 left a convert() call that cannot take the zero-copy constant path"
    )
    print("verify: residual-anchor assertions pass")
    return 0


def print_env() -> int:
    # Load the arena module by path so this works without importing the
    # (Linux-only) litert_torch package.
    import importlib.util as _u

    path = os.path.join(_package_dir(), ARENA_MODULE_NAME)
    spec = _u.spec_from_file_location("_lt_arena_probe", path)
    a = _u.module_from_spec(spec)
    spec.loader.exec_module(a)

    print("LOAD_DTYPE          %s" % a.LOAD_DTYPE)
    print("ARENA_DIR           %r" % a.ARENA_DIR)
    print("ARENA_MIN_BYTES     %d" % a.ARENA_MIN_BYTES)
    print("NO_INPLACE_CLAMP    %s" % a.NO_INPLACE_CLAMP)
    print("RESOURCE_ALL_DTYPES %s" % a.RESOURCE_ALL_DTYPES)
    print("runtime_const_fold  %r" % a.runtime_const_fold_setting())
    return 0


def main(argv=None) -> int:
    global _ROOT_OVERRIDE
    ap = argparse.ArgumentParser(description="out-of-core patches for litert-torch")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument(
        "--root",
        default=None,
        help="path to the litert_torch package dir (default: the installed one)",
    )
    ap.add_argument(
        "--no-import-verify",
        action="store_true",
        help="skip the re-import check (for hosts where litert_converter is unavailable)",
    )
    args = ap.parse_args(argv)
    _ROOT_OVERRIDE = args.root
    if args.print_env:
        return print_env()
    if args.check:
        return apply(check_only=True, import_verify=False)
    if args.apply:
        return apply(check_only=False, import_verify=not args.no_import_verify)
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
