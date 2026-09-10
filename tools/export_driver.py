#!/usr/bin/env python3
# ============================================================================
# export_driver.py -- run `litert-torch export_hf` under a memory profiler.
#
# Standalone: no repo imports, stdlib only. Used by the GitHub Actions workflow
# (`gemma4_export.yml`); it is the extracted twin of the profiler that lives
# inline in `kaggle_e4b_export.py`. The notebook keeps its own copy on purpose --
# it is the artifact-producing path that has actually been validated end to end,
# and re-cutting it to import this file would invalidate that without a 40-minute
# run to re-prove it. If you change the profiling logic, change both.
#
# WHY A PROFILER AT ALL. The number that matters on this pipeline is NOT RSS.
# The out-of-core patches (see litert_torch_patches.py) work by turning the
# weights from anonymous memory into file-backed memory, so a healthy run looks
# like "anon flat, cached huge, cached collapsing under pressure" -- which a
# single RSS figure reports as indistinguishable from a leak. AnonPages and
# Cached have to be read separately.
#
# WHY THE SUB-STAGE LABELS. `export_hf` runs three sub-models in one process
# (prefill_decode, embedder, per_layer_embedder). Kaggle v15 peaked at 30.2 GB
# and was OOM-killed; the label on the sample said `per_layer_embedder`, and one
# sub-model earlier the embedder -- the same shape of graph, the same patches --
# had cost 1 GB. That contrast is what identified the cause (a
# `converter.convert(strict_export=False)` call site with no
# lightweight_conversion argument) in one run instead of several.
#
# Usage:
#   python export_driver.py --log export.log --memlog memprofile.txt -- \
#       litert-torch export_hf --model=... --output_dir=... [flags]
# ============================================================================
import argparse
import os
import re
import subprocess
import sys
import threading
import time

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# Ordered longest-first: "per_layer_embedder model" contains "embedder model".
_SUBSTAGES = (
    "per_layer_embedder model",
    "text prefill-decode model",
    "embedder model",
    "vision encoder models",
    "auxiliary model",
    "Quantize model",
    "Write Model",
    "Merge MLIR Modules",
    "Run LiteRT Converter Passes",
    "LiteRT Runtime Constant Folding",
    "Package model",
)


def meminfo():
    d = {}
    try:
        with open("/proc/meminfo") as f:
            for ln in f:
                k, v = ln.split(":", 1)
                d[k] = int(v.split()[0]) // 1024  # MiB
    except OSError:
        pass
    return d


class MemProfiler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.stage = "start"
        self.pid = None
        self.stop_flag = False
        self.peak = {}
        self._fh = open(path, "w")

    def note(self, stage):
        self.stage = stage
        self.peak.setdefault(
            stage,
            {"anon": 0, "cached": 0, "swap_used": 0, "child_rss": 0, "min_avail": 10 ** 9},
        )

    def _child_rss(self):
        if not self.pid:
            return 0
        try:
            with open("/proc/%d/status" % self.pid) as f:
                for ln in f:
                    if ln.startswith("VmHWM:"):
                        return int(ln.split()[1]) // 1024
        except OSError:
            pass
        return 0

    def run(self):
        last_print, last_anon = 0.0, -10 ** 9
        while not self.stop_flag:
            try:
                m = meminfo()
                anon = m.get("AnonPages", 0)
                cached = m.get("Cached", 0)
                avail = m.get("MemAvailable", 0)
                swap_used = m.get("SwapTotal", 0) - m.get("SwapFree", 0)
                crss = self._child_rss()
                p = self.peak.setdefault(
                    self.stage,
                    {"anon": 0, "cached": 0, "swap_used": 0, "child_rss": 0,
                     "min_avail": 10 ** 9},
                )
                p["anon"] = max(p["anon"], anon)
                p["cached"] = max(p["cached"], cached)
                p["swap_used"] = max(p["swap_used"], swap_used)
                p["child_rss"] = max(p["child_rss"], crss)
                p["min_avail"] = min(p["min_avail"], avail)
                line = (
                    "[mem %s] %-42s anon=%6dM cached=%6dM avail=%6dM swap=%6dM "
                    "childHWM=%6dM"
                    % (time.strftime("%H:%M:%S"), self.stage, anon, cached, avail,
                       swap_used, crss)
                )
                self._fh.write(line + "\n")
                self._fh.flush()
                now = time.time()
                if abs(anon - last_anon) > 250 or (now - last_print) > 120:
                    print(line, flush=True)
                    last_print, last_anon = now, anon
            except Exception as e:  # never let the profiler kill the run
                print("[mem] sampler error: %s" % e, flush=True)
            time.sleep(5)
        self._fh.close()

    def report(self):
        print("\n=== memory peaks by stage (MiB) ===", flush=True)
        print("%-44s%11s%13s%11s%11s%11s"
              % ("stage", "peak anon", "peak cached", "peak swap", "min avail",
                 "child HWM"))
        for st, p in self.peak.items():
            print("%-44s%11d%13d%11d%11d%11d"
                  % (st, p["anon"], p["cached"], p["swap_used"],
                     p["min_avail"] if p["min_avail"] < 10 ** 9 else 0,
                     p["child_rss"]), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="export.log")
    ap.add_argument("--memlog", default="memprofile.txt")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="everything after `--` is the command to run")
    args = ap.parse_args(argv)
    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        raise SystemExit("no command given (put it after `--`)")

    prof = MemProfiler(args.memlog)
    prof.note("export")
    prof.start()

    print("$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    prof.pid = proc.pid
    prev = None
    with open(args.log, "w") as lf:
        for line in proc.stdout:
            lf.write(line)
            if line != prev:
                sys.stdout.write(line)
                sys.stdout.flush()
            prev = line
            if "[START]" in line:
                flat = _ANSI.sub("", line)
                for key in _SUBSTAGES:
                    if key in flat:
                        prof.note(key)
                        break
    rc = proc.wait()
    prof.pid = None
    prof.note("done")
    time.sleep(6)
    prof.stop_flag = True
    time.sleep(6)
    print("\nexit=%d wall=%.1f min" % (rc, (time.time() - t0) / 60), flush=True)
    if rc in (137, -9):
        # A SIGKILL here is the OOM killer, not the export failing. Say so: six
        # GitHub runs were previously read as "the export crashed" when what
        # actually happened was the runner agent being starved out.
        print("exit 137/-9 == SIGKILL == the OOM killer (or the runner agent's "
              "watchdog). Read memprofile.txt, not the export log.", flush=True)
    prof.report()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
