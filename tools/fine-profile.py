#!/usr/bin/env python3
"""Sampling profiler with inline-frame and source-line attribution.

tools/profile.sh answers "which function is hot" from sample(1)'s call tree.
It cannot see inside a function: everything LLVM inlined is charged to whatever
it was inlined into, and a 200-line function reads as one number. This answers
"which *line* is hot, and what got inlined there".

The mechanism is samply plus a dSYM. Rust on macOS leaves debug info in the
per-CGU object files and only references them from the binary, so a profiler
that reads the binary alone sees function symbols and nothing else. dsymutil
links that DWARF into a bundle; with it present samply resolves each address to
a source line plus the chain of inlined frames at that address.

Usage: tools/fine-profile.py [position] [seconds]
  tools/fine-profile.py                  # startpos, 10s
  tools/fine-profile.py kiwipete 20
  tools/fine-profile.py "8/2p5/..." 15

Env: RATE (samples/sec, default 1000), LINES (source lines shown, default 25).

macOS only: samply works on Linux too, but the dSYM step is Mach-O specific.
On Linux the debug info is already in the binary and no dsymutil run is needed.
"""

import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target/profiling/lattice")
PROFILE = "/tmp/lattice-fine-profile.json.gz"
PORT = 3599

POSITIONS = {
    "startpos": "startpos",
    "kiwipete": "fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "endgame": "fen 8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
}


def record(position, seconds, rate):
    subprocess.run(
        ["cargo", "build", "--profile", "profiling"],
        cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Rust leaves DWARF in the object files and the binary only carries OSO
    # references to them, so without this the profile has function names but no
    # lines and no inline frames.
    shutil.rmtree(BIN + ".dSYM", ignore_errors=True)
    subprocess.run(["xcrun", "dsymutil", BIN], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    fen = POSITIONS.get(position, f"fen {position}")
    # The engine exits at EOF, so stdin has to stay open for the whole search;
    # a closed pipe cuts the profile short at whatever depth finished first.
    script = f"position {fen}\ngo movetime {seconds * 1000}\n"
    print(f"recording {seconds}s at {rate}Hz ({position})", file=sys.stderr)
    proc = subprocess.Popen(
        ["samply", "record", "--save-only", "-o", PROFILE, "--rate", str(rate), BIN],
        cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, text=True,
    )
    proc.stdin.write(script)
    proc.stdin.flush()
    time.sleep(seconds + 2)
    proc.stdin.write("quit\n")
    proc.stdin.close()
    proc.wait(timeout=30)


def serve():
    """Start `samply load` and return (process, base URL with session token)."""
    proc = subprocess.Popen(
        ["samply", "load", "--port", str(PORT), PROFILE],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    for _ in range(40):
        time.sleep(0.25)
        try:
            index = urllib.request.urlopen(f"http://localhost:{PORT}/", timeout=2).read()
        except OSError:
            continue
        # The symbolication endpoint lives under a per-session token, not at
        # the server root; the index page is the only place it is published.
        match = re.search(rb"/([a-z0-9]{20,})/profile\.json", index)
        if match:
            return proc, f"http://localhost:{PORT}/{match.group(1).decode()}"
    proc.kill()
    sys.exit("samply load did not come up")


def symbolicate(base, breakpad_id, addresses):
    request = urllib.request.Request(
        f"{base}/symbolicate/v5",
        json.dumps({
            "memoryMap": [["lattice", breakpad_id]],
            "stacks": [[[0, a] for a in addresses]],
        }).encode(),
        {"Content-Type": "application/json"},
    )
    frames = json.load(urllib.request.urlopen(request))["results"][0]["stacks"][0]
    return {int(f["module_offset"], 16): f for f in frames}


def short(path):
    return path.split("/rust/library/")[-1] if path else "?"


def main():
    position = sys.argv[1] if len(sys.argv) > 1 else "startpos"
    seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    rate = int(os.environ.get("RATE", 1000))
    top_lines = int(os.environ.get("LINES", 25))

    record(position, seconds, rate)
    profile = json.load(gzip.open(PROFILE))
    lib = next(l for l in profile["libs"] if l["debugName"] == "lattice")

    # The search runs on its own thread; the main thread only reads stdin.
    thread = max(profile["threads"], key=lambda t: t["frameTable"]["length"])
    frame_table, stack_table = thread["frameTable"], thread["stackTable"]

    self_samples = Counter()
    for stack in thread["samples"]["stack"]:
        if stack is not None:
            self_samples[stack_table["frame"][stack]] += 1
    total = sum(self_samples.values())
    if not total:
        sys.exit("no samples recorded")

    server, base = serve()
    try:
        addresses = sorted({
            frame_table["address"][f] for f in self_samples
            if frame_table["address"][f] not in (None, -1)
        })
        resolved = symbolicate(base, lib["breakpadId"], addresses)
    finally:
        server.kill()

    by_function, by_line, by_inline = Counter(), Counter(), Counter()
    unresolved = 0
    for frame, count in self_samples.items():
        info = resolved.get(frame_table["address"][frame])
        if not info or "function" not in info:
            unresolved += count
            continue
        function = info["function"]
        by_function[function] += count
        if "line" in info:
            file = os.path.basename(info.get("file", "?"))
            by_line[(f"{file}:{info['line']}", function)] += count
        for inline in info.get("inlines", []):
            by_inline[(inline["function"], short(inline.get("file", "")))] += count

    def show(title, counter, fmt, limit):
        print(f"\n{title}")
        for key, count in counter.most_common(limit):
            print(f"  {100 * count / total:6.2f}%  {fmt(key)}")

    print(f"\n{total} samples, {position}, {seconds}s at {rate}Hz")
    if unresolved:
        print(f"  {100 * unresolved / total:.1f}% unresolved "
              "(system libraries and stack frames outside the binary)")

    show("self time by function", by_function, lambda k: k, 15)
    show("self time by source line", by_line,
         lambda k: f"{k[0]:<22} {k[1]}", top_lines)
    if by_inline:
        show("inlined into hot addresses (cost is charged to the caller above)",
             by_inline, lambda k: f"{k[0]:<48} {k[1]}", 15)
    else:
        print("\nno inline frames resolved - is the dSYM present next to the binary?")


if __name__ == "__main__":
    main()
