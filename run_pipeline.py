"""
Запускать в папке с camera_new_[1..5].mkv
python run_pipeline.py --root . --skip-stage1     # пересчитать только склейку и открыть интерфейс
"""

import argparse
import os
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--root", default=".")
ap.add_argument("--skip-stage1", action="store_true")
ap.add_argument("--no-app", action="store_true")
a, extra = ap.parse_known_args()
extra = [x for x in extra if x != "--"]
here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable
run = lambda args: subprocess.run([py] + args, check=True, cwd=here)
loc, res = os.path.join(a.root, "local_tracks.pkl"), os.path.join(a.root, "fusion_result.pkl")
if not a.skip_stage1:
    run(["stage1_detect_track.py", "--root", a.root, "--out", loc] + extra)
run(["fusion.py", "--inp", loc, "--out", res])
if not a.no_app:
    run(["app.py", "--root", a.root, "--result", res, "--plan", os.path.join(here, "officePlan.jpg")])
