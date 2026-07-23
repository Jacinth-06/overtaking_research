import difflib
import sys

def diff(file1, file2):
    with open(file1) as f1, open(file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        for line in difflib.unified_diff(lines2, lines1, fromfile=file2, tofile=file1, n=0):
            if line.startswith("-") or line.startswith("+"):
                print(line.strip())

diff("tests/version5.py", "tests/exp4_lane_follow_lidar_stop.py")
