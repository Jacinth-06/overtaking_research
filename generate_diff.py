import difflib

def generate_diff(file1, file2, out_file):
    with open(file1) as f1, open(file2) as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()
        
    diff = list(difflib.unified_diff(lines2, lines1, fromfile=file2, tofile=file1))
    
    with open(out_file, "w") as out:
        out.writelines(diff)

generate_diff("tests/version5.py", "tests/exp4_lane_follow_lidar_stop.py", "/Users/jacinth/.gemini/antigravity-ide/brain/540d3c5c-e6a9-4d6e-ab8c-21cc0aaa3872/scratch/diff.md")
