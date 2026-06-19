import difflib

with open('tests/version2.py', 'r') as f1, \
     open('tests/version3.py', 'r') as f2:
    lines1 = f1.readlines()
    lines2 = f2.readlines()

diff = difflib.unified_diff(lines1, lines2, fromfile='version2.py', tofile='version3.py')
with open('diff_out.txt', 'w') as out:
    for line in diff:
        out.write(line)
