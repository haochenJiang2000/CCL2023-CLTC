import sys
import os

test_file = sys.argv[1]
result_file = sys.argv[2]
submit_file = sys.argv[3]
if not os.path.exists(submit_file):
    os.makedirs(os.path.dirname(submit_file), exist_ok=True)
with open(test_file) as test, open(result_file) as result, open(submit_file, "w") as submit:
    index = 0
    for line1, line2 in zip(test, result):
        line1 = line1.strip()
        line2 = line2.strip()
        print(index, line1, line2, sep="\t", file=submit)
        index += 1
