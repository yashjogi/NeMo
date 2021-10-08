import json
import sys

with open(sys.argv[1]) as f:
    data = json.load(f)
for k in data.keys():
    print(k)
for v in data.values():
    print(v)
