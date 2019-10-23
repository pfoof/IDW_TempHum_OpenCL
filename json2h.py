import json
import sys

index = 0

if len(sys.argv) > 1:
    file = sys.argv[1]
    if len(sys.argv) > 2:
        index = int(sys.argv[2])
else:
    print("Usage json2h.py input.json [index=0]")
    quit()

arr = []

with open(file, 'r') as f:
    arr = json.load(f)

if index >= len(arr):
    print("Index out of range!")
    quit()

record = arr[index]

