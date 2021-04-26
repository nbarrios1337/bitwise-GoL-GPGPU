#! /usr/bin/env python

import subprocess, sys
import datetime

if len(sys.argv) <= 1:
    print("Not enough arguments")
    sys.exit(1)

filename = sys.argv[1]

# warm-up
cproc = subprocess.run(filename, capture_output=True, text=True)
out = cproc.stdout
time = float(out.splitlines()[-1].split(' ')[1])
secs, ms = divmod(time*10, 1000)
print("One run will take at most", (time)/1000, "seconds", file=sys.stderr)
print("Time to Complete (seconds):", (time*10)/1000, file=sys.stderr)
dt = datetime.datetime.now() + datetime.timedelta(seconds=secs, milliseconds=ms)
print("Current Time", datetime.datetime.now(), file=sys.stderr)
print("Est. Completion Time:", dt.time(), file=sys.stderr)

# benchmark
benchmarks = []
for i in range(10):
    cproc = subprocess.run(filename, capture_output=True, text=True)
    out = cproc.stdout
    benchmarks.append(out.splitlines()[-1].split(' ')[1])

print("\n".join(benchmarks))