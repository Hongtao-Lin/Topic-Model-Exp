#!/bin/bash

prefix="output-all"

python btm.py "$prefix-k50-fnone" 800 > "$prefix-k50-fnone-n800.out"
python btm.py "$prefix-k-fnone" 800 > "$prefix-k50b-fstop-n800.out"

for it in {600, 800}
do
python btm.py "$prefix-k50-fstop" $it > "$prefix-k50-fstop-n$it.out"
python btm.py "$prefix-k500-fstop" $it > "$prefix-k500-fstop-n$it.out"
done


