#!/bin/bash

prefix="output-all"
lprefix="/lustre/home/acct-csyk/csyk/users/htl11/topic-model/btm/log/eval/$prefix"
python btm.py "$prefix-k50-fnone" 800 > "$lprefix-k50-fnone-n800.out"
python btm.py "$prefix-k50b-fnone" 800 > "$lprefix-k50b-fnone-n800.out"

for it in 600 800 1000
do
python btm.py "$prefix-k50-fstop" $it > "$lprefix-k50-fstop-n$it.out"
python btm.py "$prefix-k200-fstop" $it > "$lprefix-k200-fstop-n$it.out"
python btm.py "$prefix-k500-fstop" $it > "$lprefix-k500-fstop-n$it.out"
done


