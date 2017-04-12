#!/bin/bash

prefix="output-all"
lprefix="/lustre/home/acct-csyk/csyk/users/htl11/topic-model/btm/log/$prefix"
python btm.py "$prefix-k50-fnone" 800 > "$lprefix-k50-fnone-n800.out"
python btm.py "$prefix-k-fnone" 800 > "$lprefix-k50b-fstop-n800.out"

for it in 600 800
do
python btm.py "$prefix-k50-fstop" $it > "$lprefix-k50-fstop-n$it.out"
python btm.py "$prefix-k500-fstop" $it > "$lprefix-k500-fstop-n$it.out"
done


