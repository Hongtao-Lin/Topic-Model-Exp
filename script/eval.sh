#!/bin/bash

prefix="output-all"
lprefix="/lustre/home/acct-csyk/csyk/users/htl11/topic-model/btm/log/eval/$prefix"
for it in 400 600 800
do
for f in "stop" "none"
do
python btm.py "$prefix-k50-f$f" $it > "$lprefix-k50-f$f-n$it.out"
python btm.py "$prefix-k50b-f$f" $it > "$lprefix-k50b-f$f-n$it.out"

python btm.py "$prefix-k100b-f$f" $it > "$lprefix-k100b-f$f-n$it.out"

python btm.py "$prefix-k200-f$f" $it > "$lprefix-k200-f$f-n$it.out"
python btm.py "$prefix-k200b-f$f" $it > "$lprefix-k200b-f$f-n$it.out"

python btm.py "$prefix-k500-f$f" $it > "$lprefix-k500-f$f-n$it.out"
python btm.py "$prefix-k500b-f$f" $it > "$lprefix-k500b-f$f-n$it.out"
done
done

