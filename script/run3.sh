#!/bin/bash
# run an toy example for BTM

#dir=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
cd ~/workspace/topic-model/btm/script/

K=1000   # number of topics

alpha=`echo "scale=3;50/$K"|bc`
beta=0.001
niter=1000
save_step=20
has_b=0

input_dir=~/data/stc-data/
output_dir=../output-all-k${K}-fstop/
model_dir=${output_dir}model/

mkdir ${output_dir}
mkdir -p ${output_dir}model 

# the input docs for training
doc_pt=${input_dir}train.1.txt

echo "=============== Index Docs ============="
# docs after indexing
dwid_pt=${output_dir}doc_wids.txt
# vocabulary file
voca_pt=${output_dir}vocab.txt
# filtered words goes to here
filter_pt=${output_dir}filter_words.txt
python indexDocs2.py $doc_pt $dwid_pt $voca_pt $filter_pt

## learning parameters p(z) and p(w|z)
echo "=============== Topic Learning ============="
W=`wc -l < $voca_pt` # vocabulary size
make -C ../src
echo "../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir"
../src/btm est $K $W $alpha $beta $niter $save_step $dwid_pt $model_dir $has_b

## infer p(z|d) for each doc
echo "================ Infer P(z|d)==============="
echo "../src/btm inf sum_b $K $dwid_pt $model_dir"
../src/btm inf sum_b $K $dwid_pt $model_dir

## output top words of each topic
echo "================ Topic Display ============="
python topicAna.py $model_dir $K $voca_pt
