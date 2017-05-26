# coding=utf-8
from __future__ import print_function
import itertools
import os, sys, re, json, logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/"
ROOT_DIR = "/slfs1/users/htl11/"
WORK_DIR = ROOT_DIR + "topic-model/btm/"
MODELS = []
for k in [50, 100, 200, 500]:
    # for it in [400, 600, 800]:
    for it in [600]:
        for f in ["stop", "none"]:
            MODELS.append("output-all-k%d-f%s-n%d" % (k, f, it))
            MODELS.append("output-all-k%db-f%s-n%d" % (k, f, it))
MODELS = ["output-all-k500-fstop-n600"]
ALL_RES = ["PPL", "NPMI", "Umass", "Macro Purity", "Micro Purity", "NMI"]

def get_res(in_pt):
    res = []
    f = open(in_pt)
    line = f.readline()
    ppl = 0.0
    # get ppl
    while (not line.startswith("Perplexity")):
        line = f.readline()
    ppl = float(line.split()[-1])
    res.append(ppl)

    # get topic coherence
    topic_dist = []
    while (not line.startswith("Topic Coherence")):
        line = f.readline()
    while True:
        line = f.readline()
        if line.startswith("[[") and topic_dist == []:
            topic_dist = json.loads(line.strip())
        if not line.strip():
            break
        if len(line.split()) != 2:
            continue
        metric, score = line.split()
        metric = metric.strip()
        score = float(score.strip())
        res.append(score)

    print(topic_dist)
    if topic_dist != []:
        pz = [l[1] for l in topic_dist]
        score = [l[2] for l in topic_dist]
        fig, ax = plt.subplots()
        ax.bar(range(len(pz)), pz, label="pz") 
        ax.plot(range(len(pz)), score, color="purple", marker="o", label="npmi") 
        fig.savefig("npmi.png")

    # get doc eval
    while (not line.startswith("Doc Coherence")):
        line = f.readline()
    new_purity = 0
    topic_cnt = 0
    while True:
        line = f.readline()
        if len(line.split()) != 4:
            break
        z, pz, p, cz = line.strip().split()
        if int(cz) > 5:
            new_purity += float(p)
            topic_cnt += 1
    new_purity /= topic_cnt
    res.append(new_purity)
    # print(topic_cnt)
    while True:
        line = f.readline()
        if not line.strip():
            break
        if len(line.split()) != 2:
            continue
        metric, score = line.split()
        metric = metric.strip()
        score = float(score.strip())
        res.append(score)

    output = " | ".join(["%.4f" % v for v in res])
    print(output)
    f.close()
    return res


def analyze_top_words(in_pt, data_pt, post_pt, cmnt_pt):
    words = set()
    word_list = []
    with open(in_pt) as f:
        for line in f.xreadlines():
            top_words = line.decode("utf8").strip().split()
            assert len(top_words) == 200
            words.update(top_words)
            word_list.append(top_words)
    with open(data_pt) as fd, open(post_pt) as fp, open(cmnt_pt) as fc:
        for ld in fd.xreadlines():
            lp, lc = fp.readline(), fc.readline()
            
    print(len(words))


def main():
    # print(ALL_RES)
    # for model in MODELS:
    #     in_pt = WORK_DIR + "log/eval/" + model + ".out"
    #     if os.path.exists(in_pt) and os.path.getsize(in_pt) > 1024:
    #         print(model)
    #         res = get_res(in_pt)
    data_pt = ROOT_DIR + "data/stc-kw-data/train.txt"
    post_pt = ROOT_DIR + "data/stc-kw-data/train-btm.post.txt"
    cmnt_pt = ROOT_DIR + "data/stc-kw-data/train-btm.cmnt.txt"
    analyze_top_words(WORK_DIR + "output-all-k200-fstop/model/k200.top_words", data_pt, post_pt, cmnt_pt)
    analyze_top_words(WORK_DIR + "output-all-k200-fstop/model/k200.general.top_words", data_pt, post_pt, cmnt_pt)
    pass
         
main()
