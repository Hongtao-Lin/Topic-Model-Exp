# coding=utf-8
from __future__ import print_function
import itertools
import os, sys, re, json, logging

ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/"
WORK_DIR = ROOT_DIR + "topic-model/btm/"
MODELS = []
for k in [50, 100, 200, 500]:
    # for it in [400, 600, 800]:
    for it in [400]:
        for f in ["stop", "none"]:
            MODELS.append("output-all-k%d-f%s-n%d" % (k, f, it))
            MODELS.append("output-all-k%db-f%s-n%d" % (k, f, it))
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
    while (not line.startswith("Topic Coherence")):
        line = f.readline()
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

    # get topic eval I, II and doc eval
    # teval1, teval2 = [], []k
    # deval = []
    # for _ in range(20):
    #     while (not line.startswith("Display Top and Middle")):
    #         line = f.readline()
    #     t1, t2 = list(line.strip().replace(" ", "")[-2:])
    #     #print(line)
    #     t1, t2 = int(t1), int(t2)
    #     teval1.append(int(t1))
    #     teval2.append(int(t2))
    #     _deval = []
    #     assert t1 in [0, 1] and t2 in [0, 1]
    #     if t1 == 0 and t2 == 0:
    #         _deval = [0] * 10
    #         deval.append(_deval)
    #         line = f.readline()
    #         continue
    #     while (not line.startswith("Display Doc")):
    #         line = f.readline()
    #     for _ in range(10):
    #         if not line.strip():
    #             break
    #         d = int(line.strip().replace(" ", "")[-1])
    #         _deval.append(d)
    #         line = f.readline()
    #     deval.append(_deval)
    # scores = []
    # for i in range(4):
    #     score = []
    #     s, e = len(teval1)*i / 4, len(teval1)*(i+1) / 4
    #     score.append(sum(teval1[s:e])/float(e-s))
    #     score.append(sum(teval2[s:e])/float(e-s))
    #     tmp = list(itertools.chain(*deval[s:e]))
    #     score.append(sum(tmp)/float(len(tmp)))
    #     scores.append(score)
    # teval1 = sum(teval1) / float(len(teval1))
    # teval2 = sum(teval2) / float(len(teval2))
    # tmp = list(itertools.chain(*deval))
    # deval = sum(tmp) / float(len(tmp))
    # for score in scores:
    #     output = " | ".join(["%.4f" % v for v in score])
    #     print(output)
    # res += [teval1, teval2, deval]

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
    print(topic_cnt)
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


def main():
    print(ALL_RES)
    for model in MODELS:
        in_pt = WORK_DIR + "log/eval/" + model + ".out"
        if os.path.exists(in_pt) and os.path.getsize(in_pt) > 1024:
            print(model)
            res = get_res(in_pt)
    pass

main()
