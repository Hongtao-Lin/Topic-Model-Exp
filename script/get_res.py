# coding=utf-8
from __future__ import print_function

ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/"
WORK_DIR = ROOT_DIR + "topic-model/btm/"
MODELS = []
for k in [50, 200, 500]:
    for it in [600, 800, 1000]:
        MODELS.append("output-all-k%d-fstop-n%d"%(k, it))
for f in ["", "b"]:
    MODELS.append("output-all-k50%s-fnone-n800"%(f))

def get_res(in_pt):
    res = []
    f = open(in_pt)
    line = f.readline() 
    # get ppl
    while (not line.startswith("Perplexity")):
        line = f.readline()
    ppl = float(line.split()[-1])

    # get topic coherence
    while (not line.startswith("Topic Coherence")):
        line = f.readline()
    tc = float(line.split()[-1])

    # get topic eval I, II and doc eval
    teval1, teval2 = [], []
    deval = []
    for _ in range(20):
        while (not line.startswith("Display Top and Middle")):
            line = f.readline()
        t1, t2 = line.split()[-2:]
        teval1.append(int(t1))
        teval2.append(int(t2))
        while (not line.startswith("Display Doc")):
            line = f.readline()
        for _ in range(10):
            if not line.strip():
                break
            d = int(line.split()[-1])
            deval.append(d)
            line = f.readline()
    teval1 = sum(teval1) / float(len(teval1))
    teval2 = sum(teval2) / float(len(teval2))
    deval = sum(deval) / float(len(deval))
    res += [ppl, tc, teval1, teval2, deval]
    f.close()
    return res
def main():
    for model in MODELS:
        in_pt = WORK_DIR + "log/" + model + ".out"
        res = get_res(in_pt)
        print(model, res)
    pass

main()
