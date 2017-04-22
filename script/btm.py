#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM
from __future__ import print_function
import sys, os, math, time, random, logging, json, copy, re
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess


NUM_TOP_WORDS = 50
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
# SUFFIX = ".test.pz_d"
ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/"
# ROOT_DIR = "/slfs1/users/htl11/"
# MODEL_STR = "output-cmnt-k50-fstop"
WORK_DIR = ROOT_DIR + "topic-model/btm/"
# MODEL_STR = "output-cmnt-k40-fstop"
MODEL_STR = "output-all-k50-fstop"
ITER = 500
SRC_NAME = "src/btm"
FILTER_WORDS = (u"不 人 好 小 大 会 才 都 再 还 去 点 太 一个 没 真 上 下 做").split()

DOC_PT = "%s/data/stc-data/valid-btm.txt" % ROOT_DIR
DOC_PT2 = "%s/data/10-news-group/test_clf.txt" % ROOT_DIR
WORD_PT = "%s/data/zhwiki/count_unigram_bd.txt" % ROOT_DIR
BITERM_PT = "%s/data/zhwiki/count_bigram_bd.txt" % ROOT_DIR
WORD_SW_PT = "%s/data/zhwiki/count_unigram_sw.txt" % ROOT_DIR
BITERM_SW_PT = "%s/data/zhwiki/count_bigram_sw.txt" % ROOT_DIR

logging.basicConfig(level=logging.INFO)

class Biterm(namedtuple("Biterm", "wi wj")):
    __slots__ = ()

    def __new__(_cls, wi, wj):
        _wi, _wj = wi, wj
        if wi > wj:
            _wi, _wj = wj, wi
        return tuple.__new__(_cls, (_wi, _wj))

    @property
    def __str__(self):
        return "Biterm: wi=%d, wj=%d" % (self.wi, self.wj)


class BTM(object):

    def __init__(self, model_str="output-all-k50-fstop", it=None):
        self.base_dir = "%s%s/" % (WORK_DIR, model_str)
        self.K = self.base_dir.split("-k")[-1].split("-")[0]
        self.it = it
        if self.K[-1] == "b":
            self.K = int(self.K[:-1])
        else:
            self.K = int(self.K[:])
        voca_pt = self.base_dir + "vocab.txt"
        self.w2id = word2id(voca_pt)
        self.V = len(self.w2id)
        self.id2w = id2word(voca_pt)
        self.fwid = []

        self.model_dir = self.base_dir + "model/"
        self.pz, self.pw_z = self.load_model(self.model_dir)
        self.top_words = self.get_topic_words_from_range(num_words=NUM_TOP_WORDS)

    def load_model(self, model_dir):
        logging.debug("Loading Models from %s" % model_dir)
        pz, pw_z = [], []
        it_suffix = "" if self.it is None else ".%d" % self.it
        z_pt = model_dir + "k%d.pz%s" % (self.K, it_suffix)
        wz_pt = model_dir + "k%d.pw_z%s" % (self.K, it_suffix)
        with open(z_pt) as f:
            pz = [float(z) for z in f.readline().strip().split()]
            assert len(pz) == self.K

        with open(wz_pt) as f:
            for line in f:
                pw_z.append([float(p) for p in line.strip().split()])

        return np.array(pz), np.array(pw_z)

    def get_topic_words_from_range(self, start=0, num_words=20, z=None):
        """sort topic word by their probability in descending order,
        and get topic words in a certain range
        
        args:
            start (int, optional): start of word rank
            num_words (int, optional): number of words to pick
            z (none, optional): if specified, operate only on this topic
        
        """
        if start < 0 or num_words < 0 or start + num_words > self.V:
            raise valueerror("topic word range invalid!")
        if z is not None:
            topic_prob = [(i, p) for i, p in enumerate(self.pw_z[z]) if i not in self.fwid]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=true)[
                start:start + num_words]
            return np.array(topic_prob)

        top_words = []
        for pw_z in self.pw_z:
            # print(pw_z.shape)
            topic_prob = [(i, p) for i, p in enumerate(pw_z) if i not in self.fwid]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=True)[
                start:start + num_words]
            top_words.append(topic_prob)
        return np.array(top_words)

    def filter_words(self, filter_pt):
        """Filter topic-words according to filter vocab, i.e, stopwords, 
        Also filter out manually defined words

        Args:
            filter_pt (str): file path for filtered words
        """
        logging.debug("Filter vocab...")
        filter_words = FILTER_WORDS
        with open(filter_pt) as f:
            filter_words += json.loads(f.read().decode("utf8"))
        filter_words = [w for w in filter_words if w in self.w2id]
        self.fwid = [self.w2id[w] for w in filter_words]
        self.top_words = self.get_topic_words_from_range(num_words=NUM_TOP_WORDS)
        print("n(w) = %d" % (self.pw_z.shape[1] - len(self.fwid)))   # 38090

    def disp_topic(self, z, pz=0.0):
        output = "%.3f" % pz
        output += " : " + " ".join(["%s:%.4f" %
                                    (self.id2w[w], p) for (w, p) in self.top_words[z][:]])
        print(output.encode("utf8") + "\n")

    def disp_all_topics(self):
        start, end = 0, 10
        print("Display Topics of Range %d - %d..." % (start, end))
        print("p(z)\t\tTop Words")
        for z, pz in zip(np.argsort(self.pz), np.sort(self.pz))[::-1][start:end]:
            self.disp_topic(z, pz)

    def disp_top_and_middle_topic(self, z=None, base_k=1000):
        """Display topic by its top words and middle-ranked words
        
        Args:
            z (None, optional): if specified, display only this topic
            base_k (int, optional): starting rank for middle-ranked words
        """
        if z is None:
            z = random.randint(0, self.K - 1)
        print("Display Top and Middle Words of Topic #%d" % z)
        pz = self.pz[z]
        self.disp_topic(z, pz)
        mid_topic_words = self.get_topic_words_from_range(start=base_k, num_words=NUM_TOP_WORDS)
        output = "%.3f" % pz
        output += " : " + " ".join(["%s:%.4f" %
                                    (self.id2w[w], p) for (w, p) in mid_topic_words[z][:]])
        print(output.encode("utf8") + "\n")

    def get_topic_coherence(self, num_top_words=10, cal_type="umass"):
        """get topic coherence as a ref metric, according to cal_type
        
        Args:
            num_top_words (int, optional): number of top words to consider in each topic
            cal_type (str, optional): select different metric type, 
                either: umass, uci, npmi
                see "Exploring the Space of Topic Coherence Measures" for the detailed metric desc
        """
        # step1: get prob from external corpus
        if cal_type in ["umass"]:
            word_pt, biterm_pt = WORD_PT, BITERM_PT
        else:
            word_pt, biterm_pt = WORD_SW_PT, BITERM_SW_PT
        word_prob, biterm_prob = self.load_probs(word_pt, biterm_pt)

        # step2: get word subset segmentation
        subsets = []
        for z in range(self.K):
            top_words = [w for w, p in self.top_words[z][:num_top_words]]
            subset = []
            if cal_type in ["umass"]:
                for i, wi in enumerate(top_words[1:]):
                    for wj in top_words[:i + 1]:
                        subset.append([wi, wj])
            else:
                for i, wi in enumerate(top_words[:-1]):
                    for wj in top_words[i+1:]:
                        subset.append([wi, wj])

        topic_coherences = [] 
        eps = 1e-6
        # step3: calculate topic coherence:
        for z, subset in enumerate(subsets):
            topic_coherence = 0
            for wi, wj in subset:
                biterm = Biterm(wi, wj)
                if cal_type in ["umass"]:
                    topic_coherence += math.log((biterm_prob.get(biterm, 0) + eps) / word_prob[wj])
                elif cal_type in ["npmi"]:
                    tmp = math.log((biterm_prob.get(biterm, 0) + eps) / (word_prob[wi] * word_prob[wj]))
                    topic_coherence += tmp / (-math.log(biterm_prob.get(biterm, 0) + eps))
            topic_coherences.append([z, self.pz[z], topic_coherence])

        sort_topic_coherence = sorted(topic_coherences, lambda k: k[1], reverse=True)    

        coherence = sum(v[-1] for v in topic_coherences) / self.K
        print(sort_topic_coherence)
        return coherence

    def get_perplexity(self, doc_pt, is_raw=False):
        """Perplexity of the test dataset.
        The calculation is as follows:
            prob = \sum_n,m{logsum_t{p(w|z)p(z|d)}} / \sum{n_m}
            ppl = exp{-prob}

        Args:
            did_pt (str): file path for test document (converted to word ids)

        Returns:
            float: perplexity
        """

        total_words, total_prob = 0, 0
        did_pt, zd_pt = self.quick_infer_topics_from_file(doc_pt, is_raw=is_raw, infer_type="prob")
        # p_unk = 1.0 / self.V
        fpw_z = copy.deepcopy(self.pw_z)
        for wid in self.fwid:
            fpw_z[:, wid] = 0
        fpw_z /= np.sum(fpw_z, axis=1, keepdims=True)
        for pz_d, wids in zip(open(zd_pt), open(did_pt)):
            pz_d = map(float, pz_d.split())  # T
            wids = map(int, wids.strip().split())   # N_m
            for wid in wids:
                if wid >= self.V or wid in self.fwid:
                    continue
                p = sum([p[wid] * pz_d[i] for i, p in enumerate(fpw_z)])
                total_prob += math.log(p)
                total_words += 1
            # print(p, len(wids))
        total_prob /= total_words
        ppl = math.exp(-total_prob)
        return ppl

    def load_probs(self, word_pt, biterm_pt):
        word_cnt = {}
        biterm_cnt = {}
        with open(word_pt) as f:
            for line in f.xreadlines():
                w, cnt = line.decode("utf8").split("\t")
                if w not in self.w2id:
                    continue
                word_cnt[self.w2id[w]] = float(cnt)
        with open(biterm_pt) as f:
            for line in f.xreadlines():
                wi, wj, cnt = line.decode("utf8").split("\t")
                if wi not in self.w2id or wj not in self.w2id:
                    continue
                biterm = Biterm(self.w2id[wi], self.w2id[wj])
                biterm_cnt[biterm] = float(cnt)
        return word_cnt, biterm_cnt

    def _doc2id(self, doc_pt, did_pt=""):
        if not did_pt:
            did_pt = doc_pt + ".id"
        out_f = open(did_pt, "w")
        for line in open(doc_pt):
            line = " ".join([str(self.w2id[w]) for w in line.decode("utf8").strip().split()
                             if w in self.w2id])
            out_f.write(line.encode("utf8") + "\n")
        out_f.close()
        return did_pt

    # Add an additional blank class!
    def infer_topic_from_wids(self, wids, infer_type="prob"):
        pz_d = np.zeros((self.K))
        last_z = 0
        if len(wids) == 0:
            last_z = 1.0
        elif len(wids) == 1:
            pz_d = [self.pz[k] * self.pw_z[k][wids[0]] for k in range(self.K)]
        else:
            for biterm in get_biterm(wids):
                wi, wj = biterm.wi, biterm.wj
                pz_b = np.zeros((self.K))
                for k in range(self.K):
                    assert self.pw_z[k][wi] > 0 and self.pw_z[k][wj] > 0
                    pz_b[k] = self.pz[k] * self.pw_z[k][wi] * self.pw_z[k][wj]
                pz_b = pz_b / np.sum(pz_b)
                pz_d += pz_b
            pz_d = pz_d / np.sum(pz_d)

        pz_d = np.append(pz_d, last_z)
        if infer_type == "max_idx":
            pz_d = np.argmax(pz_d)
        return pz_d

    def infer_topics(self, sent_list, is_raw=False, infer_type="prob"):
        """Infer doc-topic distribution from a list of sentences

        Args:
            sent_list (list): a list of sentences, word/wordid seperated by space
            is_raw (bool, optional): if is_raw: the sentences are raw strings
            infer_type (str, optional): whether to get the max topic id or all prob

        Returns:
            np.array: array of output distribution
        """
        logging.debug("Infering Doc from Input Sentences")
        t1 = time.time()
        pz_d = []
        for sent in sent_list:
            if is_raw:
                wids = [self.w2id[w] for w in sent.strip().split()
                        if w in self.w2id]
            else:
                wids = [int(w) for w in sent.strip().split() if int(w) < self.V]
            pz_d.append(self.infer_topic_from_wids(wids, infer_type=infer_type))
        logging.debug("Time Spend: %.3f" % (time.time() - t1))
        return np.array(pz_d)

    def quick_infer_topics(self, sent_list, is_raw=False, cal_type="sum_b", infer_type="prob"):
        assert infer_type in ["max_idx", "prob"]

        wids = []
        for sent in sent_list:
            if is_raw:
                wids.append([self.w2id[w] for w in sent.strip().split()
                             if w in self.w2id])
            else:
                wids.append([int(w) for w in sent.strip().split() if int(w) < self.V])

        did_pt = self.model_dir + "tmp_doc.id"
        with open(did_pt, "w") as f:
            for wid in wids:
                output = " ".join([str(w) for w in wid])
                f.write(output + "\n")

        did_pt, zd_pt = self.quick_infer_topics_from_file(
            did_pt, is_raw=False, cal_type=cal_type, infer_type=infer_type)

        res_list = []
        with open(zd_pt) as f:
            for line in f.xreadlines():
                if infer_type == "prob":
                    res = [float(p) for p in line.strip().split()]
                elif infer_type == "max_idx":
                    res = int(line.strip())
                res_list.append(res)

        return np.array(res_list)

    def quick_infer_topics_from_file(self, doc_pt, is_raw=False,
                                     cal_type="sum_b", infer_type="prob", suffix="pz_d"):
        """infer topics of new documents given topic model using the origianl C++ executable

        Args:
            doc_pt (str): file path for input doc (word/id segmented by space)
                format: each line a list of word/id seperated by space
            is_raw (bool, optional): if true, we will generate a doc_id file inplace
            cal_type (str, optional): how to get document-topic distribution by biterms:
                sum_b, sum_w or mix
            infer_type (str, optional): whether to return a max topic_id or a distribution
                Either "max_idx" or "prob"
            suffix (str, optional): suffix of the output file

        Returns:
            str: file path for word ids
            str: file path for output topic-doc distribution
        """
        logging.debug("Inferring Doc from File using C++")
        filename = doc_pt.split("/")[-1]
        if is_raw:
            did_pt = self.model_dir + "%s.id" % filename
            did_pt = self._doc2id(doc_pt, did_pt)
        else:
            did_pt = doc_pt
        suffix = ".%s.%s" % (filename, suffix)
        zd_pt = self.model_dir + "k%d%s" % (self.K, suffix)

        cmd = ["%s%s" % (WORK_DIR, SRC_NAME), "inf", "sum_b",
               str(self.K), did_pt, self.model_dir, suffix, infer_type, str(self.it)]
        logging.debug("Running Command: " + " ".join(cmd))
        t1 = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # returncode = process.wait()
        logging.debug("Time Spend: %.3f" % (time.time() - t1))
        # print(returncode)
        logging.debug(process.stdout.read())
        return did_pt, zd_pt

    def get_doc_coherence(self, prob_list, label_list, cal_type=["nmi", "purity"]):
        """Get document clustering measurement from other open-source project

        Args:
            cal_type (str, optional): either nmi, purity, 

        Returns:
            str: Description
        """
        cluster_pt = "%stmp_cluster.txt" % (self.base_dir)
        label_pt = "%stmp_label.txt" % (self.base_dir)
        with open(cluster_pt, "w") as f:
            cluster_idx = np.argmax(prob_list, axis=1)
            for i, idx in enumerate(cluster_idx):
                f.write("%d\t%d\n" % (i, idx))
        with open(label_pt, "w") as f:
            for i, idx in enumerate(label_list):
                f.write("%d\t%d\n" % (i, idx))
        cmd = ["python", "cluster_eval.py"] + ["-%s" % cal for cal in cal_type] + \
              [label_pt, cluster_pt]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = process.stdout.read()
        logging.debug(result)
        scores = {}
        for cal in cal_type:
            if cal == "purity":
                score1 = float(re.search(r"macro purity = (.*)\n", result).group(1))
                score2 = float(re.search(r"micro purity = (.*)\n", result).group(1))
                scores.update({"macro-purity": score1, "micro-purity": score2})
                # get detailed score
                i = 0
                while not result.split("\n")[i].startswith("detailed purity stat"):
                    i += 1
                print("Detailed purity stat:")
                for k in range(i+1, i+1+self.K):
                    if len(result.split("\n")[k].split()) != 3:
                        continue
                    z, score, cluster_size = result.split("\n")[k].split()
                    print(z, self.pz[int(z)], score, cluster_size)
            else:
                scores[cal] = float(re.search(r"%s = (.*)\n" % cal, result).group(1))
        return scores

    def disp_doc(self, sent):
        print("Display Topics for Doc: %s" % sent.encode("utf8"))
        _sent = " ".join([w for w in sent.split() if w in self.w2id])
        print("Fit doc: %s" % _sent.encode("utf8"))
        wids = [int(self.w2id[w]) for w in _sent.split()]
        pz_d = [(i, p) for i, p in enumerate(self.infer_topic_from_wids(wids))]
        pz_d = sorted(pz_d, key=lambda t: t[1], reverse=True)
        print("Top-2 Topics:")
        for z, pz in pz_d[:2]:
            out_words = " ".join(["%s:%.4f" % (self.id2w[w], p)
                                  for (w, p) in self.top_words[z]])
            print("%.4f" % pz, ":", out_words.encode("utf8"))
            # print("\n")


# util funcs
def id2word(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.decode("utf8").strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca


def word2id(pt):
    voca = {}
    for l in open(pt):
        wid, w = l.decode("utf8").strip().split('\t')[:2]
        voca[w] = int(wid)
    return voca


def get_biterm(word_list):
    biterms = []
    for i, wi in enumerate(word_list[:-1]):
        for wj in word_list[i + 1:]:
            biterms.append(Biterm(wi, wj))
    return biterms


def get_normal_samples(k):
    idxs = []
    for i in range(4):
        sidx = int(math.floor(k * i / 4))
        idxs += range(sidx, sidx + 5)
    return idxs


# tmp files
def parse_line(line, w2id, mode=0):
    """parse line according to line format

    Args:
        line (str): input line
        mode (int, optional):
            0: q1.valid/train
            1: valid/train.txt

    Returns:
        str: line_list output
    """
    if not line.strip():
        return ""
    line = line.decode("utf8").strip()
    line_list = []
    if mode == 0:
        tmp_list = line.split("\t")
        line_list.append(int(tmp_list[0].split()[-1]))
        line_list.append([int(wid) for wid in tmp_list[1].split()])
    elif mode == 1:
        tmp_list = line.split("\t")
        line_list.append(int(tmp_list[0].split()[-1]))
        line_list.append([int(w2id[w]) for w in tmp_list[1].split() if w in w2id])

    return line_list

def transform_doc(doc_pt, w2id, mode):
    filename = doc_pt.split("/")[-1]
    did_pt = WORK_DIR + "%s.id" % filename
    out_f = open(did_pt, "w")
    for line in open(doc_pt):
        line_list = parse_line(line, w2id, mode=mode)
        output = " ".join([str(wid) for wid in line_list[-1]])
        out_f.write(output + "\n")
    return did_pt


if __name__ == '__main__':
    if len(sys.argv) >= 3:
        MODEL_STR = sys.argv[1]
        ITER = int(sys.argv[2])

    filter_pt = "%s/res/zh-stopwords.json" % ROOT_DIR
    voca_pt = WORK_DIR + MODEL_STR + "/vocab.txt"
    # DOC_DIR = "/slfs1/users/xyw00/STC2/trigger_knowledge/dmn/data/"
    print("Evaluating model: %s %s" % (MODEL_STR, ITER))

    btm = BTM(model_str=MODEL_STR, it=ITER)
    # print(transform_doc(DOC_DIR + "q1.valid", w2id, mode=0))
    # print(transform_doc(DOC_DIR + "q1.train", w2id, mode=0))
    # print(transform_doc(DOC_DIR + "train.txt", w2id, mode=1))
    # print("Evaluating model: %s %d" % (MODEL_STR, ITER))
    btm.filter_words(filter_pt)

    # print("Human Evaluation I:")
    # for k in get_normal_samples(btm.K):
    #     btm.disp_top_and_middle_topic(k)

    # print("Perplexity:", btm.get_perplexity(DOC_PT, is_raw=True))
    print("Topic Coherence")
    # topic_metric = ["umass", "npmi"]
    topic_metric = []
    for metric in topic_metric:
        print("%s" % metric, btm.get_topic_coherence(cal_type=metric))
    print("")

    # get sample topics in order
    topic_idxs = np.argsort(-btm.pz)[get_normal_samples(btm.K)]
    topic_dict = {}
    for k in topic_idxs:
        topic_dict[k] = []

    sent_list = []
    label_list = []
    with open(DOC_PT2) as f:
        for line in f.readlines():
            sent, label = line.decode("utf8").strip().split("\t")
            sent_list.append(sent)
            label_list.append(int(label))
    prob_list = btm.quick_infer_topics(sent_list, is_raw=True, infer_type="prob")
    scores = btm.get_doc_coherence(prob_list, label_list)
    print("Doc Coherence:")
    for k, v in scores.items():
        print(k, v)
    print("")
    # print("Display Docs:")
    # nmi = btm.get_doc_
    # idx_list = (-prob_list).argsort()[:, :2]
    # perm = np.random.permutation(len(sent_list))
    # for i in perm:
    #     k = int(idx_list[i][0])
    #     if k not in topic_dict:
    #         continue
    #     if len(topic_dict[k]) >= 10:
    #         continue
    #     topic_dict[k].append([zip(prob_list[i, idx_list[i]], idx_list[i]), sent_list[i]])
    # for i, k in enumerate(topic_idxs):
    #     btm.disp_top_and_middle_topic(k)
    #     for entry in topic_dict[k]:
    #         info_str = " ".join(["%.3f:%d" % (prob, idx) for prob, idx in entry[0]])
    #         output = "Display Doc:\t" + info_str + "\t" + entry[1]
    #         print(output.encode("utf8"))
    #     print("\n")
    pass
