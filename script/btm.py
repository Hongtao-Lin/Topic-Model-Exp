#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM
from __future__ import print_function
import sys, os, math, time, random, logging, json, copy
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess


NUM_TOP_WORDS = 50
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
# SUFFIX = ".test.pz_d"
ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/topic-model/btm/"
#MODEL_STR = "output-cmnt-k50-fstop"
MODEL_STR = "output-all-k50-fstop"
ITER = 700
SRC_NAME = "src/btm"
FILTER_WORDS = (u"不 人 好 小 大 会 才 都 再 还 去 点 太 一个 没 真 上 下 做").split()

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

    def __init__(self, model_str="output-post-k100-fstop", it=None):
        self.base_dir = "%s%s/" % (ROOT_DIR, model_str)
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
        logging.debug("loading models from %s" % model_dir)
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
        if start < 0 or num_words < 0 or start + num_words > self.V:
            raise ValueError("Topic word range invalid!")
        if z is not None:
            topic_prob = [(i, p) for i, p in enumerate(self.pw_z[z]) if i not in self.fwid]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=True)[start:start + num_words]
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
        print("n(w) = %d" % (self.pw_z.shape[1]-len(self.fwid)))   # 38090

    def disp_topic(self, z, pz=0.0):
        output = "%.3f" % pz
        output += " : " + " ".join(["%s:%.4f" %
                                    (self.id2w[w], p) for (w, p) in self.top_words[z][:]])
        print(output + "\n")

    def disp_all_topics(self):
        start, end = 0, 10
        print("Display Topics of Range %d - %d..." % (start, end))
        print("p(z)\t\tTop Words")
        for z, pz in zip(np.argsort(self.pz), np.sort(self.pz))[::-1][start:end]:
            self.disp_topic(z, pz)

    def disp_topic_coherence(self, z=None, base_k=1000):
        if not z:
            z = random.randint(0, self.K-1)
        print("Display Top and Middle Words of Topic #%d" % z)
        pz = self.pz[z]
        self.disp_topic(z, pz)
        mid_topic_words = self.get_topic_words_from_range(start=base_k, num_words=NUM_TOP_WORDS)
        output = "%.3f" % pz
        output += " : " + " ".join(["%s:%.4f" %
                                    (self.id2w[w], p) for (w, p) in mid_topic_words[z][:]])
        print(output + "\n")

    def get_topic_coherence(self, doc_pt, num_top_words=10, is_raw=False):
        """get topic coherence as a ref metric, see Sec5.1.1 original paper of BTM
        Returns:
            float: topic coherence
        """
        filename = doc_pt.split("/")[-1]
        if is_raw:
            did_pt = self.model_dir + "%s.id" % filename
            did_pt = self._doc2id(doc_pt, did_pt)
        else:
            did_pt = doc_pt

        word_cnt = {}
        biterm_cnt = {}
        # get cnt from wids
        for line in open(did_pt):
            wids = map(int, line.strip().split())
            wids = [wid for wid in wids if wid not in self.fwid]
            for wid in wids:
                word_cnt[wid] = word_cnt.get(wid, 0) + 1
            for biterm in get_biterm(wids):
                biterm_cnt[biterm] = biterm_cnt.get(biterm, 0) + 1

        topic_coherence = 0
        # calculate topic coherence:
        for z in range(self.K):
            top_words = [w for w, p in self.top_words[z][:num_top_words]]
            for i, wi in enumerate(top_words[1:]):
                for wj in top_words[:i + 1]:
                    biterm = Biterm(wi, wj)
                    if biterm.wj not in word_cnt:
                        # print(biterm.wj)
                        continue
                    tmp = (biterm_cnt.get(biterm, 0) + 1.0) / word_cnt[biterm.wj]
                    topic_coherence += math.log(tmp)

        topic_coherence /= self.K
        # print("Topic Coherence: %.3f" % topic_coherence)
        return topic_coherence

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
        logging.debug(total_words)
        ppl = math.exp(-total_prob)
        return ppl

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
    def infer_topic_from_wids(self, wids):
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
        return pz_d

    def infer_topic(self, sent_list, is_raw=True):
        """Infer doc-topic distribution from a list of sentences

        Args:
            sent_list (list): a list of sentences, word/wordid seperated by space
            is_raw (bool, optional): if is_raw: the sentences are raw strings

        Returns:
            np.array: array of output distribution
        """
        logging.debug("infering doc from input sentences")
        t1 = time.time()
        pz_d = []
        for sent in sent_list:
            if is_raw:
                wids = [self.w2id[w] for w in sent.strip().split() 
                        if w in self.w2id]
            else:
                wids = [int(w) for w in sent.strip().split() if int(w) < self.V]
            pz_d.append(self.infer_topic_from_wids(wids))
        logging.debug("time spend: %.3f" % (time.time() - t1))
        return np.array(pz_d)

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
        logging.debug("inferring doc from file using C++")
        filename = doc_pt.split("/")[-1]
        if is_raw:
            did_pt = self.model_dir + "%s.id" % filename
            did_pt = self._doc2id(doc_pt, did_pt)
        else:
            did_pt = doc_pt
        suffix = ".%s.%s" % (filename, suffix)
        zd_pt = self.model_dir + "k%d%s" % (self.K, suffix)

        cmd = ["%s%s" % (ROOT_DIR, SRC_NAME), "inf", "sum_b",
               str(self.K), did_pt, self.model_dir, suffix, infer_type, str(self.it)]
        logging.debug("running command:", " ".join(cmd))
        logging.debug(" ".join(cmd))
        t1 = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # returncode = process.wait()
        logging.debug("time spend: %.3f" % (time.time() - t1))
        # print(returncode)
        logging.debug(process.stdout.read())
        return did_pt, zd_pt

    def disp_doc(self, sent):
        print("Display Doc...")
        print("Ori doc: %s" % sent.encode("utf8"))
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
        sidx = int(math.floor(k*i/4))
        idxs += range(sidx, sidx+5)
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
    did_pt = ROOT_DIR + "%s.id" % filename
    out_f = open(did_pt, "w")
    for line in open(doc_pt):
        line_list = parse_line(line, w2id, mode=mode)
        output = " ".join([str(wid) for wid in line_list[-1]])
        out_f.write(output + "\n")
    return did_pt


if __name__ == '__main__':
    if len(sys.argv) > 3:
        MODEL_STR = sys.argv[1]
        ITER = sys.argv[2]
    # voca_pt = ROOT_DIR + MODEL_STR + "/vocab.txt"
    # w2id = word2id(voca_pt)
    # DOC_DIR = "/slfs1/users/xyw00/STC2/trigger_knowledge/dmn/data/"

    # print(transform_doc(DOC_DIR + "q1.valid", w2id, mode=0))
    # print(transform_doc(DOC_DIR + "q1.train", w2id, mode=0))
    # print(transform_doc(DOC_DIR + "train.txt", w2id, mode=1))
    
    print("Evaluating model: %s %d" % (MODEL_STR, ITER)) 
    btm = BTM(model_str=MODEL_STR, it=ITER) 
    filter_pt = "/lustre/home/acct-csyk/csyk/users/htl11/res/zh-stopwords.json"
    btm.filter_words(filter_pt)

    print("Human Evaluation I:")
    # for k in get_normal_samples(btm.K):
    #     btm.disp_topic_coherence(k)

    doc_pt = "/lustre/home/acct-csyk/csyk/users/htl11/data/stc-data/valid-btm.txt"
    
    print("Perplexity:", btm.get_perplexity(doc_pt, is_raw=True))
    print("Topic Coherence:", btm.get_topic_coherence(doc_pt, is_raw=True))

    # btm.disp_doc(u"我 爱 北京 天安门")
    pass

