#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM

import sys, os, math, time, random, logging
import codecs
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess


NUM_TOP_WORDS = 20
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
# SUFFIX = ".test.pz_d"
ROOT_DIR = "/home/slhome/htl11/workspace/topic-model/btm/"
MODEL_STR = "output-all-k1000-fstop"

logging.basicConfig(level=logging.DEBUG)

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
    def __init__(self, model_str="output-post-k100-fstop"):
        self.base_dir = "%s%s/" % (ROOT_DIR, model_str)
        self.K = int(self.base_dir.split("-k")[-1].split("-")[0])
        voca_pt = self.base_dir + "vocab.txt"
        self.w2id = word2id(voca_pt)
        self.V = len(self.w2id)
        self.id2w = id2word(voca_pt)

        self.model_dir = self.base_dir + "model/"
        self.pz, self.pw_z = self.load_model(self.model_dir)
        self.top_words = self.get_top_topic_words(num_top_words=NUM_TOP_WORDS)

    def load_model(self, model_dir):
        logging.debug("loading models from %s" % model_dir)
        pz, pw_z = [], []
        z_pt = model_dir + "k%d.pz" % self.K
        wz_pt = model_dir + "k%d.pw_z" % self.K
        with open(z_pt) as f:
            pz = [float(z) for z in f.readline().strip().split()]
            assert len(pz) == self.K

        with open(wz_pt) as f:
            for line in f:
                pw_z.append([float(p) for p in line.strip().split()])

        return np.array(pz), np.array(pw_z)

    def get_top_topic_words(self, num_top_words):
        top_words = []
        for topic_prob in self.pw_z:
            topic_prob = [(i, p) for i, p in enumerate(topic_prob)]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=True)[:num_top_words]
            top_words.append(topic_prob)
        return np.array(top_words)

    def filter_words(self, filter_pt):
        """Filter topic-words according to filter vocab, i.e, stopwords,
        Also adjust vocab mappings accordingly

        Args:
            filter_pt (str): file path for filtered words
        """
        filtered_topics = []
        filter_words = codes.open(filter_pt, encoding="utf8").read().split()
        filter_wids = [self.w2id[w] for w in filter_words if w in self.w2id]
        self.pw_z = np.delete(self.pw_z, filter_wids, axis=1)
        print np.sum(self.pw_z, axis=1)
        self.pw_z /= np.sum(self.pw_z, axis=1)
        self.V -= len(filter_wids)
        for w, wid in zip(filter_words, filter_wids):
            del self.w2id[wid]
            del self.id2w[w]
        assert self.pw_z[1] == self.V

    def disp_doc(self, sent):
        print "Display Doc..."
        print "Ori doc: %s" % sent.encode("utf8")
        _sent = " ".join([w for w in sent.split() if w in self.w2id])
        print "Fit doc: %s" % _sent.encode("utf8")
        wids = [int(self.w2id[w]) for w in _sent.split()]
        pz_d = [(i, p) for i, p in enumerate(self.infer_topic_from_wids(wids))]
        pz_d = sorted(pz_d, key=lambda t: t[1], reverse=True)
        print "Top-2 Topics:"
        for z, pz in pz_d[:2]:
            out_words = " ".join(["%s:%.4f" % (self.id2w[w], p)
                                  for (w, p) in self.top_words[z]])
            print "%.4f" % pz, ":", out_words.encode("utf8")
            print "\n"

    def disp_topic(self, z, pz=0.0):
        output = " ".join(["%s:%.4f" %
                           (self.id2w[w], p) for (w, p) in self.top_words[z][:]])
        print '%f\t\t%s' % (pz, output.encode("utf8"))

    def disp_all_topics(self):
        print "Display Topics..."
        print 'p(z)\t\tTop Words'
        for z, pz in sorted(enumerate(self.pz), lambda t: t[1], reverse=True)[:10]:
            disp_topic(z, pz)

    def disp_topic_coherence(self, z=None, base_k=1000):
        if not z:
            z = random.randint(0, self.K)
        pz = self.pz[z]
        self.disp_topic(z. pz)
        output = " ".join(["%s:%.4f"] %
                          (self.id2w[w], p) for (w, p) in self.pw_z[z][base_k:base_k+NUM_TOP_WORDS])
        print '%f\t\t%s' % (pz, output.encode("utf8"))

    def get_topic_coherence(self, did_pt, num_top_words=10):
        """get topic coherence as a ref metric, see Sec5.1.1 original paper of BTM
        Returns:
            float: topic coherence 
        """
        word_cnt = {}
        biterm_cnt = {}
        # get cnt from wids
        for line in open(did_pt):
            wids = map(int, line.strip().split())
            for wid in wids:
                word_cnt[wid] = word_cnt.get(wid, 0) + 1
            for biterm in get_biterm(wids):
                biterm_cnt[biterm] = biterm_cnt.get(biterm, 0) + 1

        topic_coherence = 0
        # calculate topic coherence:
        for pz, pw_z in enumerate(self.pz, self.pw_z):
            top_words = [w for w, p in self.top_words[:num_top_words]]
            # print voca[top_words[0]]
            for i, wi in enumerate(top_words[1:]):
                for wj in top_words[:i+1]:
                    biterm = Biterm(wi, wj)
                    assert biterm.wj in word_cnt
                    tmp = (biterm_cnt.get(biterm, 0) + 1.0) / word_cnt[biterm.wj]
                    topic_coherence += math.log(tmp)

        topic_coherence /= len(topics)
        # print "Topic Coherence: %.3f" % topic_coherence
        return topic_coherence

    def get_perplexity(self, did_pt):
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
        zd_pt = self.quick_infer_topics_from_file(did_pt)
        # p_unk = 1.0 / self.V
        for pz_d, wids in zip(open(zd_pt), open(did_pt)):
            pz_d = map(float, pz_d.split())  # T
            wids = map(int, wids.strip().split())   # N_m
            for wid in wids:
                if wid >= self.V:
                    continue
                p = sum([pw_z[wid] * pz_d[i] for i, (pz, pw_z) in zip(self.pz, self.pw_z)])
                total_prob += math.log(p)
            # print p, len(wids)
            total_words += len(wids)
        total_prob /= total_words
        ppl = math.exp(-total_prob)

        # print ppl
        return ppl


    def doc2id(self, doc_pt, did_pt=""):
        if not did_pt:
            did_pt = doc_pt + ".id"
        out_f = open(did_pt, "w")
        for line in codecs.open(doc_pt, encoding="utf8"):
            new_line = 
        out_f.close()
        return did_pt

    # Add an additional blank class!
    def infer_topic_from_wids(self, wids):
        pz_d = np.zeros((self.K))
        last_z = 0
        if len(wids) == 0:
            last_z = 1.0
        elif len(wids) == 1:
            pz_d = [self.pz[k]*self.pw_z[k][wids[0]] for k in range(self.K)]
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

    def infer_topic(self, sent_list):
        logging.debug("infering doc from input sentences")
        t1 = time.time()
        pz_d = []
        for sent in sent_list:
            wids = [self.w2id[w] for w in sent.strip().split() if w in self.w2id]
            pz_d.append(self.infer_topic_from_wids(wids))
        logging.debug("time spend: %.3f" % (time.time() - t1))
        return np.array(pz_d)

    def infer_topics_from_file(self, doc_pt, is_raw=False, out_pt=""):
        """Read and write to file the result of inference
        
        Args:
            doc_pt (str): file path for document, either raw or wids
            is_raw (bool, optional): if True, only the doc strings
            out_pt (str, optional): file path for write
        
        Returns:
            str: file path for output file
        """
        # pz_d = []
        # sent_list = []
        logging.debug("infering doc from file")
        if not out_pt:
            out_pt = self.model_dir + "k%d.test.pz_d" % self.K
        out_f = open(out_pt, "w")
        t1 = time.time()
        for line in open(did_pt):
            # if len(sent_list) > 100:
                # pz_d.extend(self.infer_topic(sent_list))
                # sent_list = []
            line_list = line.decode("utf8").strip().split()
            if is_raw:
                line_list = [self.w2id[w] for w in line_list if w in self.wi2d]
            assert len(line_list) != 0

            wids = [int(wid) for wid in line_list]
            # print wids[0]
            pz_d = self.infer_topic_from_wids(wids)
            out = " ".join([str(p) for p in pz_d])
            out_f.write(out + "\n")
            # sent_list.append(wids)
            # break
        # pz_d.extend(self.infer_topic(sent_list))
        out_f.close()
        logging.debug("time spend: %.3f" % (time.time() - t1))
        return out_pt

    def quick_infer_topics_from_file(self, doc_pt, is_raw=False):
        """infer topics of new documents given topic model
        The process will automatically write a doc_id file in the origial doc path
        
        Args:
            doc_pt (str): file path for input doc (word idx / word segmented by space)
            model_dir (str): directory path for model
            K (str): number of topics
            cal_type (str, optional): how to get document-topic distribution by biterms:
                sum_b, sum_w or mix
        
        """
        logging.debug("inferring doc from file using C++")
        if is_raw:
            did_pt = doc_pt + ".id"

        cmd = ["%ssrc2/btm" % ROOT_DIR, "inf", cal_type, str(self.K), doc_pt, self.model_dir, ".test.pz_d"]
        # print "running command:", " ".join(cmd)

        t1 = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        returncode = process.wait()
        logging.debug("time spend: %.3f" % (time.time() - t1))
        # print returncode
        logging.debug(process.stdout.read())
        zd_pt = self.model_dir + "k%d.test.pz_d" % self.K
        return zd_pt

# util funcs
def id2word(pt):
    voca = {}
    for l in codecs.open(pt, encoding="utf8"):
        wid, w = l.strip().split('\t')[:2]
        voca[int(wid)] = w
    return voca


def word2id(pt):
    voca = {}
    for l in codecs.open(pt, encoding="utf8"):
        wid, w = l.strip().split('\t')[:2]
        voca[w] = int(wid)
    return voca


def get_biterm(word_list):
    biterms = []
    for i, wi in enumerate(word_list[:-1]):
        for wj in word_list[i + 1:]:
            biterms.append(Biterm(wi, wj))
    return biterms

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

def transform_doc(doc_pt, w2id, mode)
    filename = doc_pt.split("/")[-1]
    did_pt = ROOT_DIR + "%s.id" % filename
    out_f = open(did_pt, "w")
    for line in open(doc_pt):
        line_list = parse_line(line, w2id, mode=mode)
        output = " ".join([str(wid) for wid in line_list[-1]])
        did_pt.write(output + "\n")
    return did_pt


if __name__ == '__main__':
    voca_pt = ROOT_DIR + MODEL_STR + "/vocab.txt"
    w2id = word2id(voca_pt)
    # btm = BTM(model_str=MODEL_STR)
    DOC_DIR = "/slfs1/users/xyw00/STC2/trigger_knowledge/dmn/data/"

    print transform_doc(DOC_DIR + "q1.valid", w2id, mode=0)
    print transform_doc(DOC_DIR + "q1.train", w2id, mode=0)
    print transform_doc(DOC_DIR + "train.txt", w2id, mode=1)
    # btm.disp_topics()
    # print btm
    # btm.disp_doc(u"太假 了 , 不过 创意 不错")
    # print btm.infer_topic([u"太假 了 , 不过 创意 不错"])
    # print btm.infer_topic([u"我 爱"])
    # print btm.infer_topic([u""])
    # did_pt = "/home/slhome/htl11/data/stc-data/valid.1.txt"
    # doc_pt = "/home/slhome/htl11/data/stc-data/newvalid.txt"
    # doc_pt = "/home/slhome/htl11/data/stc-data/newvalid.txt"
    # print btm.infer_topics_from_file(doc_pt, is_raw=True)
    # print btm.quick_infer_topics_from_file(doc_pt)
    # did_pt = "/home/slhome/htl11/data/stc-data/valid.1.txt.id"
    # print btm.infer_topics_from_file(did_pt)
