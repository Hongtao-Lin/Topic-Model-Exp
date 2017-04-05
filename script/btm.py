#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM

import sys, os, math, time
import codecs
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess


NUM_TOP_WORDS = 20
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
SUFFIX = ".test.pz_d"
ROOT_DIR = "/home/slhome/htl11/workspace/topic-model/btm/"


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
        base_dir = "%s%s/" % (ROOT_DIR, model_str)
        self.K = int(base_dir.split("-k")[-1].split("-")[0])
        voca_pt = base_dir + "vocab.txt"
        self.w2id = word2id(voca_pt)
        self.id2w = id2word(voca_pt)

        model_dir = base_dir + "model/"
        self.pz, self.pw_z = self.load_model(model_dir)
        self.top_words = self.get_top_topic_words(num_top_words=NUM_TOP_WORDS)

    def load_model(self, model_dir):
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
        return top_words


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
        return '%f\t\t%s' % (pz, output.encode("utf8"))

    def disp_topics(self):
        print "Display Topics..."
        print 'p(z)\t\tTop Words'
        for z, pz in sorted(enumerate(self.pz), lambda t: t[1], reverse=True)[:10]:
            print disp_topics(z, pz)

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
        print "Topic Coherence: %.3f" % topic_coherence
        return topic_coherence

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
        pz_d = []
        for sent in sent_list:
            wids = [self.w2id[w] for w in sent.strip().split() if w in self.w2id]
            pz_d.append(self.infer_topic_from_wids(wids))
        return np.array(pz_d)

    def infer_topics_from_file(self, did_pt):
        pz_d = []
        # sent_list = []
        for line in open(did_pt):
            # if len(sent_list) > 100:
                # pz_d.extend(self.infer_topic(sent_list))
                # sent_list = []
            wids = [int(wid) for wid in line.strip().split()]
            print wids[0]
            pz_d.append(self.infer_topic_from_wids(wids))
            # sent_list.append(wids)
            # break
        # pz_d.extend(self.infer_topic(sent_list))
        return np.array(pz_d)

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


def infer_topics_from_file(did_pt, model_dir, K, cal_type="sum_b", suffix=SUFFIX):
    """infer topics of new documents given topic model
    The process will automatically write a doc_id file in the origial doc path
    
    Args:
        did_pt (str): file path for input doc (word idx segmented by space)
        model_dir (str): directory path for model
        K (str): number of topics
        cal_type (str, optional): how to get document-topic distribution by biterms:
            sum_b, sum_w or mix
    
    """
    cmd = ["/home/slhome/htl11/workspace/topic-model/btm/src2/btm", "inf", cal_type, str(K), did_pt, model_dir, suffix]
    # print "running command:", " ".join(cmd)
    t1 = time.time()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    returncode = process.wait()
    print time.time() - t1
    # print returncode
    print process.stdout.read()


if __name__ == '__main__':
    btm = BTM(model_str="output-post-k100-fstop")
    btm = BTM(model_str="output-all-k1000-fstop")
    # btm.disp_topics()
    # print btm
    # btm.disp_doc(u"太假 了 , 不过 创意 不错")
    print btm.infer_topic([u"太假 了 , 不过 创意 不错"])
    print btm.infer_topic([u"我 爱"])
    print btm.infer_topic([u""])
    # did_pt = "/home/slhome/htl11/data/stc-data/valid.1.txt.id"
    # print btm.infer_topics_from_file(did_pt)
