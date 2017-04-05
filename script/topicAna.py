#!/usr/bin/env python
# coding=utf-8
# Function: translate the results from BTM
# Example run:
# python topicAna.py ~/workspace/topic-model/btm/output-post-k100-fstop/model/ 100 ~/workspace/topic-model/btm/output-post-k100-fstop/vocab.txt ~/data/stc-data/valid.1.txt
# python topicAna.py ../output-post-k100-fstop/model/ 100 ../output-post-k100-fstop/vocab.txt ~/data/stc-data/valid.1.txt

import sys, os, math, time
import codecs
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy import stats, integrate
# plt.switch_backend('agg')

from indexDocs import write_doc2id

NUM_TOP_WORDS = 100
NUM_SAMPLES_DISPLAY = 10
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
FILTER_PATH = ''
SUFFIX = ".test.pz_d"
ROOT_DIR = '~/workspace/topic-model/btm/'

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
        base_dir = "/home/slhome/htl11/workspace/topic-model/btm/%s/" % model_str
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

    def infer_topic_from_wids(self, wids):
        pz_d = np.zeros((self.K))

        if len(wids) == 1:
            pz_d = [self.pz[k]*self.pw_z[k][self.voca.get(wids[0], 0)] for k in self.K]

        for biterm in get_biterm(wids):
            wi, wj = biterm.wi, biterm.wj
            pz_b = np.zeros((self.K))
            for k in range(self.K):
                assert self.pw_z[k][wi] > 0 and self.pw_z[k][wj] > 0
                pz_b[k] = self.pz[k] * self.pw_z[k][wi] * self.pw_z[k][wj]
            pz_b = pz_b / np.sum(pz_b)
            pz_d += pz_b
        pz_d = pz_d / np.sum(pz_d)
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
            print wids
            pz_d.append(self.infer_topic_from_wids(wids))
            # sent_list.append(wids)
        # pz_d.extend(self.infer_topic(sent_list))
            break
        return np.array(pz_d)

    def disp_doc(self, sent):
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

    def disp_topics(self):
        print 'p(z)\t\tTop Words'
        for z, pz in sorted(enumerate(self.pz), lambda t: t[1], reverse=True)[:10]:
            output = " ".join(["%s:%.4f" %
                               (self.id2w[w], p) for (w, p) in self.top_words[z][:]])
            print '%f\t%s' % (pz, output.encode("utf8"))

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
        return topic_coherence

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


def get_topics(wz_pt, pz):
    """Read data from saved topic model

    Args:
        wz_pt (string): file path for topic-word distribution
        pz (string): file path for topic distribution

    Returns:
        list: format like:
        [
            (topic_prob, {wordID: prob}
        ]
    """
    topics = []
    for i, l in enumerate(open(wz_pt)):
        vs = [float(v) for v in l.split()]
        wvs = zip(range(len(vs)), vs)
        # wvs = sorted(wvs,  =lambda d: d[1], reverse=True)
        wvs = OrderedDict(sorted(wvs, key=lambda t: t[1], reverse=True))
        topics.append((pz[i], wvs))
    return topics


def filter_topics(topics, voca, filter_pt):
    """Filter topic-words according to filter vocab, i.e, stopwords,
    mainly for display consideration.

    Returns:
        list: same format as "topics"
    """
    filtered_topics = []
    filter_words = []

    with codecs.open(filter_pt, encoding="utf8") as f_pt:
        filter_words = f_pt.read().split()
    # decide whether to filter out.
    for pz, topic_words in topics[:]:
        rest_words = []
        for (wid, p) in topic_words:
            if voca[wid] not in filter_words:
                rest_words.append((w, p))
        rest_words = OrderedDict(sorted(rest_words[:],
                                        key=lambda t: t[1], reverse=True)).items()
        filtered_topics.append((pz, rest_words))

    return filtered_topics


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


def get_perplexity(did_pt, zd_pt, topics):
    """Perplexity of the test dataset.
    The calculation is as follows:
        prob = \sum_n,m{logsum_t{p(w|z)p(z|d)}} / \sum{n_m}
        ppl = exp{-prob}

    Args:
        did_pt (str): file path for test document (converted to word ids)
        zd_pt (str): file path for topic-doc distribution
        topics (list): a list for storing word-topic distribution 

    Returns:
        float: perplexity
    """
    total_words, total_prob = 0, 0
    p_unk = 1.0 / len(topics[0][1].keys())
    for pz_d, wids in zip(open(zd_pt), open(did_pt)):
        pz_d = map(float, pz_d.split())  # T
        wids = map(int, wids.strip().split())   # N_m
        for wid in wids:
            p = sum([pw_z.get(wid, p_unk) * pz_d[i] for i, (pz, pw_z) in enumerate(topics)])
            total_prob += math.log(p)
        # print p, len(wids)
        total_words += len(wids)
    total_prob /= total_words
    ppl = math.exp(-total_prob)

    # print ppl
    return ppl


def get_topic_choherence(did_pt, topics, voca, num_top_words=20):
    """get topic coherence as a ref metric, see Sec5.1.1 original paper of BTM
    
    Args:
        did_pt (str): file path for test document id
        topics (list): list of topic-word distribution 
        voca (dict): mapping from wordid to word
        num_top_words (int, optional): number of top words per topic as reference
    
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
    for pz, pw_z in topics:
        top_words = [w for w, p in pw_z.items()[:num_top_words]]
        # print voca[top_words[0]]
        for i, wi in enumerate(top_words[1:]):
            for wj in top_words[:i+1]:
                biterm = Biterm(wi, wj)
                assert biterm.wj in word_cnt
                tmp = (biterm_cnt.get(biterm, 0) + 1.0) / word_cnt[biterm.wj]
                topic_coherence += math.log(tmp)

    topic_coherence /= len(topics)
    return topic_coherence


def disp_topics(topics, voca, num_top_words):
    """Display topics in descending order of prob
    
    Args:
        topics (list): list of topic-word distribution 
        voca (dict): dict of wordid to word
        num_top_words (int): num of top words for display in each topic
    
    Returns:

    """
    print 'p(z)\t\tTop words'
    for pz, pw_z in sorted(topics, reverse=True)[:10]:
        output = " ".join(["%s:%.4f" %
                           (voca[w], p) for (w, p) in pw_z.items()[:num_top_words]])
        print '%f\t%s' % (pz, output.encode("utf8"))


# def disp_docs(did_pt, zd_pt, voca, topics, num_samples, num_top_words=10, d_pt=""):

#     num_docs = sum(1 for i in open(zd_pt))
#     idxs=np.random.choice(num_docs, num_samples, replace=False)
#     idxs=sorted(idxs)
#     ptr=0
#     did_f=open(did_pt)
#     if d_pt:
#         d_f=open(d_pt)
#     for i, pz_d in enumerate(open(zd_pt)):
#         pz_d=map(float, pz_d.split())
#         wids=map(int, did_f.readline().strip().split())
#         if ptr == num_samples or i != idxs[ptr]:
#             continue
#         else:
#             ptr += 1
#             if d_f:
#                 ori_doc=d_f.readline()
#                 print "Ori doc:", ori_doc.split("=>")[0].strip()
#             doc=" ".join([voca.get(w, 'UNK') for w in wids])
#             print "Fit doc:", doc.encode("utf8")
#             print "Top-10 topic prob: %s" % " ".join(map(str, sorted(pz_d, reverse=True)[:10]))

#             # display topic words for this doc
#             pz, z=np.max(pz_d), np.argmax(pz_d)
#             topic_words=topics[z][1]
#             out_words=" ".join(["%s:%.4f" % (w, p)
#                                   for (w, p) in topic_words.items()[:num_top_words]])
#             print "%.4f" % pz, ":", out_words.encode("utf8")

#             # find most similar docs for this doc
#             dist=[]
#             for j, line in enumerate(open(did_pt)):
#                 wids=map(int, line.strip().split())
#                 pass


if __name__ == '__main__':
    btm = BTM(model_str="output-all-k1000-fstop")
    print btm.disp_doc(u"太假 了 , 不过 创意 不错")
    print btm.infer_topic([u"太假 了 , 不过 创意 不错"])

    # if len(sys.argv) < 4:
    #     print 'Usage: python %s <model_dir> <K> <voca_pt>' % sys.argv[0]
    #     print '\tmodel_dir    the output dir of BTM'
    #     print '\tK    the number of topics'
    #     print '\tvoca_pt    the vocabulary file'
    #     exit(1)

    # model_dir = sys.argv[1]
    # K = int(sys.argv[2])
    # voca_pt = sys.argv[3]
    # voca = id2word(voca_pt)
    # W = len(voca)
    # print 'K:%d, n(W):%d' % (K, W)

    # z_pt = model_dir + 'k%d.pz' % K
    # zd_pt = model_dir + 'k%d.pz_d' % K
    # wz_pt = model_dir + 'k%d.pw_z' % K

    # pz = [float(p) for p in open(z_pt).readline().split()]

    # print "loading topic-word distribution..."
    # topics = get_topics(wz_pt, pz)
    # if FILTER_PATH:
    #     topics = filter_topics(topics, voca, filter_pt=model_dir + FILTER_PATH)
    # # print "display topics..."
    # # disp_topics(topics, voca, NUM_TOP_WORDS)

    # if len(sys.argv) == 5:
    #     doc_pt = sys.argv[4]
    #     did_pt = doc_pt + ".id"
    #     zd_pt = model_dir + "k%d" % K + SUFFIX
    #     print "infering new doc from %s" % doc_pt
    #     write_doc2id(doc_pt, did_pt, voca_pt, text_mode=0)
    #     infer_topics_from_file(did_pt, model_dir, K)
    #     print "calculating topic ppl..."
    #     print get_perplexity(did_pt, zd_pt, topics)
    #     print "calculating topic coherence..."
    #     print get_topic_choherence(did_pt, topics, voca, num_top_words=10)

    # d_pt = ""
    # if len(sys.argv) == 5:
    #     d_pt = sys.argv[4]
    # disp_docs(did_pt, zd_pt, voca, topics, NUM_SAMPLES_DISPLAY, d_pt=d_pt,
    # num_top_words=NUM_TOP_WORDS)
