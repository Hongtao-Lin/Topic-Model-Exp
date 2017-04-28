#!/usr/bin/env python
# coding=utf-8
# Usage: evaluate topic model: python btm.py [model_name] [iter]
#   if the two arguments are not defined, use the default model instead.
from __future__ import print_function
import sys, os, math, time, random, logging, json, copy, re
import numpy as np
from collections import OrderedDict, namedtuple
import subprocess


NUM_TOP_WORDS = 50
# FILTER_PATH = '../filter_words.txt' # relative to model directory.
# SUFFIX = ".test.pz_d"
# ROOT_DIR = "/lustre/home/acct-csyk/csyk/users/htl11/"
ROOT_DIR = "/slfs1/users/htl11/"
# MODEL_STR = "output-cmnt-k50-fstop"
WORK_DIR = ROOT_DIR + "topic-model/btm/"
# MODEL_STR = "output-cmnt-k40-fstop"
SRC_NAME = "src/btm"
FILTER_WORDS = (u"不 人 好 小 大 会 才 都 再 还 去 点 太 一个 没 真 上 下 做").split()


DOC_PT = "%s/data/stc-data/valid-btm.txt" % ROOT_DIR
DOC_PT2 = "%s/data/10-news-group/test_clf.full.txt" % ROOT_DIR
WORD_PT = "%s/data/zhwiki/count_unigram_bd.txt" % ROOT_DIR
BITERM_PT = "%s/data/zhwiki/count_bigram_bd.txt" % ROOT_DIR
WORD_SW_PT = "%s/data/zhwiki/count_unigram_sw.txt" % ROOT_DIR
BITERM_SW_PT = "%s/data/zhwiki/count_bigram_sw.txt" % ROOT_DIR
FILTER_PT = "%s/res/zh-stopwords.json" % ROOT_DIR

logging.basicConfig(level=logging.DEBUG)

class Biterm(namedtuple("Biterm", "wi wj")):
    __slots__ = ()

    def __new__(_cls, wi, wj):
        _wi, _wj = int(wi), int(wj)
        if wi > wj:
            _wi, _wj = wj, wi
        return tuple.__new__(_cls, (_wi, _wj))

    @property
    def __str__(self):
        return "Biterm: wi=%d, wj=%d" % (self.wi, self.wj)


class BTM(object):
    def __init__(self):
        self.base_dir = ""
        self.model_dir = ""
        self.K, self.V, self.it = None, None, None
        self.w2id, self.id2w = {}, {} 
        self.fwid = []
        self.pz, self.pw_z = [], []
        self.top_words = {}
        self.word_cnt, self.word_sw_cnt = {}, {}
        self.biterm_cnt, self.biterm_sw_cnt = {}, {}

    def load_model(self, model_str="output-all-k50-fstop", it=None):
        self.base_dir = "%s%s/" % (WORK_DIR, model_str)
        self.model_dir = self.base_dir + "model/"
        voca_pt = self.base_dir + "vocab.txt"
        
        self.K = self.base_dir.split("-k")[-1].split("-")[0]
        self.it = it
        if self.K[-1] == "b":
            self.K = int(self.K[:-1])
        else:
            self.K = int(self.K[:])
        self.w2id = word2id(voca_pt)
        self.V = len(self.w2id)
        self.id2w = id2word(voca_pt)
        self.fwid = []

        # load model from file
        logging.debug("Loading Models from %s" % self.model_dir)
        pz, pw_z = [], []
        it_suffix = "" if self.it is None else ".%d" % self.it
        z_pt = self.model_dir + "k%d.pz%s" % (self.K, it_suffix)
        wz_pt = self.model_dir + "k%d.pw_z%s" % (self.K, it_suffix)
        with open(z_pt) as f:
            pz = [float(z) for z in f.readline().strip().split()]
            assert len(pz) == self.K
        with open(wz_pt) as f:
            for line in f:
                pw_z.append([float(p) for p in line.strip().split()])
        self.pz, self.pw_z = np.array(pz), np.array(pw_z)

        self.top_words = self.get_topic_words_from_range(num_words=NUM_TOP_WORDS)

    def get_topic_words_from_range(self, start=0, num_words=20, z=None):
        """sort topic word by their probability in descending order,
        and get topic words in a certain range

        args:
            start (int, optional): start of word rank
            num_words (int, optional): number of words to pick
            z (none, optional): if specified, operate only on this topic

        """
        if start < 0 or num_words < 0 or start + num_words > self.V:
            raise ValueError("topic word range invalid!")
        if z is not None:
            topic_prob = [(i, p) for i, p in enumerate(self.pw_z[z]) if i not in self.fwid]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=True)[
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

    def load_external_stats(self, word_pt, biterm_pt):
        """Load word / biterm probability distribution
           calculated from external corpus
        """
        prob_type = "sw" if "sw" in word_pt else "bd"
        if prob_type == "sw" and self.word_sw_cnt and self.biterm_sw_cnt:
            return self.word_sw_cnt, self.biterm_sw_cnt
        if prob_type == "bd" and self.word_cnt and self.biterm_cnt:
            return self.word_cnt, self.biterm_cnt
        word_cnt = {}
        biterm_cnt = {}
        if prob_type == "sw":
            word_cnt = self.word_sw_cnt
            biterm_cnt = self.biterm_sw_cnt
        else:
            word_cnt = self.word_cnt
            biterm_cnt = self.biterm_cnt

        with open(word_pt) as f:
            for line in f.xreadlines():
                w, cnt = line.decode("utf8").split("\t")
                if w not in self.w2id:
                    continue
                word_cnt[self.w2id[w]] = float(cnt)
        with open(biterm_pt) as f:
            for i, line in enumerate(f.xreadlines()):
                wi, wj, cnt = line.decode("utf8").split("\t")
                if wi not in self.w2id or wj not in self.w2id:
                    continue
                biterm = Biterm(self.w2id[wi], self.w2id[wj])
                biterm_cnt[biterm] = float(cnt)
            logging.debug("Load External Stat Complete") 
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

    def disp_topic(self, z, pz=0.0):
        if pz < 1e-6:
            pz = self.pz[z]
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

    def get_topic_coherence_metric(self, num_top_words=10, cal_type="umass"):
        """get topic coherence as a ref metric, according to cal_type

        Args:
            num_top_words (int, optional): number of top words to consider in each topic
            cal_type (str, optional): select different metric type,
                either: umass, npmi
                see "Exploring the Space of Topic Coherence Measures" for the detailed metric desc
        """
        # step1: get prob from external corpus
        if cal_type in ["umass"]:
            word_pt, biterm_pt = WORD_PT, BITERM_PT
        else:
            word_pt, biterm_pt = WORD_SW_PT, BITERM_SW_PT
        word_prob, biterm_prob = self.load_external_stats(word_pt, biterm_pt)
        # step2: get word subset segmentation
        subsets = []
        for z in range(self.K):
            top_words = [w for w, p in self.top_words[z][:num_top_words]]
            top_words = []
            for w, p in self.top_words[z]:
                if len(top_words) == num_top_words:
                    break
                if w in word_prob:
                    top_words.append(w)
            assert len(top_words) == num_top_words
            subset = []
            if cal_type in ["umass"]:
                for i, wi in enumerate(top_words[1:]):
                    for wj in top_words[:i + 1]:
                        subset.append([wi, wj])
            else:
                for i, wi in enumerate(top_words[:-1]):
                    for wj in top_words[i + 1:]:
                        subset.append([wi, wj])
            subsets.append(subset)
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
                    tmp = math.log((biterm_prob.get(biterm, 0) + eps) /
                                   (word_prob[wi] * word_prob[wj]))
                    topic_coherence += tmp / (-math.log(biterm_prob.get(biterm, 0) + eps))
            topic_coherences.append([z, self.pz[z], topic_coherence])
        # get detailed tc by topic descending order
        sort_topic_coherence = sorted(topic_coherences, key=lambda k: k[1], reverse=True)    
        print(sort_topic_coherence)
        coherence = sum(v[-1] for v in topic_coherences) / self.K
        return coherence

    def _get_perplexity(self, sent, is_raw=False):
        """
        Test Function
        Investigate the detailed procedure of calculating ppl
        """
        total_words, total_prob = 0, 0
        wids = [self.w2id[w] for w in sent.split() if w in self.w2id]
        pz_d = self.quick_infer_topics([wids], is_raw=False)
        fpw_z = copy.deepcopy(self.pw_z)
        for wid in self.fwid:
            fpw_z[:, wid] = 0
        fpw_z /= np.sum(fpw_z, axis=1, keepdims=True)
        for wid in wids:
            if wid >= self.V or wid in self.fwid:
                continue
            print(wid)
            p = sum([p[wid] * pz_d[i] for i, p in enumerate(fpw_z)])
            total_prob += math.log(p)
            total_words += 1
            # print(p, len(wids))
        total_prob /= total_words
        ppl = math.exp(-total_prob)           
        print(ppl)
        return ppl

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

    def quick_infer_topics(self, sent_list, is_raw=False, cal_type="sum_b", infer_type="prob"):
        """Infer topic given sentence list as input, built upon previous function
        
        Args:
            sent_list (str): a list of space-seperated sentences
            is_raw (bool, optional): whether the sentence is transformed into word idxs
            cal_type (str, optional): either sum_b, sum_w or mix
            infer_type (str, optional): either prob or max_idx
                if max_idx, only the max topic idx is returned
        """
        assert infer_type in ["max_idx", "prob"]

        wids = []
        for sent in sent_list:
            if is_raw:
                wids.append([self.w2id[w] for w in sent.strip().split()
                             if w in self.w2id])
            else:
                wids.append([int(w) for w in sent.split() if int(w) < self.V])

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

    def get_doc_coherence(self, prob_list, label_list, cal_type=["nmi", "purity"]):
        """Get document clustering measurement from other open-source project

        Args:
            cal_type (str, optional): either nmi or purity
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
                # get detailed score by topic descending order
                i = 0
                while not result.split("\n")[i].startswith("detailed purity stat"):
                    i += 1
                for k in range(i + 1, i + 1 + self.K):
                    if len(result.split("\n")[k].split()) != 3:
                        break
                    z, score, cluster_size = result.split("\n")[k].split()
                    print(z, self.pz[int(z)], score, cluster_size)
            else:
                scores[cal] = float(re.search(r"%s = (.*)\n" % cal, result).group(1))
        return scores

    def disp_doc(self, sent):
        """Display sentence prediction in topic model:
            Sentence after stopwords are filtered
            Top-2 topics and its top-N words
        """
        print("Display Topics for Doc: %s" % sent.encode("utf8"))
        _sent = " ".join([w for w in sent.split() if w in self.w2id])
        print("Fit doc: %s" % _sent.encode("utf8"))
        pz_d = [(i, p) for i, p in enumerate(self.quick_infer_topics([_sent], is_raw=True)[0])]
        pz_d = sorted(pz_d, key=lambda t: t[1], reverse=True)
        print("Top-2 Topics:")
        for z, pz in pz_d[:2]:
            out_words = " ".join(["%s:%.4f" % (self.id2w[w], p)
                                  for (w, p) in self.top_words[z]])
            print("%.4f" % pz, ":", out_words.encode("utf8"))
            # print("\n")
    
    def evaluate_model(self, topic_metric, doc_metric):
        """Get automated metrics for topic model:
            Topic-related: npmi, umass
            Doc-related: purity, nmi
        
        Args:
            topic_metric (str): a list of metric defined within npmi, umass
            doc_metric (str): a list of metric defined within nmi, purity
        """
        print("Perplexity:", self.get_perplexity(DOC_PT, is_raw=True))
        print("Topic Coherence")
        for metric in topic_metric:
            print("%s" % metric, self.get_topic_coherence_metric(num_top_words=10, cal_type=metric))
        print("")

        print("Doc Coherence:")
        sent_list = []
        label_list = []
        with open(DOC_PT2) as f:
            for line in f.readlines():
                sent, label = line.decode("utf8").strip().split("\t")
                sent_list.append(sent)
                label_list.append(int(label))
        prob_list = self.quick_infer_topics(sent_list, is_raw=True, infer_type="prob")
        scores = self.get_doc_coherence(prob_list, label_list, cal_type=doc_metric)
        for k, v in scores.items():
            print(k, v)

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
    model_str = "output-all-k50-fstop"
    iteration = 400
    topic_metric = ["npmi", "umass"]
    doc_metric = ["purity", "nmi"]

    if len(sys.argv) >= 3:
        model_str = sys.argv[1]
        iteration = int(sys.argv[2])

    btm = BTM()
    print("Evaluating model: %s %s" % (model_str, iteration))
    btm.load_model(model_str=model_str, it=iteration)
    btm.filter_words(FILTER_PT)
    # btm.disp_doc(u"今天 上海 的 天气 不错")

    btm.evaluate_model(topic_metric, doc_metric)
    # btm.disp_topic(z=44)
    # btm.disp_topic(z=32)
    # btm.disp_topic(z=0)
    # print("Test ppl:", btm._get_perplexity(u"除了 李承鹏 的 书 之外 , 都 是 好 书 。"))

    # Manually look at topic models:
    # print("Human Evaluation I:")
    # topic_idxs = np.argsort(-btm.pz)[get_normal_samples(btm.K)]
    # for k in topic_idxs:
    #     btm.disp_top_and_middle_topic(k)
    # print("Doc Coherence:")
    # sent_list = []
    # label_list = []
    # with open(DOC_PT2) as f:
    #     for line in f.readlines():
    #         sent, label = line.decode("utf8").strip().split("\t")
    #         sent_list.append(sent)
    #         label_list.append(int(label))
    # prob_list = btm.quick_infer_topics(sent_list, is_raw=True, infer_type="prob")
    # scores = btm.get_doc_coherence(prob_list, label_list)
    # for k, v in scores.items():
    #     print(k, v)

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
