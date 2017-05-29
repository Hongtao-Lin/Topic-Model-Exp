#!/usr/bin/env python
# coding=utf-8
# Usage: evaluate topic model: python btm.py [model_name] [iter]
#   if the two arguments are not defined, use the default model instead.
from __future__ import print_function
import sys, os, math, time, random, logging, json, copy, re, string
from collections import OrderedDict, namedtuple
import numpy as np
import subprocess
import config

# FILTER_PATH = '../filter_words.txt' # relative to model directory.
# SUFFIX = ".test.pz_d"
# MODEL_STR = "output-cmnt-k50-fstop"
# MODEL_STR = "output-cmnt-k40-fstop"
SRC_NAME = "src/btm"

debug = config.debug
if debug:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)


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

def is_valid_pos(word):
    return True

    from pyltp import Postagger
    postagger = Postagger()
    postagger.load("/slfs1/users/xl526/stc-data/ltp_data/pos.model")
    pos = postagger.postag([word.encode("utf8")])
    postag = list(pos)[0]
    # print(postag)
    return postag in "a b i j n nh ni nl ns nz v".split()


def is_valid_keyword(w):
    filter_char = string.letters + string.punctuation + string.digits
    return (w[0] not in filter_char and is_valid_pos(w))


class BTM(object):

    def __init__(self):
        self.base_dir = ""
        self.model_dir = ""
        self.K, self.V, self.it = None, None, None
        self.w2id, self.id2w = {}, {}
        self.fwid = []
        self.pz, self.pw_z = [], []
        self.top_words, self.reranked_top_words = [], []
        self.word_cnt, self.word_sw_cnt = {}, {}
        self.biterm_cnt, self.biterm_sw_cnt = {}, {}
        self.n_top_words = 200

    def load_model(self, model_str="output-all-k50-fstop", it=None, filter_pt=""):
        self.base_dir = "%s%s/" % (config.btm_dir, model_str)
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
        logging.debug("Loading models from %s" % self.model_dir)
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
        if filter_pt:
            self.filter_words(filter_pt)
        self.top_words = self.get_topic_words_from_range()

    def filter_words(self, filter_pt):
        """Filter topic-words according to filter vocab, i.e, stopwords,
        Also filter out manually defined words

        Args:
            filter_pt (str): file path for filtered words, a json file
        """
        logging.debug("Filter vocab...")
        filter_words = (u"不 人 好 小 大 会 才 都 再 还 去 点 太 一个 没 真 上 下 做 想 说 觉得 看 不要").split()
        with open(filter_pt) as f:
            filter_words += json.loads(f.read().decode("utf8"))
        filter_words = [w for w in filter_words if w in self.w2id]
        self.fwid = [self.w2id[w] for w in filter_words]
        logging.info("n(w) = %d" % (self.pw_z.shape[1] - len(self.fwid)))

    # topic word related functions
    def get_topic_by_rank(self, rank=None):
        sort_topic = np.argsort(-self.pz)
        if rerank is not None:
            return sort_topic[rerank]
        return sort_topic

    def get_reranked_top_words(self):
        logging.debug("Get reranked top words...")
        pwz = self.pw_z * self.pz.reshape(-1, 1)
        pz_w = (pwz / np.sum(pwz, axis=0)).T
        entropy = np.exp(-np.sum((pz_w * np.log(pz_w)), axis=1))

        topic_probs = self.pw_z / entropy
        logging.debug(topic_probs.shape)

        top_words = []
        for pw_z in topic_probs:
            # print(pw_z.shape)
            topic_prob = [(i, p) for i, p in enumerate(pw_z)
                          if i not in self.fwid and is_valid_keyword(self.id2w[i])]
            topic_prob = sorted(topic_prob, key=lambda t: t[1], reverse=True)[:self.n_top_words]
            top_words.append(topic_prob)
        self.reranked_top_words = np.array(top_words)

    def get_topic_words_from_range(self, start=0, num_words=None, z=None):
        """sort topic word by their probability in descending order,
        and get topic words in a certain range

        args:
            start (int, optional): start of word rank
            num_words (int, optional): number of words to pick
            z (none, optional): if specified, operate only on this topic

        """
        if num_words is None:
            num_words = self.n_top_words

        if start < 0 or num_words < 0 or start + num_words > self.V:
            raise ValueError("In top words func: topic word range invalid!")
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

    def disp_topic(self, z, pz=0.0, reranked=False, wids=None):
        if pz < 1e-6:
            pz = self.pz[z]
        if reranked:
            wids = self.reranked_top_words[z][:, 0]
        elif wids is None:
            wids = self.top_words[z][:, 0]

        output = "%.3f" % pz
        output += " : " + " ".join(["%s" % self.id2w[w] for w in wids])
        print(output.encode("utf8") + "\n")

    def disp_all_topics(self, reranked=False, start = 0, end = 10):
        print("Display topics of range %d - %d in descending order..." % (start, end))
        print("p(z)\t\tTop Words")
        for z, pz in zip(np.argsort(self.pz), np.sort(self.pz))[::-1][start:end]:
            self.disp_topic(z, pz, reranked=reranked)

    def disp_top_and_middle_topic(self, z=None, base_k=1000):
        """Display topic by its top words and middle-ranked words

        Args:
            z (None, optional): if specified, display only this topic
            base_k (int, optional): starting rank for middle-ranked words
        """
        if z is None:
            z = random.randint(0, self.K - 1)
        print("Display top and middle words of topic #%d" % z)
        pz = self.pz[z]
        self.disp_topic(z, pz)
        mid_topic_words = self.get_topic_words_from_range(start=base_k)
        self.disp_topic(z, wids=mid_topic_words[z][:, 0])

    def save_top_words(self):
        logging.debug("Save top words...")

        o = open(self.model_dir + "k%d.top_words" % self.K, "w")
        for words in self.top_words:
            output = " ".join([self.id2w[w[0]] for w in words])
            o.write(output.encode("utf8") + "\n")
        o.close()

        # o = open(self.model_dir + "k200.pz_w", "w")
        # for p in pz_w:
        #     o.write(" ".join([str(pi) for pi in p]) + "\n")
        # o.close()

        if not len(self.reranked_top_words):
            return

        o = open(self.model_dir + "k%d.reranked_top_words" % self.K, "w")
        for words in self.reranked_top_words:
            output = " ".join([self.id2w[w[0]] for w in words])
            o.write(output.encode("utf8") + "\n")
        o.close()

    # doc inference functions
    def _doc2id(self, doc_pt, did_pt=""):
        if not did_pt:
            did_pt = doc_pt + ".tmp"
        out_f = open(did_pt, "w")
        for line in open(doc_pt):
            line = " ".join([str(self.w2id[w]) for w in line.decode("utf8").strip().split()
                             if w in self.w2id])
            out_f.write(line.encode("utf8") + "\n")
        out_f.close()
        return did_pt

    def quick_infer_topics_from_file(self, doc_pt, out_file="", is_raw=False,
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
            did_pt = self.model_dir + "%s.tmp" % filename
            did_pt = self._doc2id(doc_pt, did_pt)
        else:
            did_pt = doc_pt
        suffix = ".%s.%s" % (filename, suffix)
        zd_pt = self.model_dir + "k%d%s" % (self.K, suffix)
        cmd = ["%s%s" % (config.btm_dir, SRC_NAME), "inf", "sum_b",
               str(self.K), did_pt, self.model_dir, suffix, infer_type, str(self.it)]
        logging.debug("Running Command: " + " ".join(cmd))
        t1 = time.time()
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        # returncode = process.wait()
        logging.debug("Time Spend: %.3f" % (time.time() - t1))
        # print(returncode)
        logging.debug(process.stdout.read())
        if out_file != "":
            os.system("cp %s %s" % (zd_pt, out_file))
        return did_pt, zd_pt

    def quick_infer_topics(self, sent_list, is_raw=False, cal_type="sum_b", infer_type="prob"):
        """Infer topic given sentence list as input, built upon previous function

        Args:
            sent_list (list): a list of space-seperated sentences
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

        doc_pt = self.model_dir + "doc.tmp"
        with open(doc_pt, "w") as f:
            for wid in wids:
                output = " ".join([str(w) for w in wid])
                f.write(output + "\n")

        did_pt, zd_pt = self.quick_infer_topics_from_file(
            doc_pt, is_raw=False, cal_type=cal_type, infer_type=infer_type)

        res_list = []
        with open(zd_pt) as f:
            for line in f.xreadlines():
                if infer_type == "prob":
                    res = [float(p) for p in line.strip().split()]
                elif infer_type == "max_idx":
                    res = int(line.strip())
                res_list.append(res)

        return np.array(res_list)

    def disp_doc(self, sent):
        """Display sentence prediction in topic model:
            Sentence after stopwords are filtered
            Top-2 topics and its top-N words
        """
        logging.info("Display topics for doc: %s" % sent.encode("utf8"))
        _sent = " ".join([w for w in sent.split() if w in self.w2id])
        logging.info("Fit doc: %s" % _sent.encode("utf8"))
        pz_d = [(i, p) for i, p in enumerate(self.quick_infer_topics([_sent], is_raw=True)[0])]
        pz_d = sorted(pz_d, key=lambda t: t[1], reverse=True)
        print("Top-2 Topics:")
        for z, pz in pz_d[:2]:
            self.disp_topic(z)

    # evaluation related functions
    def get_topic_coherence_metric(self, n_top_words=10, cal_type="umass"):
        """get topic coherence as a ref metric, according to cal_type

        Args:
            n_top_words (int, optional): number of top words to consider in each topic
            cal_type (str, optional): select different metric type,
                either: umass, npmi
                see "Exploring the Space of Topic Coherence Measures" for the detailed metric desc
        """
        def load_external_stats(word_pt, biterm_pt):
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
                for i, line in enumerate(f.xreadlines()):
                    w, cnt = line.decode("utf8").split("\t")
                    if w not in self.w2id:
                        continue
                    # if debug and i > 1000:
                    #     break
                    word_cnt[self.w2id[w]] = float(cnt)

            with open(biterm_pt) as f:
                for i, line in enumerate(f.xreadlines()):
                    wi, wj, cnt = line.decode("utf8").split("\t")
                    if wi not in self.w2id or wj not in self.w2id:
                        continue
                    if debug and i > 1000:
                        break
                    biterm = Biterm(self.w2id[wi], self.w2id[wj])
                    biterm_cnt[biterm] = float(cnt)
                logging.debug("Load external stat complete")

            return word_cnt, biterm_cnt

        word_pt = "%szhwiki/count_unigram_bd.txt" % config.data_dir
        biterm_pt = "%szhwiki/count_bigram_bd.txt" % config.data_dir
        word_sw_pt = "%szhwiki/count_unigram_sw.txt" % config.data_dir
        biterm_sw_pt = "%szhwiki/count_bigram_sw.txt" % config.data_dir

        # step1: get prob from external corpus
        if cal_type in ["umass"]:
            word_pt, biterm_pt = word_pt, biterm_pt
        else:
            word_pt, biterm_pt = word_sw_pt, biterm_sw_pt
        word_prob, biterm_prob = load_external_stats(word_pt, biterm_pt)

        # step2: get word subset segmentation
        subsets = []
        for z in range(self.K):
            top_words = [w for w, p in self.top_words[z][:n_top_words]]
            top_words = []
            for w, p in self.top_words[z]:
                if len(top_words) == n_top_words:
                    break
                if w in word_prob:
                    top_words.append(w)
            # print(len(self.top_words[z]))
            assert len(top_words) == n_top_words
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
        
        # step3: calculate topic coherence:
        topic_coherences = []
        eps = 1e-6
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
        # logging.debug(sort_topic_coherence)
        coherence = sum(v[-1] for v in topic_coherences) / self.K
        return coherence

    def _get_sent_perplexity(self, sent, is_raw=False):
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

    def get_doc_coherence(self, prob_list, label_list, cal_type="nmi"):
        """Get document clustering measurement from other open-source project

        Args:
            cal_type (str, optional): either nmi or purity
        """
        cluster_pt = "%scluster.tmp" % (self.base_dir)
        label_pt = "%slabel.tmp" % (self.base_dir)
        with open(cluster_pt, "w") as f:
            cluster_idx = np.argmax(prob_list, axis=1)
            for i, idx in enumerate(cluster_idx):
                f.write("%d\t%d\n" % (i, idx))
        with open(label_pt, "w") as f:
            for i, idx in enumerate(label_list):
                f.write("%d\t%d\n" % (i, idx))
        cmd = ["python", "cluster_eval.py"] + ["-%s" % cal_type] + \
              [label_pt, cluster_pt]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        result = process.stdout.read()
        # logging.debug(result)
        scores = {}
        if cal_type == "purity":
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
                output = "%s %.4f %s %s" % (z, self.pz[int(z)], score, cluster_size)
                # logging.debug(output)
        else:
            scores[cal_type] = float(re.search(r"%s = (.*)\n" % cal_type, result).group(1))
        return scores

    def evaluate_model(self, topic_metric, doc_metric):
        """Get automated metrics for topic model:
            Topic-related: npmi, umass
            Doc-related: purity, nmi

        Args:
            topic_metric (str): a list of metric defined within npmi, umass
            doc_metric (str): a list of metric defined within nmi, purity
        """
        ppl_pt = "%sstc-data/test_small.txt" % config.data_dir
        cls_pt = "%s10-news-group/test_clf.full.txt" % config.data_dir
        print("Perplexity:", self.get_perplexity(ppl_pt, is_raw=True))
        print("Topic Coherence")
        for metric in topic_metric:
            print("%s" % metric, self.get_topic_coherence_metric(n_top_words=10, cal_type=metric))
        print("")

        print("Doc Coherence:")
        sent_list = []
        label_list = []
        with open(cls_pt) as f:
            for line in f.readlines():
                sent, label = line.decode("utf8").strip().split("\t")
                sent_list.append(sent)
                label_list.append(int(label))
        prob_list = self.quick_infer_topics(sent_list, is_raw=True, infer_type="prob")
        scores = {}
        for metric in doc_metric:
            scores.update(self.get_doc_coherence(prob_list, label_list, cal_type=metric))
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

def get_even_samples(k):
    idxs = []
    for i in range(4):
        sidx = int(math.floor(k * i / 4))
        idxs += range(sidx, sidx + 5)
    return idxs

if __name__ == '__main__':
    model_str = "output-all-k200-fstop"
    iteration = 800
    topic_metric = ["npmi", "umass"]
    doc_metric = ["purity", "nmi"]

    if len(sys.argv) >= 3:
        model_str = sys.argv[1]
        iteration = int(sys.argv[2])

    btm = BTM()
    print("Evaluating model: %s %s" % (model_str, iteration))
    btm.load_model(model_str=model_str, it=iteration, filter_pt=config.sw_file)
    # btm.filter_words(config.sw_file)
    # btm.disp_doc(u"哈 哈")

    # btm.evaluate_model(topic_metric, doc_metric)
    btm.disp_topic(z=93)
    btm.get_reranked_top_words()
    btm.disp_topic(z=93)

    # print("Test ppl:", btm._get_sent_perplexity(u"除了 李承鹏 的 书 之外 , 都 是 好 书 。"))

    # btm.quick_infer_topics_from_file(config.root_dir + "data/stc-data/train.txt", is_raw=True)
    pass
