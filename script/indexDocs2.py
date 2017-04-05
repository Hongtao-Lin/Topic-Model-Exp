#!/usr/bin/env python
# coding=utf-8
# translate word into id in documents
import sys
import os
import json
import codecs

res_dir = '/home/slhome/htl11/res/'
stopword_f = res_dir + 'zh-stopwords.json'

w2id = {}
w2cnt = {}
freq_thes = 5
most_freq_thes = 100
text_mode = 0
# mode 0: post + comment, 1: post, 2: comment
debug = False


def load_vocab(pt):
    for line in codecs.open(pt, encoding='utf8'):
        if not line.split():
            continue
        wid, w = line.split()
        w2id[w] = wid


def build_vocab(pt, filter_pt=""):
    print "index file: %s" % pt
    stopwords = json.load(codecs.open(stopword_f, encoding='utf8'))
    stopword_cnt = {}

    for s in stopwords:
        stopword_cnt[s] = 0
    for i, l in enumerate(codecs.open(doc_pt, encoding='utf8')):
        if debug and i > 1000:
            break
        # especially for stc-data format
        if text_mode == 0:
            ws = l.replace("=>", "").strip().split()
        elif text_mode == 1:
            ws = l.split("=>")[0].strip().split()
        else:
            ws = " ".join(l.split("=>")[1:]).strip().split()
        for w in ws:
            if w in stopwords:
                stopword_cnt[w] += 1
            if w not in w2id:
                w2id[w] = 0
            if w not in w2cnt:
                w2cnt[w] = 0
            w2cnt[w] += 1

    total_cnt = sum(w2cnt.values())
    print "# words:", total_cnt

    filtered_words = sum(cnt <= freq_thes for cnt in w2cnt.values())
    filtered_cnt = sum(cnt for cnt in w2cnt.values() if cnt <=
                       freq_thes)  # + sum(stopword_cnt.values())
    # freq_thes_up = sorted(w2cnt.values(), reverse=True)[most_freq_thes]
    # if filter_pt:
    # f_pt = codecs.open(filter_pt, "w", encoding="utf8")
    for (w, cnt) in w2cnt.items():
        if cnt <= freq_thes:
            del w2id[w]
        elif w in stopword_cnt:
            del w2id[w]
        # elif cnt >= freq_thes_up:
        #     print >>f_pt, w
        #     del w2id[w]
        #     filtered_cnt += cnt

    for i, w in enumerate(w2id.keys()):
        w2id[w] = i

    coverage = 1 - filtered_cnt / float(total_cnt)
    print "vocab size:", len(w2id.keys())
    print "words with freq <=", freq_thes, ":", filtered_words
    print "coverage after filter:", coverage


def filter_doc(doc_pt):
    """Filter out doc if its length < 3 within vocab."""
    new_doc = doc_pt + ".filter.%d" % text_mode
    f_pt = codecs.open(new_doc, "w", encoding="utf8")
    for l in codecs.open(doc_pt, encoding='utf8'):
        if text_mode == 0:
            sent_list = l.split("=>")
        elif text_mode == 1:
            sent_list = l.split("=>")[:1]
        else:
            sent_list = l.split("=>")[1:]
        new_sent_list = []

        for i, sent in enumerate(sent_list):
            new_sent = ' '.join([w for w in sent.strip().split() if w in w2id])
            # if len(sent.split()) > 3:
            new_sent_list.append(new_sent)
            print >>f_pt, new_sent + " vs " + sent
        # print >>f_pt, "=>".join(new_sent_list)


def write_w2id(doc_pt, dwid_pt, voca_pt):
    if not os.path.isfile(voca_pt):
        print 'write:', voca_pt
        wf = open(voca_pt, 'w')
        for w, wid in sorted(w2id.items(), key=lambda d: d[1]):
            print >>wf, '%d\t%s' % (wid, w.encode("utf8"))

    print 'write:', dwid_pt
    filter_cnt = 0
    total_doc = 0
    wf = open(dwid_pt, 'w')
    for i, l in enumerate(codecs.open(doc_pt, encoding='utf8')):
        if debug and i > 1000:
            break
        if text_mode == 0:
            sent_list = l.split("=>")
        elif text_mode == 1:
            sent_list = l.split("=>")[:1]
        else:
            sent_list = l.split("=>")[1:]
        # print sent_list[0]
        total_doc += len(sent_list)
        for sent in sent_list:
            ws = sent.strip().split()
            wids = [w2id[w] for w in ws if w in w2id]
            if len(wids) >= 3:
                print >>wf, ' '.join(map(str, wids))
            else:
                filter_cnt += 1
    print 'total num of doc:', total_doc
    print 'filter cnt:', filter_cnt


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print 'Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % sys.argv[0]
        print '\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."'
        print '\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId ..."'
        print '\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"'
        exit(1)

    doc_pt = sys.argv[1]
    dwid_pt = sys.argv[2]
    voca_pt = sys.argv[3]
    # filter_pt = sys.argv[4]
    if os.path.isfile(voca_pt):
        load_vocab(voca_pt)
    else:
        build_vocab(doc_pt, filter_pt="")
    # filter_doc(doc_pt)
    print 'n(w)=', len(w2id)
    write_w2id(doc_pt, dwid_pt, voca_pt)
