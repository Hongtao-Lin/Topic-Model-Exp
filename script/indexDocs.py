#!/usr/bin/env python
"""Building vocabs and preprocess the training file

Attributes:
    debug (str): Description
"""
# coding=utf-8
# from __future__ import print_function
import sys, os, json
import config

debug = config.debug


def load_vocab(pt):
    w2id = {}
    for line in open(pt):
        if not line.split():
            continue
        wid, w = line.decode("utf8").split()
        w2id[w] = wid
    return w2id

def save_vocab(w2id, voca_pt):
    print 'write:', voca_pt
    wf = open(voca_pt, 'w')
    for w, wid in sorted(w2id.items(), key=lambda d: d[1]):
        print >>wf, '%d\t%s' % (wid, w.encode("utf8"))

def read_line(l, text_mode = 0):
    """Read sentences from a single line, the format below is especially for stc data like:
    post => cmnt1 => cmnt2 ...
    
    Args:
        text_mode (int, optional): range from 0, 1, 2
            if 0, post and cmnt are all selected
            if 1, only post, if 2, only cmnts
    """
    if text_mode == 0:
        return l.split("=>")
    elif text_mode == 1:
        return l.split("=>")[:1]
    else:
        return l.split("=>")[1:]

def build_vocab(pt, text_mode = 0, filter_stop=False, filter_pt=""):
    """Build vocab from data source
    
    Args:
        text_mode (int, optional): range from 0, 1, 2
            if 0, post and cmnt are all selected
            if 1, only post, if 2, only cmnts
        filter_stop (bool, optional): whether to filter stopwords
        filter_pt (str, optional): filter additional words besides stopwords
    """
    print "index file: %s" % pt
    freq_thes = 5
    w2cnt = {}
    w2id = {}
    stopwords = json.loads(open(config.sw_file).read().decode("utf8"))
    stopword_cnt = {}

    for s in stopwords:
        stopword_cnt[s] = 0
    for i, l in enumerate(open(doc_pt)):
        if debug and i > 1000:
            break
        sent_list = read_line(l.decode("utf8"), text_mode)
        ws = (" ".join(sent_list)).strip().split()
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
    filtered_cnt = sum(cnt for cnt in w2cnt.values() if cnt <= freq_thes)  # + sum(stopword_cnt.values())
    # freq_thes_up = sorted(w2cnt.values(), reverse=True)[most_freq_thes]
    # if filter_pt:
    # f_pt = codecs.open(filter_pt, "w", encoding="utf8")
    for (w, cnt) in w2cnt.items():
        if cnt <= freq_thes:
            del w2id[w]
        elif filter_stop and w in stopword_cnt:
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
    return w2id

def filter_doc(doc_pt, w2id, text_mode):
    """Filter out doc if its length < 3 within vocab."""
    new_doc = doc_pt + ".filter.%d" % text_mode
    f_pt = open(new_doc, "w")
    for l in open(doc_pt):
        sent_list = read_line(l.decode("utf8"), text_mode)
        new_sent_list = []

        for i, sent in enumerate(sent_list):
            new_sent = ' '.join([w for w in sent.strip().split() if w in w2id])
            # if len(sent.split()) > 3:
            new_sent_list.append(new_sent)
            print >>f_pt, (new_sent + " vs " + sent).encode("utf8")
        # print >>f_pt, "=>".join(new_sent_list)


def write_doc2id(doc_pt, dwid_pt, voca_pt, text_mode = 0)
    if os.path.isfile(dwid_pt):
        return

    print 'write:', dwid_pt
    filter_cnt = 0
    total_doc = 0
    w2id = load_vocab(voca_pt)
    wf = open(dwid_pt, 'w')
    for i, l in enumerate(open(doc_pt)):
        if debug and i > 1000:
            break
        sent_list = read_line(l.decode("utf8"), text_mode)
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
    if len(sys.argv) < 5:
        print 'Usage: python %s <doc_pt> <dwid_pt> <voca_pt>' % sys.argv[0]
        print '\tdoc_pt    input docs to be indexed, each line is a doc with the format "word word ..."'
        print '\tdwid_pt   output docs after indexing, each line is a doc with the format "wordId wordId ..."'
        print '\tvoca_pt   output vocabulary file, each line is a word with the format "wordId    word"'
        print '\tfstop   bool, whether to filter out stopwords'
        exit(1)

    text_mode = 0
    doc_pt = sys.argv[1]
    dwid_pt = sys.argv[2]
    voca_pt = sys.argv[3]
    filter_stop = bool(int(sys.argv[4]))
    # filter_pt = sys.argv[4]
    if os.path.isfile(voca_pt):
        w2id = load_vocab(voca_pt)
    else:
        w2id = build_vocab(doc_pt, text_mode, filter_stop=filter_stop, filter_pt="")
        save_vocab(w2id, voca_pt)
    print 'n(w)=', len(w2id)
    # filter_doc(doc_pt, w2id, text_mode)
    write_doc2id(doc_pt, dwid_pt, voca_pt, text_mode)
