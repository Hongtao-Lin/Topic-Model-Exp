#!/usr/bin/env python
# coding=utf-8
"""Building vocabs and preprocess the training file

Attributes:
    debug (str): Description
"""
from __future__ import print_function
import sys, os, json
import config
from btm import BTM

debug = config.debug
btm = BTM()


def init_model():
    btm.load_model(model_str="output-all-k200-fstop", it=800, filter_pt=config.sw_file)
    btm.get_reranked_top_words()
    btm.save_top_words()


def main():
    init_model()
    btm.disp_all_topics(end=2)
    btm.disp_all_topics(end=2, reranked=True)
    btm.disp_top_and_middle_topic()
    btm.disp_top_and_middle_topic(z=2)

    btm.disp_doc(u"我 爱 你 中国")
    btm.disp_doc(u"我 爱 你 中国", infer_type="max_idx")

    topic_metric = ["npmi", "umass"]
    doc_metric = ["purity", "nmi"]
    btm.evaluate_model(topic_metric, doc_metric)
    # print("Test ppl:", btm._get_sent_perplexity(u"除了 李承鹏 的 书 之外 , 都 是 好 书 。"))

    pass

if __name__ == '__main__':
    main()
