#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function
import sys, os, json, re


__all__ = [
    "root_dir", "data_dir", "res_dir", "btm_dir",
    "sw_file",
    "debug",
]

root_dir = "/slfs1/users/htl11/"
# root_dir = "/lustre/home/acct-csyk/csyk/users/htl11/"
# root_dir = "/speechlab/users/htl11/"
btm_dir = root_dir + "topic-model/btm/"

data_dir = root_dir + "data/"
res_dir = root_dir + "res/"
sw_file = res_dir + 'zh-stopwords.json'

debug = True
