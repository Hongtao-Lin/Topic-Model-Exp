# coding=utf-8
from __future__ import print_function
from btm import *
import logging

logging.basicConfig(level=logging.DEBUG)

MODEL_STR = "output-all-k500-fstop"
ITER = 500

def modify(in_pt, out_pt):
	with open(in_pt) as ori_f, open(out_pt, "w") as f:
		for line in ori_f:
			if line.strip() == "":
				continue
			f.write(line)

def get_labels(in_pt):
	btm = BTM(MODEL_STR, ITER)
	sent_list = []
	with open(in_pt) as f:
		for line in f:
			labels, post, _, cmnt = line.decode("utf8").split("\t")
			sent_list.append(cmnt)
	topics = btm.quick_infer_topics(sent_list, infer_type="max_idx", is_raw=True)
	logging.debug(sent_list[0], topics[0])
	tmp_pt = in_pt + ".tmp"
	i = 0
	assert len(sent_list) == topics.shape[0]
	logging.debug("start labeling...")
	with open(tmp_pt, "w") as f, open(in_pt) as ori_f:
		for line in ori_f:
			labels, post, sym, cmnt = line.decode("utf8").strip().split("\t")
			labels = labels.split()
			labels[-1] = topics[i]
			labels = [str(label) for label in labels]
			i += 1
			output = "\t".join([" ".join(labels), post, sym, cmnt])
			f.write(output.encode("utf8") + "\n")
	
def main():
	in_pt = WORK_DIR + "output-all-k1000-fstop/model/valid.txt.1"
	out_pt = WORK_DIR + "output-all-k1000-fstop/model/train.txt.1.tmp.tmp"
	get_labels(in_pt)
	# modify(in_pt, out_pt)
main()
