# coding=utf-8
from btm import *

MODEL_STR = "output-all-k500-fstop"
ITER = 500

def get_labels(in_pt):
	btm = BTM(MODEL_STR, ITER)
	sent_list = []
	with open(in_pt) as f:
		for line in f:
			labels, post, _, cmnt = line.decode("utf8").split("\t")
			sent_list.append(cmnt)
	topics = btm.quick_infer_topics(sent_list, infer_type="max_idx", is_raw=True)
	tmp_pt = in_pt + ".tmp"
	i = 0
	assert len(sent_list) == topics.shape[0]
	with open(tmp_pt, "w") as f, open(in_pt) as ori_f:
		for line in ori_f:
			labels, post, sym, cmnt = line.decode("utf8").split("\t")
			labels = labels.split()
			labels[-1] = int(topics[i])
			i += 1
			output = "\t".join([" ".join(labels), post, sym, cmnt])
			f.write(output.encode("utf8") + "\n")
	
def main():
	get_labels(WORK_DIR + "output-all-k1000-fstop/model/train.txt.1")


main()
