# Copyright (c) 2013, Chris De Vries (chris@de-vries.id.au)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    * Neither the name of the Queensland University of Technology nor the
#      names of its contributors may be used to endorse or promote products
#      derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Chris De Vries BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#!/usr/bin/env python

import math
import os.path
import random
import sys
import time

def purity(clust2size, clust2cat):
    results = [] # list of (purity, size) per cluster
    for cluster, categories in clust2cat.iteritems(): 
        category_count = get_category_count(categories)
        majority = max(category_count.items(), key=lambda x: x[1])
        purity = float(majority[1]) / clust2size[cluster] # divide by number of docs, not by number of labels
        results.append((purity, clust2size[cluster]))
    return results

def entropy(clust2size, clust2cat):
    results = [] # list of (entropy, size) per cluster
    for cluster, categories in clust2cat.iteritems():
        category_count = get_category_count(categories)
        entropy = 0
        total_categories = len(categories)
        categories_present = len(category_count)
        if categories_present == 1: # math.log(1,2) == 0.0, causes div by zero when normalizing
            categories_present += 1
        for category, count in category_count.items():
            p_category = float(count) / total_categories
            entropy += p_category * math.log(p_category, 2)
        entropy = -((1/math.log(categories_present,2)) * entropy)
        results.append((entropy, clust2size[cluster]))
    return results

def negentropy(clust2size, clust2cat):
    results = [] # list of (negentropy, size) per cluster
    for entropy_score, cluster_size in entropy(clust2size, clust2cat): 
        results.append((1 - entropy_score, cluster_size))
    return results

def oldf1(clust2cat, categories):
    """
    This is the old f1 measure used at INEX which does not match the description of f1 in the Divergence from Random Baseline papers.
    """
    # convert clustering problem into a classification problem
    contingency = Matrix(None, 'n') # indexed i=predicted category, j=actual category
    for cluster, cluster_categories in clust2cat.iteritems():
        category_count = get_category_count(cluster_categories)
        majority = max(category_count.items(), key=lambda x: x[1])
        predicted_category = majority[0]
        for actual_category in cluster_categories:
            contingency.increment(predicted_category, actual_category)

    # calculate f1
    results = [] # list of (f1, size) per cluster
    for category in categories:
        numerator = float(contingency.get(category, category))

        row_sum = contingency.get_sum_row(category)
        precision = 0.0
        if row_sum > 0:
            precision = numerator / row_sum

        col_sum = contingency.get_sum_col(category)
        recall = 0.0
        if col_sum > 0:
            recall = numerator / col_sum

        f1 = 0.0
        if precision + recall > 0:
            f1 = (2 * precision * recall) / (precision + recall)

        results.append((f1, col_sum))
    return results
    
def f1(doc2clust, doc2cat):
    # Use document IDs from ground truth doc2cat, as it may contain less documents than doc2clust.
    docids = list(doc2cat.keys()) 
    #print('comparing %d pair-wise documents for F1' % len(docids))

    # calculate true positivies, false negative and false positives for all unique pairs of documents where are x,y and y,x are NOT unique 
    start = time.time()
    tp = 0
    fn = 0
    fp = 0
    document_count = len(docids);
    for i in range(document_count):
        #if (i + 1) % 1000 == 0:
        #    print '%d of %d : %s seconds' % (i, document_count, time.time() - start)
        for j in  range(i, document_count):
            if i != j:
                # for multi label submissions treat each label as an example for tp, fn, fp
                # TODO(cdevries): this does not matter for SED2013 as it is single label, but how should this be treated for multi label? Should it be based on set intersections between documents? i.e. tp for all categories that match (set intersection) and then work out fp and fn for the rest 
                for docid_i_cluster in doc2clust[docids[i]]:
                    for docid_j_cluster in doc2clust[docids[j]]:
                        for docid_i_category in doc2cat[docids[i]]:
                            for docid_j_category in doc2cat[docids[j]]:
                                #print('cluster: i=%s j=%s' % (docid_i_cluster, docid_j_cluster))
                                #print('category: i=%s j=%s' % (docid_i_category, docid_j_category))
                                if docid_i_cluster == docid_j_cluster and docid_i_category == docid_j_category:
                                    tp += 1
                                elif docid_i_cluster == docid_j_cluster and docid_i_category != docid_j_category:
                                    fp += 1
                                elif docid_i_cluster != docid_j_cluster and docid_i_category == docid_j_category:
                                    fn += 1
    #print('tp = %d, fn = %d, fp = %d' % (tp, fn, fp))
    numerator = 2 * float(tp)  
    denominator = (2 * float(tp)) + fn + fp                
    if denominator < 1e-15:
        score = 0
    else:
        score = numerator / denominator
    return score

def nmi(clust2cat, categories):
    contingency = Matrix(None, 'n') # indexed by i=cluster, j=category
    for cluster, cluster_categories in clust2cat.iteritems():
        for category in cluster_categories:
            contingency.increment(cluster, category)

    clust_cat_total = contingency.get_sum_all()

    # calculate mutual information
    mi = 0.0
    for cluster, col in contingency.iterrows():
        for category, clust_cat in col.iteritems():
            p_clust = contingency.get_sum_row(cluster) / float(clust_cat_total)
            p_cat = contingency.get_sum_col(category) / float(clust_cat_total)
            p_clust_cat = clust_cat / float(clust_cat_total)
            mi += p_clust_cat * math.log(p_clust_cat / (p_clust * p_cat), 2)

    # calculate entropy
    e_clust = 0.0
    for cluster in clust2cat.iterkeys():
        p_clust = contingency.get_sum_row(cluster) / float(clust_cat_total)
        e_clust += p_clust * math.log(p_clust, 2)
    e_clust = -e_clust

    e_cat = 0.0
    for category in categories:
        p_cat = contingency.get_sum_col(category) / float(clust_cat_total)
        e_cat += p_cat * math.log(p_cat, 2)        
    e_cat = -e_cat

    # calculate NMI
    nmi = mi / ((e_clust + e_cat) / 2)
    return nmi

def micro(l):
    total_count = 0
    for score, cluster_size in l:
        total_count += int(cluster_size)
    total_score = 0.0
    for score, cluster_size in l:
        total_score += score * (float(cluster_size) / total_count)
    return total_score 

def macro(l):
    cluster_count = len(l)
    total_score = 0.0
    for score, cluster_size in l:
        total_score += score
    total_score /= cluster_count
    return total_score

def get_category_count(categories):
    category_count = {}
    for category in categories:
        if category not in category_count:
            category_count[category] = 0
        category_count[category] += 1
    return category_count

# a matrix of ints
class Matrix:
    def __init__(self, file, opt):
        self._db = {} # row -> (col -> val)
        self._sum_all = 0L # accumulator for all rows and cols
        self._sum_row= {} # accumulator for each row
        self._sum_col = {} # accumulator for each col

    def increment(self, i, j):
        self._sum_all += 1

        if i not in self._sum_row:
            self._sum_row[i] = 0
        self._sum_row[i] += 1

        if j not in self._sum_col:
            self._sum_col[j] = 0
        self._sum_col[j] += 1

        if i not in self._db:
            self._db[i] = {}
        row = self._db[i]

        if j not in row:
            row[j] = 0 
        row[j] += 1

    def get(self, i, j):
        if i not in self._db:
            return 0
        row = self._db[i]

        if j not in row:
            return 0
        else:
            return row[j]

    def get_sum_all(self):
      return self._sum_all

    def get_sum_row(self, i):
        if i not in self._sum_row:
            return 0
        else:
            return self._sum_row[i]

    def get_sum_col(self, j):
        if j not in self._sum_col:
            return 0
        else:
            return self._sum_col[j]

    def iterrows(self):
        return self._db.iteritems()

def assign_categories_to_clusters(doc2clust, doc2cat):
    clust2cat = {}
    clust2size = {} 
    
    for docid, doc_categories in doc2cat.iteritems():
        for cluster in doc2clust[docid]:
            if cluster in clust2cat:
                clust_cats = clust2cat[cluster]
            else:
                clust_cats = []
                clust2cat[cluster] = clust_cats 
            
            for category in doc_categories:
                clust_cats.append(category)

            if cluster in clust2size:
                clust2size[cluster] += 1
            else:
                clust2size[cluster] = 1

    return (clust2cat, clust2size)

def load_doc2labels(cluster_file):
    doc2labels = {} 

    for line in cluster_file:
        line = line.strip()
        if line[0] == '#':
            continue
        tokens = line.split()
        docid = tokens[0]
        labels = tokens[1:]
        if docid in doc2labels:
            doc2labels[docid].extend(labels)
        else:
            doc2labels[docid] = labels

    return doc2labels

def unique_values(map): 
    unique = set()

    for key, list in map.iteritems():
        for item in list: 
            unique.add(item)

    return unique

def remove_docs_no_categories(doc2clust, doc2cat):
    """
    returns the number of documents with no categories
    """
    remove = []

    for docid, categories in doc2cat.iteritems():
        if len(categories) == 0:
            remove.append(docid)

    for docid in doc2clust.keys():
        if docid not in doc2cat:
            remove.append(docid)

    for docid in remove:
        if docid in doc2clust:
            del doc2clust[docid]
        if docid in doc2cat:
            del doc2cat[docid]

    return len(remove)

def print_clust_size_distribution(clust2size, log):
    # cluster size distribution 
    sizes = clust2size.values()
    mean_size = float(sum(sizes)) / len(sizes)
    diff_mean_squared = [(float(x) - mean_size)**2 for x in sizes]
    observations = len(sizes) # population distribution (i.e. all clusters) no need to subtract 1 
    stddev = math.sqrt(sum(diff_mean_squared) / float(observations))
    log('cluster sizes:')
    log('\tmean = %f' % mean_size)
    log('\tstandard deviation = %f' % stddev)
    log('\tmax = %d' % max(sizes))
    log('\tmin = %d' % min(sizes))

def load(cluster_file, category_file, log, options):
    log('orginal cluster statistics:')
    doc2clust = load_doc2labels(cluster_file)
    original_document_count = len(doc2clust)
    log('document count = %s' % original_document_count) 

    all_clusters = unique_values(doc2clust)
    original_cluster_count = len(all_clusters)
    log('cluster count = %s' % original_cluster_count)

    doc2cat = load_doc2labels(category_file)
    all_categories = unique_values(doc2cat)
    log('category count = %s' % len(all_categories))

    original_clust2size = {}
    for cluster, docs in invert(doc2clust).items():
        original_clust2size[cluster] = len(docs)
    print_clust_size_distribution(original_clust2size, log)
    log('')

    # check the docid match between categories and clusters
    category_docs = set(doc2cat.keys())
    cluster_docs = set(doc2clust.keys())
    missing = category_docs.difference(cluster_docs) 
    if missing:
        raise Exception('Document IDs do not match between cluster and category file. missing = %s' % missing)
    
    # remove docs with no category information
    no_cat_count = remove_docs_no_categories(doc2clust, doc2cat)
    clust2cat, clust2size = assign_categories_to_clusters(doc2clust, doc2cat) # reassign categories to clusters now that some categories have been removed

    log('reduced cluster statistics:')
    log('documents with no categories = %s' % no_cat_count)
    log('reduced document count = %s' % len(doc2clust))
    log('reduced cluster count = %s' % len(unique_values(doc2clust)))
    print_clust_size_distribution(clust2size, log)
    log('')

    if '-stats' in options:
        log('---------\nCLUSTER STATISTICS\n---------\n')
        clust2docs = invert(doc2clust)
        print_stats(clust2cat, clust2size, clust2docs, doc2cat, log)

    return (doc2clust, doc2cat, clust2cat, clust2size, original_cluster_count, original_document_count, all_categories)

def invert(key2values): # key -> list of values
    value2keys = {}
    for key, values in key2values.items():
        for value in values:
            if value in value2keys:
                value2keys[value].append(key)
            else:
                value2keys[value] = [key]
    return value2keys

def print_stats(clust2cat, clust2size, clust2docs, doc2cat, log):
    # process clusters
    for cluster in clust2cat.keys():
        cat_counts = get_category_count(clust2cat[cluster])
        log('cluster %s (size=%d) contains %d categories' % (cluster, clust2size[cluster], len(cat_counts)))
        log('-- categories in cluster --')
        items = cat_counts.items()
        items.sort(key=lambda x: x[1], reverse=True)
        for category, size in items:
            log('category %s exists %d times (%f %% of documents have this category)' % (category, size, float(size)*100/clust2size[cluster]))
        log('---------------------------------\n')

def generate_baseline(doc2clust, clust2size, category_file, doc2cat, log):
    """
    This only works for single label categories and submissions.
    """
    docids = doc2clust.keys()
    # shuffle document IDs based on hash of category_file so that the same random baseline is always generated with the same inputs
    category_file.seek(0)
    random.seed(category_file.read()) # seed random number generator with hash of category file contents     
    random.shuffle(docids)
    baseline_doc2clust = {}
    offset = 0
    for cluster, size in clust2size.items():
        # choose next cluster from randomly shuffled document list that matches the cluster size
        cluster_docids = docids[offset:offset+size]
        for docid in cluster_docids:
            baseline_doc2clust[docid] = [cluster]
        offset += size
    baseline_clust2cat, baseline_clust2size = assign_categories_to_clusters(baseline_doc2clust, doc2cat)
    log('baseline cluster statistics:')
    print_clust_size_distribution(baseline_clust2size, log)
    log('')
    return baseline_doc2clust, baseline_clust2cat, baseline_clust2size

def are_single_label(doc2list):
    documents_with_more_than_one_label = [x for x in doc2list.items() if len(x[1]) > 1] 
    return len(documents_with_more_than_one_label) == 0 

def run(cluster_file, category_file, log, options):
    """
    returns (list of (metric, score), cluster_count, document_count)
    """

    start = time.time()
    doc2clust, doc2cat, clust2cat, clust2size, cluster_count, document_count, categories = load(cluster_file, category_file, log, options)
    stop = time.time()
    log('load() took %s seconds' % (stop - start))
    log('')

    clusters_are_single_label = are_single_label(doc2clust)
    if clusters_are_single_label:
        log('clusters are single label\n')
    else:
        log('clusters are multi label\n')

    categories_are_single_label = are_single_label(doc2cat)
    if categories_are_single_label:
        log('categories are single label\n')
    else:
        log('categories are multi label\n')

    if '-baseline' in options:
        if not clusters_are_single_label:
            raise Exception('-baseline only supports single label clusters')
        start = time.time()
        baseline_doc2clust, baseline_clust2cat, baseline_clust2size = generate_baseline(doc2clust, clust2size, category_file, doc2cat, log)
        stop = time.time()
        log('generate_baseline() took %s seconds' % (stop - start))
        log('')

    scores = []
    def append_score(score):
        scores.append(score)
        log('%s = %s' % scores[-1]) 

    if '-purity' in options:
        start = time.time()
        score_list = purity(clust2size, clust2cat) 
        append_score(('macro purity', macro(score_list)))
        append_score(('micro purity', micro(score_list)))
        if '-baseline' in options:
            baseline_score_list = purity(baseline_clust2size, baseline_clust2cat) 
            append_score(('random baseline macro purity', macro(baseline_score_list)))
            append_score(('random baseline micro purity', micro(baseline_score_list)))
            append_score(('divergence macro purity', macro(score_list) - macro(baseline_score_list)))
            append_score(('divergence micro purity', micro(score_list) - micro(baseline_score_list)))
        stop = time.time()
        log('')
        log('purity() took %s seconds' % (stop - start))
        log('')

    if '-entropy' in options:
        start = time.time()
        score_list = entropy(clust2size, clust2cat) 
        append_score(('macro entropy', macro(score_list)))
        append_score(('micro entropy', micro(score_list)))
        if '-baseline' in options:
            baseline_score_list = entropy(baseline_clust2size, baseline_clust2cat) 
            append_score(('random baseline macro entropy', macro(baseline_score_list)))
            append_score(('random baseline micro entropy', micro(baseline_score_list)))
            append_score(('divergence macro entropy', macro(baseline_score_list) - macro(score_list)))
            append_score(('divergence micro entropy', micro(baseline_score_list) - micro(score_list)))
        stop = time.time()
        log('')
        log('entropy() took %s seconds' % (stop - start))
        log('')

    if '-negentropy' in options:
        start = time.time()
        score_list = negentropy(clust2size, clust2cat) 
        append_score(('macro negentropy', macro(score_list)))
        append_score(('micro negentropy', micro(score_list)))
        if '-baseline' in options:
            baseline_score_list = negentropy(baseline_clust2size, baseline_clust2cat) 
            append_score(('random baseline macro negentropy', macro(baseline_score_list)))
            append_score(('random baseline micro negentropy', micro(baseline_score_list)))
            append_score(('divergence macro negentropy', macro(score_list) - macro(baseline_score_list)))
            append_score(('divergence micro negentropy', micro(score_list) - micro(baseline_score_list)))
        stop = time.time()
        log('')
        log('negentropy() took %s seconds' % (stop - start))
        log('')

    if '-f1' in options:
        if not clusters_are_single_label:
            raise Exception('-f1 only supports single label clusters')
        if not categories_are_single_label:
            raise Exception('-f1 only supports single label categories')
        start = time.time()
        score = f1(doc2clust, doc2cat)
        append_score(('f1', score))
        if '-baseline' in options:
            baseline_score = f1(baseline_doc2clust, doc2cat)
            append_score(('random baseline f1', baseline_score))
            append_score(('divergence f1', score - baseline_score))
        stop = time.time()
        log('')
        log('f1() took %s seconds' % (stop - start))
        log('')

    if '-oldf1' in options:
        start = time.time()
        score_list = oldf1(clust2cat, categories) 
        append_score(('macro oldf1', macro(score_list)))
        append_score(('micro oldf1', micro(score_list)))
        if '-baseline' in options:
            baseline_score_list = oldf1(baseline_clust2cat, categories) 
            append_score(('random baseline macro oldf1', macro(baseline_score_list)))
            append_score(('random baseline micro oldf1', micro(baseline_score_list)))
            append_score(('divergence macro oldf1', macro(score_list) - macro(baseline_score_list)))
            append_score(('divergence micro oldf1', micro(score_list) - micro(baseline_score_list)))
        stop = time.time()
        log('')
        log('oldf1() took %s seconds' % (stop - start))
        log('')

    if '-nmi' in options:
        start = time.time()
        score = nmi(clust2cat, categories)
        append_score(('nmi', score))
        if '-baseline' in options:
            baseline_score = nmi(baseline_clust2cat, categories)
            append_score(('random baseline nmi', baseline_score))
            append_score(('divergence nmi', score - baseline_score))
        stop = time.time()
        log('')
        log('nmi() took %s seconds' % (stop - start))
        log('')

    return (scores, cluster_count, document_count)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("""ClusterEval 1.0

ABOUT
-----
This program compares a clustering to a ground truth set of categories according to multiple different measures. It also includes a novel approach called 'Divergence from a Random Baseline' that augments existing measures to correct for ineffective clusterings. It has been used in the evaluation of clustering at the INEX XML Mining track at INEX in 2009 and 2010, and the Social Event Detection task at MediaEval in 2013.

A manual describing this program and its use can be found at http://eprints.qut.edu.au/60711/.

For details of the INEX evaluation see http://www.inex.otago.ac.nz/tracks/wiki-mine/wiki-mine.asp.

For details of the MediaEval SED evaluation see http://www.multimediaeval.org/mediaeval2013/sed2013/.

Complete details of the measures can be found in the following paper. Please cite this paper when using this software.
    De Vries, Christopher M., Shlomo Geva, and Andrew Trotman.
    "Document clustering evaluation: Divergence from a random baseline."
    Workshop Information Retrieval, Dortmund, Germany. 2012.
    http://eprints.qut.edu.au/53371/
                 
Written by Chris de Vries (chris@de-vries.id.au) 2009-2013.
Please send any questions, bugs, errors or enhancements to the email address above.
                 
This work has been supported by the Queensland University of Technology Deputy Vice Chancellor's Scholarship.

CREDITS
-------
Many people have helped in the formation and correction of ideas implemented in this software.
Persons who have contributed:
    Philipp Cimiano
    Shlomo Geva
    Sangeetha Kutty
    Richi Nayak
    Timo Reuter
    Andrew Trotman
    Peer reviewers of related published material
    Participants at INEX and MediaEval

USAGE
-----
%s [zero or more OPTIONS] [in: category ground truth] [in: cluster submission]

OPTIONS (default = -nmi):
    -baseline = report measures with Divergence from Random Baseline as well
    -purity = calculate the purity measure
    -entropy = calcualte the entropy measure
    -negentropy = calculate the negentropy measure
    -f1 = calculate the F1 measure using pair-wise document comparisons (WARNING: this is slow as it compares O(n^2) documents)
    -oldf1 = calculate the F1 measure by transforming clustering into a classification using the majority class label. This was used previously at INEX.
    -nmi = calculate the normalised mutual information measure
    -stats = print detailed statistics of categories in clusters"""
        % sys.argv[0])
        sys.exit(1)

    def log(s):
        print(s)

    options_list = sys.argv[1:-2]
    if not options_list:
        options = set(['-nmi'])
    else:
        options = set([x.lower() for x in options_list])

    category_file_path = sys.argv[-2]
    category_file = open(category_file_path, 'r')

    cluster_file_path = sys.argv[-1]
    cluster_file = open(cluster_file_path, 'r')
    try:
        start = time.time()
        scores, cluster_count, document_count = run(cluster_file, category_file, log, options)
        #for name, score in scores:
        #    log('%s = %f' % (name, score))
        stop = time.time()
        log('total time = %s' % (stop - start))
    finally:
        cluster_file.close()
        category_file.close()