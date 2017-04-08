#include <cassert>
#include <iostream>
#include <string>
#include <cmath>
#include <limits>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <sched.h>

#include "sampler.h"
#include "str_util.h"
#include "model.h"

void Model::run(string doc_pt, string res_dir, int P) {
  load_docs(doc_pt);

  if (model_exist(res_dir)) {
  	load_model(res_dir);
  } else {
    model_init();
  }
  // model_init();
  int N = bs.size();

  cout << "Begin iteration" << endl;
  string out_dir = res_dir + "k" + str_util::itos(K) + ".";


  for (int it = 1; it < n_iter + 1; ++it) {
    cout << "iter " << it << '/' << n_iter;
    fflush(stdout);
    double t1 = omp_get_wtime();
    #pragma omp parallel for num_threads(P)
    for (int b = 0; b < bs.size(); ++b) {
      update_biterm(bs[b]);
    }
    recompute_count();
    // save_res(out_dir);
    if (it % save_step == 0) {
      save_res(out_dir);
    }
    double t2 = omp_get_wtime();
    printf("Time per iter: %f\n", t2-t1);
  }

  save_res(out_dir);
}

bool Model::model_exist(string res_dir) {
  string tmp_dir = res_dir + "k" + str_util::itos(K) + ".bs";
  ifstream f(tmp_dir.c_str());
  return f.good();
}

void Model::load_model(string res_dir) {
  cout << "Reload model..." << endl;
  string pt = res_dir + "k" + str_util::itos(K) + ".bs";
  cout << "Load bs: " << pt <<endl;
  int B = bs.size();
  bs.clear();

  ifstream rf(pt.c_str());
  if (!rf) 
    EXIT_ERR("file not find:", pt.c_str());
  int w1, w2, k;
  while (rf >> w1 >> w2 >> k) {
    bs.push_back(Biterm(w1, w2, k));
    nb_z[k] += 1;
    nwz[k][w1] += 1;
    nwz[k][w2] += 1;
  }
  assert(bs.size() == B);
  cout << "Reload complete" << endl;
}

void Model::model_init() {
  srand(time(NULL));
  // random initialize
  for (vector<Biterm>::iterator b = bs.begin(); b != bs.end(); ++b) {
    int k = Sampler::uni_sample(K);
    assign_biterm_topic(*b, k);
  }
}

// input, each line is a doc
// format: wid  wid  wid ...
void Model::load_docs(string dfile) {
  cout << "load docs: " << dfile << endl;
  ifstream rf( dfile.c_str() );
  if (!rf) {
    cout << "file not find:" << dfile << endl;
  exit(-1);
  }

  string line;
  while(getline(rf, line)) {
  Doc doc(line);
  doc.gen_biterms(bs);
  // statistic the exmperial word distribution
  for (int i = 0; i < doc.size(); ++i) {
    int w = doc.get_w(i);
    pw_b[w] += 1;
  }
  }
  
  pw_b.normalize();
}

// sample procedure for ith biterm 
void Model::update_biterm(Biterm& bi) {
  reset_biterm_topic(bi);
  
  // compute p(z|b)
  Pvec<double> pz;
  compute_pz_b(bi, pz);

  // sample topic for biterm b
  int k = Sampler::mult_sample(pz.to_vector());
  assign_biterm_topic(bi, k);
}

// reset topic assignment of biterm i
void Model::reset_biterm_topic(Biterm& bi) {
  int k = bi.get_z();
  // not is the background topic
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  
  nb_z[k] -= 1; // update number of biterms in topic K
  nwz[k][w1] -= 1;  // update w1's occurrence times in topic K
  nwz[k][w2] -= 1;
  assert(nb_z[k] > -10e-7); 
  assert(nwz[k][w1] > -10e-7); 
  assert(nwz[k][w2] > -10e-7);
  bi.reset_z();
}

// recomput count from all biterm-topic assignment
void Model::recompute_count() {
  // remove all numbers in it!
  nb_z.fill(0);
  nwz.fill(0);

  for (int b = 0; b < bs.size(); ++b) {
    int k = bs[b].get_z();
    int w1 = bs[b].get_wi();
    int w2 = bs[b].get_wj();
  
    nb_z[k] += 1;
    nwz[k][w1] += 1;
    nwz[k][w2] += 1;
  }
  // cout << bs[0].get_z() << endl;
}

// compute p(z|w_i, w_j)
void Model::compute_pz_b(Biterm& bi, Pvec<double>& pz) {
  pz.resize(K);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  
  double pw1k, pw2k, pk;
  for (int k = 0; k < K; ++k) {
  // avoid numerical problem by mutipling W
  if (has_background && k == 0) {
    pw1k = pw_b[w1];
    pw2k = pw_b[w2];
  }
  else {
    pw1k = (nwz[k][w1] + beta) / (2 * nb_z[k] + W * beta);
    pw2k = (nwz[k][w2] + beta) / (2 * nb_z[k] + 1 + W * beta);
  }
  pk = (nb_z[k] + alpha) / (bs.size() + K * alpha);
  pz[k] = pk * pw1k * pw2k;
  }

  //pz.normalize();
}

// assign topic k to biterm i
void Model::assign_biterm_topic(Biterm& bi, int k) {
  bi.set_z(k);
  int w1 = bi.get_wi();
  int w2 = bi.get_wj();
  nb_z[k] += 1;
  nwz[k][w1] += 1;
  nwz[k][w2] += 1;
}


void Model::save_res(string dir, int iter) {
  string pt = dir + "pz." + str_util::itos(iter);
  cout << "\nwrite p(z): " << pt << endl;
  save_pz(pt);
  
  string pt2 = dir + "pw_z." + str_util::itos(iter);
  cout << "write p(w|z): " << pt2 << endl;
  save_pw_z(pt2);

  string pt3 = dir + "bs";
  cout << "write biterms: " << pt3 << endl;
  save_bs(pt3);
}

// p(z) is determinated by the overall proportions
// of biterms in it
void Model::save_pz(string pt) {
  Pvec<double> pz(nb_z);
  pz.normalize(alpha);
  pz.write(pt);
}

void Model::save_pw_z(string pt) {
  Pmat<double> pw_z(K, W);   // p(w|z) = phi, size K * M
  ofstream wf(pt.c_str());
  for (int k = 0; k < K; k++) {
  for (int w = 0; w < W; w++) 
    pw_z[k][w] = (nwz[k][w] + beta) / (nb_z[k] * 2 + W * beta);

  wf << pw_z[k].str() << endl;
  }
}

void Model::save_bs(string pt) {
  ofstream wf(pt.c_str());
  for (int b = 0; b < bs.size(); ++b) {
    wf << bs[b].str() << endl;
  }
}
