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

void Model::run(string doc_pt, string res_dir) {
  load_docs(doc_pt);

  if ( model_exist(res_dir) ) {
  	load_model(res_dir);
  }
  model_init();
  int N = bs.size();
  int P = 32;

  cout << "Begin iteration" << endl;
  string out_dir = res_dir + "k" + str_util::itos(K) + ".";


  for (int it = 1; it < n_iter + 1; ++it) {
    cout << "iter " << it << '/' << n_iter;
    fflush(stdout);
    double t1 = omp_get_wtime();
    // #pragma omp parallel for num_threads(P)
    #pragma omp parallel for num_threads(P)
    for (int b = 0; b < bs.size(); ++b) {
      update_biterm(bs[b]);
    }
    // {
      // int p = omp_get_thread_num();
      // printf("Thread %d, on cpu %d\n", omp_get_thread_num(), sched_getcpu());
      // for (int b = p*N/P; b < (p+1)*N/P; ++b) {
    // }
    recompute_count();
    save_raw_res(out_dir);
    if (it % save_step == 0) {
      save_res(out_dir);
    }
    double t2 = omp_get_wtime();
    printf("Time per iter: %f\n", t2-t1);
  }

  save_res(out_dir);
}

bool Model::model_exist(string res_dir) {
  string tmp_dir = res_dir + "k" + str_util::itos(K) + ".pz";
  ifstream f(tmp_dir.c_str());
  return f.good();
}

void Model::load_model(string res_dir) {
  // string pt = res_dir + "k" + str_util::itos(K) + ".nz";
  // cout << "load nb_z:" << pt <<endl;
  // nb_z.loadFile(pt);
  // assert(nb_z.sum() == bs.size());

  // string pt2 = res_dir + "k" + str_util::itos(K) + ".nw_z";
  // cout << "load nwz:" << pt2 <<endl;
  // nwz.load(pt2);
  // printf("n(z)=%d, n(w)=%d\n", nwz.rows(), nwz.cols());
  cout << "Reload model" << endl;
  string pt = res_dir + "k" + str_util::itos(K) + ".pz";
  cout << "Load nb_z:" << pt <<endl;
  Pvec<double> pz;
  pz.loadFile(pt);
  // cout << bs.size() << " " << pz.sum() << endl;
  nb_z = pz.de_normalize(bs.size(), alpha);

  string pt2 = res_dir + "k" + str_util::itos(K) + ".pw_z";
  cout << "Load nwz:" << pt2 <<endl;
  Pmat<double> pw_z;
  pw_z.load(pt2);
  for(size_t k = 0; k < K; k++) {
    nwz[k] = pw_z[k].de_normalize(2* nb_z[k], beta);
  }
  // nwz.load(pt2);
  printf("n(z)=%d, n(w)=%d\n", nwz.rows(), nwz.cols());
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
  assert(nb_z[k] > -10e-7 && nwz[k][w1] > -10e-7 && nwz[k][w2] > -10e-7);
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

void Model::save_raw_res(string dir) {
  string pt = dir + "nz";
  cout << "\nwrite n(z): " << pt << endl;
  save_nz(pt);
  
  string pt2 = dir + "nw_z";
  cout << "write n(w|z): " << pt2 << endl;
  save_nw_z(pt2);
}


void Model::save_res(string dir) {
  string pt = dir + "pz";
  cout << "\nwrite p(z): " << pt << endl;
  save_pz(pt);
  
  string pt2 = dir + "pw_z";
  cout << "write p(w|z): " << pt2 << endl;
  save_pw_z(pt2);
}

void Model::save_nz(string pt) {
  nb_z.write(pt);
}


void Model::save_nw_z(string pt) {
  ofstream wf(pt.c_str());
  for (int k = 0; k < K; k++) {
    wf << nwz[k].str() << endl;
  }
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
