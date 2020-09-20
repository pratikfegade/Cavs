#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace std::chrono;

DEFINE_int32 (batch_size,  10,       "batch");
DEFINE_int32 (input_size,  21701,    "input size");
DEFINE_int32 (embedding,   256,      "embedding size");
DEFINE_int32 (hidden,      256,      "hidden size");
DEFINE_int32 (epoch,       1,        "epochs");
DEFINE_int32 (iters,       100,    "iterations");
DEFINE_double(init_scale,  0.1f,     "init random scale of variables");
DEFINE_double(lr,          1.f,      "learning rate");

int MAX_LEN = 56;
int MAX_DEPENDENCY = 111;
int NUM_SAMPLES = 8544;

DEFINE_string(input_file, "/users/shizhenx/projects/Cavs/apps/lstm/data/sst/train/sents_idx.txt", "input sentences");
DEFINE_string(graph_file, "/users/shizhenx/projects/Cavs/apps/lstm/data/sst/train/parents.txt",   "graph dependency");

class Reader {
 public:
  Reader(const string input, const string graph) :
      input_path(input), graph_path(graph),
      input_file(input), graph_file(graph) {
  }

  void next_batch( vector<int>* batch_graph, vector<float>* batch_input) {
    std::fill(batch_input->begin(), batch_input->end(), 0);
    std::fill(batch_graph->begin(), batch_graph->end(), -1);
    int i = 0;
    while (i < FLAGS_batch_size) {
      if (input_file.eof()) { // which mean it reaches the end of the file
        input_file.clear();
        input_file.seekg(0, ios::beg);
        graph_file.clear();
        graph_file.seekg(0, ios::beg);
      }

      string input_str, graph_str;
      getline(input_file, input_str);
      if (input_str.length() > 0) {
        getline(graph_file, graph_str);
        int length;
        process_graph<int>(batch_graph->data() + i*MAX_DEPENDENCY, &length, graph_str);
        CHECK(MAX_DEPENDENCY >= length);

        process_data<float>(batch_input->data() + i*MAX_DEPENDENCY, &length, input_str);
        CHECK(MAX_LEN >= length);
        i++;
      }
    }
  }

 private:
  template<typename T>
  void process_data(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      data[idx] = val;
      idx++;
    }
    *len = idx;
  }

  template<typename T>
  void process_graph(T* data, int* len, const string& str) {
    stringstream input_stream(str);
    int val, idx = 0;
    while (input_stream >> val) {
      // data[idx] = val-1;
      data[idx] = val;
      idx++;
    }
    *len = idx;
    if (data[idx-1] != -1) {
      for (int i = 0; i < idx; ++i) {
	cout << data[i] << " ";
      }
      cout << endl;
      cout << "STR: " << str << endl;
    }
    // CHECK((data[idx-1] == -1)) << str << " " << data[idx - 1] << " " << idx - 1;
    CHECK((data[idx-1] == -1));
  }

  string input_path;
  string graph_path;

  ifstream input_file;
  ifstream graph_file;
};

class TreeModel : public GraphSupport {
 public:
  TreeModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    embedding = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_embedding},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    W = Sym::Variable(DT_FLOAT, {4 * FLAGS_embedding * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    U = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden * FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden}, Sym::Zeros());

    // prepare parameter symbols
    b_i = B.Slice(0, FLAGS_hidden);
    b_f = B.Slice(FLAGS_hidden, FLAGS_hidden);
    b_u = B.Slice(2 * FLAGS_hidden, FLAGS_hidden);
    b_o = B.Slice(3 * FLAGS_hidden, FLAGS_hidden);

    U_iou = U.Slice(0, 3 * FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, 3 * FLAGS_hidden});
    U_f   = U.Slice(3 * FLAGS_hidden * FLAGS_hidden, FLAGS_hidden * FLAGS_hidden).Reshape({FLAGS_hidden, FLAGS_hidden});
  }

  void Node() override {
    // this 4 lines of code (interface) is a bit counter-intuitive that needs a revision
    // Gather from child nodes
    Sym left = Gather(0, {2 * FLAGS_hidden});
    Sym right = Gather(1, {2 * FLAGS_hidden});
    Sym h_l, c_l, h_r, c_r;
    tie(h_l, c_l) = left.Split2();
    tie(h_r, c_r) = right.Split2();
    Sym h_lr = h_l + h_r;

    // Pull the input word
    Sym x = Pull(0, {1});
    x = x.EmbeddingLookup(embedding.Mirror());

    // layout: i, o, u, f
    // start computation
    // xW is 1 x 4*FLAGS_hidden
    Sym xW = Sym::MatMul(x, W.Reshape({FLAGS_embedding, 4 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 4});
    Sym xW_i, xW_o, xW_u, xW_f;
    tie(xW_i, xW_o, xW_u, xW_f) = xW.Split4();

    // hU_iou is 1 x 3*FLAGS_hidden
    Sym hU_iou = Sym::MatMul(h_lr.Reshape({1, FLAGS_hidden}), U_iou.Mirror()).Reshape({FLAGS_hidden * 3});
    Sym hU_i, hU_o, hU_u;
    tie(hU_i, hU_o, hU_u) = hU_iou.Split3();

    // forget gate for every child
    Sym hU_fl = Sym::MatMul(h_l.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});
    Sym hU_fr = Sym::MatMul(h_r.Reshape({1, FLAGS_hidden}), U_f.Mirror()).Reshape({FLAGS_hidden});

    // Derive i, f_l, f_r, o, u
    Sym i = (xW_i + hU_i + b_i.Mirror()).Sigmoid();
    Sym o = (xW_o + hU_o + b_o.Mirror()).Sigmoid();
    Sym u = (xW_u + hU_u + b_u.Mirror()).Tanh();

    Sym f_l = (xW_f + hU_fl + b_f.Mirror()).Sigmoid();
    Sym f_r = (xW_f + hU_fr + b_f.Mirror()).Sigmoid();

    Sym c = i * u + f_l * c_l + f_r * c_r;
    Sym h = o * Sym::Tanh(c.Mirror());

    Scatter(Sym::Concat({h.Mirror(), c.Mirror()}));
    Push(h.Mirror());
  }

 private:
  Sym W, U, B;
  Sym embedding;
  Sym b_i;
  Sym b_f;
  Sym b_u;
  Sym b_o;

  Sym U_iou;
  Sym U_f;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Reader sst_reader(FLAGS_input_file, FLAGS_graph_file);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, MAX_DEPENDENCY});

  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_input_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_input_size}, Sym::Zeros());
  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Session sess(OPT_BATCHING + OPT_FUSION + OPT_STREAMMING);
  int iterations = FLAGS_iters;
  vector<float> input_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);
  vector<int>   graph_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);

  std::cout << "[CONFIG] " << FLAGS_batch_size << " " << FLAGS_embedding << " " << FLAGS_hidden << std::endl;
  float tot_time = 0.0;
  for (int i = 0; i < FLAGS_epoch; i++) {
    for (int j = 0; j < iterations; j++) {
      sst_reader.next_batch(&graph_data, &input_data);
      for (int k = 0; k < 10; ++k) {
	sess.Run({graph_output}, {{graph,    graph_data.data()},
				  {word_idx, input_data.data()}});
      }

      time_point<system_clock> start = system_clock::now();
      for (int k = 0; k < 10; ++k) {
	sess.Run({graph_output}, {{graph,    graph_data.data()},
				  {word_idx, input_data.data()}});
      }
      time_point<system_clock> end = system_clock::now();
      std::chrono::duration<float> exe_fs = (end - start);
      tot_time += duration_cast<nanoseconds>(exe_fs).count() / 10.0;
    }
  }
  cout << tot_time / (FLAGS_epoch * iterations * 1e6) << endl;


  return 0;
}
