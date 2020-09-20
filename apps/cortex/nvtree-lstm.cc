#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "common.h"

using namespace std;

DEFINE_int32(batch_size, 10, "batch");
DEFINE_int32(vocab_size, 21701, "input size");
DEFINE_int32(hidden, 256, "hidden size");
DEFINE_int32(max_batches, 100, "iterations");
DEFINE_double(init_scale, 0.1f, "init random scale of variables");
DEFINE_string(input_file, "", "input sentences");
DEFINE_string(graph_file, "", "graph dependency");

DEFINE_validator(input_file, &IsNonEmptyMessage);
DEFINE_validator(graph_file, &IsNonEmptyMessage);

class TreeModel : public GraphSupport {
 public:
  TreeModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    embedding = Sym::Variable(DT_FLOAT, {FLAGS_vocab_size, FLAGS_hidden},
                            Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    W = Sym::Variable(DT_FLOAT, {4 * FLAGS_hidden * FLAGS_hidden},
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
    Sym xW = Sym::MatMul(x, W.Reshape({FLAGS_hidden, 4 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 4});
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

  SSTReader sst_reader(FLAGS_input_file, FLAGS_graph_file);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY});

  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_vocab_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_vocab_size}, Sym::Zeros());
  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Session sess(OPT_BATCHING + OPT_FUSION + OPT_STREAMMING);
  int max_batches = FLAGS_max_batches;
  vector<float> input_data(FLAGS_batch_size*SST_MAX_DEPENDENCY, -1);
  vector<int>   graph_data(FLAGS_batch_size*SST_MAX_DEPENDENCY, -1);

  std::cout << "[CONFIG] " << FLAGS_batch_size << " " << FLAGS_hidden << std::endl;

  float all_time = 0.0;
  int num_nodes = 0;
  for (int j = 0; j < max_batches; j++) {
    int this_num_nodes = 0;
    sst_reader.next_batch(&graph_data, &input_data, &this_num_nodes);
    num_nodes += this_num_nodes;

    auto runner = [&] {
      time_point<system_clock> start = system_clock::now();

      sess.Run({graph_output}, {{graph,    graph_data.data()},
	    {word_idx, input_data.data()}});

      time_point<system_clock> end = system_clock::now();
      std::chrono::duration<float> fs = (end - start);
      return duration_cast<microseconds>(gen_fs).count();
    };
    all_time += measure_time(runner);
  }

  report_time(all_time, num_nodes, num_batches);

  return 0;
}
