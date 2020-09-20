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

    Wi = Sym::Variable(DT_FLOAT, {3 * FLAGS_hidden * FLAGS_hidden},
		       Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    Whr = Sym::Variable(DT_FLOAT, {FLAGS_hidden * FLAGS_hidden},
		       Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Whh = Sym::Variable(DT_FLOAT, {FLAGS_hidden * FLAGS_hidden},
		       Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    Whz = Sym::Variable(DT_FLOAT, {FLAGS_hidden * FLAGS_hidden},
		       Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {3 * FLAGS_hidden}, Sym::Zeros());

    // prepare parameter symbols
    Br = B.Slice(0, FLAGS_hidden);
    Bh = B.Slice(FLAGS_hidden, FLAGS_hidden);
    Bz = B.Slice(2 * FLAGS_hidden, FLAGS_hidden);
  }

  void Node() override {
    Sym h_l = Gather(0, {2 * FLAGS_hidden});
    Sym h_r = Gather(1, {2 * FLAGS_hidden});
    Sym htm1 = h_l + h_r;

    // Pull the input word
    Sym x = Pull(0, {1});
    Sym input = x.EmbeddingLookup(embedding.Mirror());

    Sym Xi = Sym::MatMul(input, Wi.Reshape({FLAGS_hidden, 3 * FLAGS_hidden}).Mirror()).Reshape({FLAGS_hidden * 3});
    Sym Xr, Xz, Xh;
    tie(Xr, Xz, Xh) = Xi.Split4();


    Sym r = htm1 * (Br + Sym::MatMul(Whr.Mirror(), htm1) + Xr).Sigmoid();
    Sym z = (Bz + Sym::MatMul(Whz.Mirror(), htm1) + Xz).Sigmoid();
    Sym h = (Bh + Sym::MatMul(Whh.Mirror(), r) + Xz).Tanh();
    Sym ht = (1 - z) * h + z * htm1;

    Scatter(ht.Mirror());
    Push(ht.Mirror());
  }

 private:
  Sym Wi, Whr, Whh, Whz, B;
  Sym embedding;
  Sym Br, Bh, Bz;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Reader sst_reader(FLAGS_input_file, FLAGS_graph_file);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY});

  Sym weight   = Sym::Variable(DT_FLOAT, {FLAGS_vocab_size, FLAGS_hidden},
                               Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  Sym bias     = Sym::Variable(DT_FLOAT, {1, FLAGS_vocab_size}, Sym::Zeros());
  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  Session sess(OPT_BATCHING + OPT_FUSION + OPT_STREAMMING);
  int iterations = FLAGS_iters;
  vector<float> input_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);
  vector<int>   graph_data(FLAGS_batch_size*MAX_DEPENDENCY, -1);

  std::cout << "[CONFIG] " << FLAGS_batch_size << " " << FLAGS_hidden << std::endl;

  float all_time = 0.0;
  int num_nodes = 0;
  for (int j = 0; j < max_batches; j++) {
    int this_num_nodes = 0;
    sst_reader.next_batch(&graph_data, &input_data, &this_num_nodes);
    num_nodes += this_num_nodes;

    auto runner = [&] {
      time_point<system_clock> start = system_clock::now();

      sess.Run({graph_output}, {{graph, graph_data.data()},
	                        {word_idx,input_data.data()}});

      time_point<system_clock> end = system_clock::now();
      std::chrono::duration<float> fs = (end - start);
      return duration_cast<microseconds>(gen_fs).count();
    };
    all_time += measure_time(runner);
  }

  report_time(all_time, num_nodes, num_batches);

  return 0;
a}
