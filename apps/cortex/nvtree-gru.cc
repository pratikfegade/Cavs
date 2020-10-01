#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "common.h"

using namespace std;

DEFINE_bool(mem, false, "Mem profiling");
DEFINE_int32(batch_size, 1, "batch");
DEFINE_int32(max_num_nodes, 5000, "input size");
DEFINE_int32(hidden_size, 256, "hidden size");
DEFINE_int32(max_batches, 1, "iterations");
DEFINE_double(init_scale, 0.1f, "init random scale of variables");
DEFINE_string(input_file, "", "input sentences");
DEFINE_string(graph_file, "", "graph dependency");

DEFINE_validator(input_file, &IsNonEmptyMessage);
DEFINE_validator(graph_file, &IsNonEmptyMessage);

class TreeModel : public GraphSupport {
 public:
  TreeModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    embedding = Sym::Variable(DT_FLOAT, {FLAGS_max_num_nodes, 3 * FLAGS_hidden_size},
			      Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));

    U = Sym::Variable(DT_FLOAT, {3 * FLAGS_hidden_size * FLAGS_hidden_size},
		      Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {3 * FLAGS_hidden_size}, Sym::Zeros());
    One = Sym::Variable(DT_FLOAT, {FLAGS_hidden_size}, Sym::Zeros());

    // prepare parameter symbols
    b_r = B.Slice(0, FLAGS_hidden_size);
    b_h = B.Slice(FLAGS_hidden_size, FLAGS_hidden_size);
    b_z = B.Slice(2 * FLAGS_hidden_size, FLAGS_hidden_size);

    U_rz = U.Slice(0, 2 * FLAGS_hidden_size * FLAGS_hidden_size).Reshape({FLAGS_hidden_size, 2 * FLAGS_hidden_size});
    U_h   = U.Slice(2 * FLAGS_hidden_size * FLAGS_hidden_size, FLAGS_hidden_size * FLAGS_hidden_size).Reshape({FLAGS_hidden_size, FLAGS_hidden_size});
  }

  void Node() override {
    Sym left = Gather(0, {FLAGS_hidden_size});
    Sym right = Gather(1, {FLAGS_hidden_size});
    Sym h_lr = left + right;

    Sym x = Pull(0, {1});
    Sym xW = x.EmbeddingLookup(embedding.Mirror());

    Sym xW_r, xW_h, xW_z;
    tie(xW_r, xW_h, xW_z) = xW.Split3();

    Sym hU_rz = Sym::MatMul(h_lr.Reshape({1, FLAGS_hidden_size}), U_rz.Mirror()).Reshape({FLAGS_hidden_size * 2});
    Sym hU_r, hU_z;
    tie(hU_r, hU_z) = hU_rz.Split2();

    Sym r = Sym::Sigmoid(hU_r + xW_r + b_r);
    Sym z = Sym::Sigmoid(hU_z + xW_z + b_z);

    Sym hU_h = Sym::MatMul(r.Reshape({1, FLAGS_hidden_size}), U_h.Mirror()).Reshape({FLAGS_hidden_size});
    Sym h = Sym::Tanh(hU_h + xW_h + b_h);

    Sym ht = (One - z) * h + z * h_lr;

    Scatter(ht.Mirror());
    Push(ht.Mirror());
  }

 private:
  Sym W, U, B, One;
  Sym embedding;
  Sym b_r;
  Sym b_h;
  Sym b_z;

  Sym U_rz;
  Sym U_h;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SSTReader sst_reader(FLAGS_input_file, FLAGS_graph_file);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY}, "CPU");
  Sym word_idx = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY});

  TreeModel model(graph, word_idx);
  Sym graph_output = model.Output();
  // Session sess(OPT_BATCHING + OPT_STREAMMING);
  // Session sess(OPT_BATCHING + OPT_FUSION);
  Session sess(OPT_BATCHING);
  int max_batches = FLAGS_max_batches;
  vector<float> input_data(FLAGS_batch_size*SST_MAX_DEPENDENCY, -1);
  vector<int>   graph_data(FLAGS_batch_size*SST_MAX_DEPENDENCY, -1);

  std::cout << "[CONFIG] " << FLAGS_batch_size << " " << FLAGS_hidden_size << std::endl;

  CHECK(FLAGS_batch_size * FLAGS_max_batches <= SST_NUM_SAMPLES);

  float all_time = 0.0;
  int num_nodes = 0;
  for (int j = 0; j < max_batches; j++) {
    int this_num_nodes = 0;
    sst_reader.next_batch(FLAGS_batch_size, &graph_data, &input_data, &this_num_nodes);
    num_nodes += this_num_nodes;

    auto runner = [&] {
      // time_point<system_clock> start = system_clock::now();
      Timing::TimingBegin("Overall");

      sess.Run({graph_output}, {{graph,    graph_data.data()},
	    {word_idx, input_data.data()}});

      Timing::TimingEnd("Overall");
      float ms_time = Timing::TimeInMs("Overall");
      Timing::Reset("Overall");
      return ms_time * 1000.0;
      // time_point<system_clock> end = system_clock::now();
      // std::chrono::duration<float> fs = (end - start);
      // return duration_cast<microseconds>(fs).count();
    };
    all_time += measure_time(runner, FLAGS_mem);
  }

  long model_size_in_bytes = -1000000000;
  report_time(all_time, num_nodes, max_batches, model_size_in_bytes);

  return 0;
}
