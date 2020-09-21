#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"
#include "cavs/util/timing.h"

#include <iostream>
#include <fstream>
#include <vector>

#include "common.h"

using namespace std;

DEFINE_int32(batch_size, 10, "batch");
DEFINE_int32(hidden_size, 256, "hidden size");
DEFINE_int32(max_batches, 100, "iterations");
DEFINE_double(init_scale, 0.1f, "init random scale of variables");
DEFINE_string(input_file, "", "input sentences");
DEFINE_string(graph_file, "", "graph dependency");

DEFINE_validator(input_file, &IsNonEmptyMessage);
DEFINE_validator(graph_file, &IsNonEmptyMessage);

class TreeFCModel : public GraphSupport {
 public:
  TreeFCModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    W = Sym::Variable(DT_FLOAT, {2 * FLAGS_hidden_size, FLAGS_hidden_size}, Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
    B = Sym::Variable(DT_FLOAT, {FLAGS_hidden_size}, Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  }

  void Node() override {
    Sym left = Gather(0, {FLAGS_hidden_size});
    Sym right = Gather(1, {FLAGS_hidden_size});
    Sym fc = Sym::MatMul(Sym::Concat({left, right}).Reshape({1, 2*FLAGS_hidden_size}), W.Mirror()).Reshape({FLAGS_hidden_size});
    Sym biased = fc.Mirror() + B.Mirror();
    Sym res = biased.Relu();
    Scatter(res.Mirror());
    Push(res.Mirror());
  }

 private:
  Sym W, B;
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  SSTReader sst_reader(FLAGS_input_file, FLAGS_graph_file);

  Sym graph = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY}, "CPU");
  Sym vertex = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, SST_MAX_DEPENDENCY});

  TreeFCModel model(graph, vertex);
  Sym graph_output = model.Output();

  // TODO(ppf): Streaming leads to a segfault here. Debug if time.
  Session sess(OPT_BATCHING + OPT_FUSION/* + OPT_STREAMMING*/);

  int max_batches = FLAGS_max_batches;
  vector<float> input_data(FLAGS_batch_size * SST_MAX_DEPENDENCY, -1);
  vector<int> graph_data(FLAGS_batch_size * SST_MAX_DEPENDENCY, -1);

  std::cout << "[CONFIG] " << FLAGS_batch_size << " " << FLAGS_hidden_size << std::endl;

  CHECK(FLAGS_batch_size * max_batches <= SST_NUM_SAMPLES);

  float all_time = 0.0;
  int num_nodes = 0;
  for (int j = 0; j < max_batches; j++) {
    int this_num_nodes = 0;
    sst_reader.next_batch(FLAGS_batch_size, &graph_data, &input_data, &this_num_nodes);
    num_nodes += this_num_nodes;

    auto runner = [&] {
      time_point<system_clock> start = system_clock::now();

      sess.Run({graph_output}, {{graph, graph_data.data()}});

      time_point<system_clock> end = system_clock::now();
      std::chrono::duration<float> fs = (end - start);
      return duration_cast<microseconds>(fs).count();
    };
    all_time += measure_time(runner);
  }

  report_time(all_time, num_nodes, max_batches);

  return 0;
}
