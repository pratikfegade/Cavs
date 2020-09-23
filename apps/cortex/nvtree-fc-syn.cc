
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

DEFINE_bool(mem, false, "Mem profiling");
DEFINE_int32(batch_size, 1, "batch");
DEFINE_int32(hidden_size, 256, "hidden size");
DEFINE_int32(max_batches, 1, "iterations");
DEFINE_double(init_scale, 0.1f, "init random scale of variables");
DEFINE_int32(height, 7, "height");

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
    Sym fc = Sym::MatMul(Sym::Concat({left, right}).Reshape({1, 2*FLAGS_hidden_size}), W.Mirror());
    Sym biased = fc.Reshape({FLAGS_hidden_size}) + B.Mirror();
    Sym res = biased.Relu();
    Scatter(res.Mirror());
    Push(res.Mirror());
  }

 private:
  Sym W, B;
};

void binaryTree(vector<int>* graph) {
  vector<int> one_tree;
  int count = 0;
  for (int width = (1 << FLAGS_height); width > 1; width >>= 1) {
    count += width;
    for (int i = 0; i < width; i++) {
      one_tree.push_back(i/2+count);
    }
  }
  one_tree.push_back(-1);
  CHECK(one_tree.size() == 2*(1 << FLAGS_height)-1);
  std::cout << "[NODES] " << one_tree.size() << std::endl;
  graph->clear();
  for (int i = 0; i < FLAGS_batch_size; i++) {
    std::copy(one_tree.begin(), one_tree.end(), graph->begin() + i*(2*(1 << FLAGS_height)-1));
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Sym graph = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, 2*(1 << FLAGS_height)-1}, "CPU");
  Sym vertex = Sym::Placeholder(DT_FLOAT, {FLAGS_batch_size, 2*(1 << FLAGS_height)-1});

  TreeFCModel model(graph, vertex);
  Sym graph_output = model.Output();

  Session sess(OPT_BATCHING + OPT_FUSION);
  vector<int> graph_data(FLAGS_batch_size*(2*(1 << FLAGS_height)-1), -1);

  float all_time = 0.0;
  int num_nodes = 0;
  binaryTree(&graph_data);
  for (int j = 0; j < FLAGS_max_batches; j++) {
    int this_num_nodes = 0;
    num_nodes += FLAGS_batch_size * (2 * (1 << FLAGS_height) - 1);

    auto runner = [&] {
      time_point<system_clock> start = system_clock::now();

      sess.Run({graph_output}, {{graph, graph_data.data()}});

      time_point<system_clock> end = system_clock::now();
      std::chrono::duration<float> fs = (end - start);
      return duration_cast<microseconds>(fs).count();
    };
    all_time += measure_time(runner, FLAGS_mem);
  }

  long model_size_in_bytes = 4 * (2 * FLAGS_hidden_size * FLAGS_hidden_size +
				  FLAGS_hidden_size);
  report_time(all_time, num_nodes, FLAGS_max_batches, model_size_in_bytes);
  return 0;
}
