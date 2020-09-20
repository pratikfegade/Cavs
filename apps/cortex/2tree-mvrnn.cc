#include "cavs/frontend/cxx/sym.h"
#include "cavs/frontend/cxx/graphsupport.h"
#include "cavs/frontend/cxx/session.h"
#include "cavs/proto/opt.pb.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

DEFINE_int32 (batch,      10,       "batch"                         );
DEFINE_int32 (hidden,     64,      "hidden size"                   );
DEFINE_int32 (tree,       64,      "the leaf number of the tree"   );
DEFINE_int32 (iters,      1,      "iterations"                    );
DEFINE_double(init_scale, 0.1f,     "init random scale of variables");
DEFINE_double(lr,         0.00001f, "learning rate"                 );


class TreeFCModel : public GraphSupport {
 public:
  TreeFCModel(const Sym& graph_ph, const Sym& vertex_ph) :
    GraphSupport(graph_ph, vertex_ph) {
    W = Sym::Variable(DT_FLOAT, {2 * FLAGS_hidden, FLAGS_hidden}, Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));//v2
    B = Sym::Variable(DT_FLOAT, {FLAGS_hidden, FLAGS_hidden}, Sym::Uniform(-FLAGS_init_scale, FLAGS_init_scale));
  }

  void Node() override {
    Sym left = Gather(0, {FLAGS_hidden * FLAGS_hidden}).Reshape({FLAGS_hidden, FLAGS_hidden});
    Sym right = Gather(1, {FLAGS_hidden * FLAGS_hidden}).Reshape({FLAGS_hidden, FLAGS_hidden});
    Sym fc = Sym::MatMul(Sym::Concat({left, right}).Reshape({FLAGS_hidden, 2*FLAGS_hidden}), W.Mirror()) +
      B.Reshape({FLAGS_hidden, FLAGS_hidden}).Mirror();
    Sym res = fc.Reshape({FLAGS_hidden * FLAGS_hidden}).Tanh();
    Scatter(res.Mirror());
    Push(res.Mirror());
  }

 private:
  Sym W, B;
};

void binaryTree(vector<int>* graph) {
  vector<int> one_tree;
  int count = 0;
  for (int width = FLAGS_tree; width > 1; width >>= 1) {
    count += width;
    for (int i = 0; i < width; i++) {
      one_tree.push_back(i/2+count);
    }
  }
  one_tree.push_back(-1);
  CHECK(one_tree.size() == 2*FLAGS_tree-1);

  graph->clear();
  for (int i = 0; i < FLAGS_batch; i++) {
    std::copy(one_tree.begin(), one_tree.end(), graph->begin() + i*(2*FLAGS_tree-1));
  }
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Sym graph    = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, 2*FLAGS_tree-1}, "CPU");//p0
  Sym vertex   = Sym::Placeholder(DT_FLOAT, {FLAGS_batch, 2*FLAGS_tree-1});//p1

  TreeFCModel model(graph, vertex);
  Sym graph_output = model.Output();

  //Session sess;
  Session sess(OPT_BATCHING + OPT_FUSION);
  vector<int>   graph_data(FLAGS_batch*(2*FLAGS_tree-1), -1);

  binaryTree(&graph_data);
  for (int j = 0; j < FLAGS_iters; j++) {
    sess.Run({graph_output}, {{graph, graph_data.data()}});
  }


  return 0;
}
