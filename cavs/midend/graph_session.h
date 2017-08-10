#ifndef CAVS_MIDEND_GRAPH_SESSION_H_
#define CAVS_MIDEND_GRAPH_SESSION_H_

#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/proto/opt.pb.h"

namespace midend {

class SessionBase;

class GraphSession : public SessionBase {
 public:
  GraphSession(SessionBase* sb, const std::string& name, int max_graph_node_count)
    : SessionBase(sb->opt_type()),
      global_sess_(sb),name_(name), MAX_NODE_(max_graph_node_count) {
    CHECK(name_.length());
    scope_ = main_scope();
    //if (sb->opt_type() & OPT_FUSION) {
      //this->AddType(OFUSION);
    //}
    if (opt_type() & OPT_BATCHING) {
      gscheduler_ = new BatchGraphScheduler();
    }else {
      gscheduler_ = new SerialGraphScheduler();
    }
  }
  const Tensor* GetTensor(const std::string& name, bool recursive = false) const override;
  OpContext* GetContext(const Node* node) override;
  std::string TensorNameInFunctionContext(const Edge* e) const;
  //virtual int session_type() const { return GRAPH; }
  GraphSchedulerBase* graph_scheduler() { return gscheduler_; }

 private:
  SessionBase* global_sess_;
  const Scope* scope_;
  GraphSchedulerBase* gscheduler_;
  const int MAX_NODE_;
  std::string name_;
};

GraphSession* GetGraphSession(const std::string& name);
bool InsertGraphSession(const std::string& name, GraphSession* sess);

} //namespace midend

#endif
