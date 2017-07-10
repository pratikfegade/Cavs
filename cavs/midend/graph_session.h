#ifndef CAVS_MIDEND_GRAPH_SESSION_H_
#define CAVS_MIDEND_GRAPH_SESSION_H_

#include "cavs/midend/session_base.h"
#include "cavs/midend/tensor.h"
#include "cavs/midend/graph_scheduler.h"

namespace midend {

class SessionBase;

class GraphSession : public SessionBase {
 public:
  GraphSession(SessionBase* sb, const Scope* rs,
      GraphScheduler* gs, int max_graph_node_count)
    : global_sess_(sb), running_scope_(rs), 
      gscheduler_(gs), MAX_NODE_(max_graph_node_count) {}
  const Tensor* GetTensor(const std::string& name, bool recursive = false) const override;
  OpContext* GetContext(const Node* node) override;
  std::string TensorNameInFunctionContext(const Edge* e) const;
  virtual int session_type() const { return GRAPH; }
  GraphScheduler* graph_scheduler() { return gscheduler_; }

 private:
  SessionBase* global_sess_;
  const Scope* running_scope_;
  GraphScheduler* gscheduler_;
  const int MAX_NODE_;
};

} //namespace midend

#endif
