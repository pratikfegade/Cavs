#ifndef CAVS_MIDEND_SESSION_H_
#define CAVS_MIDEND_SESSION_H_

#include "cavs/midend/op_chain_def.pb.h"
#include "cavs/midend/tensor.h"

namespace cavs {

class SessionBase {
 public:
  SessionBase() {}
  //SessionBase(const OpChainDef& def) : op_chain_def_(def) {}
  virtual void SetOpChainDef(const OpChainDef& def) {
    op_chain_def_ = def;
  }
  const Tensor* GetTensor(const string& name) const;
  void InsertTensor(const Tensor& t);
  virtual void Run(const vector<string>& output_name, 
                   vector<Tensor>* output_tensors) = 0;
 protected:
  //Tensor* CreateTensor(const OpDef& op_def);
  unordered_map<string, Tensor> tensor_map_;
  OpChainDef op_chain_def_;
};

SessionBase* GetSession(const string& name);

#define REGISTER_SESSION_BUILDER(key, ...)                 \
    REGISTER_SESSION_BUILDER_UNIQ(__COUNTER__, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_UNIQ(ctr, key, ...)       \
    REGISTER_SESSION_BUILDER_CONCAT(ctr, key, __VA_ARGS__)
#define REGISTER_SESSION_BUILDER_CONCAT(ctr, key, ...)     \
    static session_factory::SessionRegister                \
        register_body_##ctr##_session(key,                 \
            []() -> SessionBase* {                         \
                return new __VA_ARGS__();                  \
              }) 

namespace session_factory {

class SessionRegister {
 public:
  typedef SessionBase* (*Factory)();
  SessionRegister(const string& name, Factory factory) {
    InitInternal(name, factory); 
  }
 private:
  void InitInternal(const string& name, Factory factory); 
};

} //namespace session_factory

} //namespace cavs

#endif