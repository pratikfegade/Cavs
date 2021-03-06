#include "cavs/midend/session_base.h"
#include "cavs/midend/allocator.h"
#include "cavs/util/logging.h"

#include <unordered_map>

using std::string;
using std::unordered_map;

namespace midend {

const Tensor* SessionBase::GetTensor(
    const string& name, bool recursive) const {
  if (scoped_tensor_map_.find(name) != scoped_tensor_map_.end())
    return &(scoped_tensor_map_.at(name));
  else if (recursive) {
    CHECK(name.find_last_of(":") != string::npos);
    string tensor_name = name.substr(name.find_last_of(":")+1);
    CHECK(tensor_name.length());
    return raw_tensor_map_.find(tensor_name) == raw_tensor_map_.end() ?
           NULL : &(raw_tensor_map_.at(tensor_name));
  }else {
    return NULL;
  }
}

void SessionBase::InsertTensor(const Tensor& t){
  CHECK(t.name().find_last_of(":") != string::npos) 
       << "tensor name must be a scoped name: " << t.name();
  CHECK(scoped_tensor_map_.find(t.name()) == scoped_tensor_map_.end());
  scoped_tensor_map_[t.name()] = t;
  string tensor_name = t.name().substr(t.name().find_last_of(":")+1);
  CHECK(tensor_name.length());
  if (raw_tensor_map_.find(tensor_name) != raw_tensor_map_.end()) {
    CHECK(raw_tensor_map_.at(tensor_name).buf_.get() == t.buf_.get());
  }else {
    raw_tensor_map_[tensor_name] = t;
  }
}

OpContext* SessionBase::GetContext(const Node* node) {
  OpContext* ctxt  = new OpContext();
  CHECK(node->IsSingleNode());
  const OpDef& op_def = dynamic_cast<const SingleNode*>(node)->op_def();
  for (auto* input : node->input()) {
    const Tensor* t = GetTensor(input->scoped_name()); 
    CHECK(t) << "Getting " << input->scoped_name();
    ctxt->AppendInput(t);
  }
  for (auto* output : node->output()) {
    const Tensor* t = GetTensor(output->scoped_name());
    if (!t) {
      const Tensor* upper_t = GetTensor(output->scoped_name(), true);
      if (upper_t) {
        VLOG(V_DEBUG) << "Found underlying tensor(" << upper_t->name()
                      << "," << upper_t->count() << " elements"
                      << ") for " << output->scoped_name()
                      << " with shape info: " << output->shape().DebugString();
        Tensor out(output->scoped_name(), *upper_t);
        out.Reshape(output->shape());
        InsertTensor(out);
      }else if (GetSingleArg<bool>(op_def, "ShareMemory", false)) {
        //CHECK(node->inputs_size() == 1); //reshape need two inputs
        CHECK(node->output_size() == 1); 
        Tensor out(output->scoped_name(),
            *GetTensor(node->input(0)->scoped_name()));
        out.Reshape(output->shape());
        VLOG(V_DEBUG) << "Share Memory Tensor" << out.debug_info();
        InsertTensor(out);
      }else {
        CHECK(output->shape().dim_size() > 0);
        TensorShape shape(output->shape()); 
        Allocator* alloc = GetAllocator(op_def); 
        CHECK_NOTNULL(alloc);
        VLOG(V_DEBUG) << "allocating tensor for " << output->scoped_name()
                      << " with shape info: " << shape.debug_info();
        Tensor out(output->scoped_name(), alloc, op_def.dtype(), std::move(shape));
        VLOG(V_DEBUG) << out.debug_info();
        InsertTensor(out);
      }
    }
    t = GetTensor(output->scoped_name());
    CHECK(t) << t->debug_info();
    if (node->IsStatefulOp()) {
      CHECK(node->output_size() == 1);
      const_cast<Tensor*>(t)->SetZeroInitEnforced(); 
    }
    ctxt->AppendOutput(const_cast<Tensor*>(t));
  }
  return ctxt;
}

string SessionBase::debug_info() const {
  string ret;
  for (auto& one_pair : scoped_tensor_map_)
    ret += one_pair.first + "\t";
  return ret;
}

namespace session_factory {

typedef std::unordered_map<string, 
                           SessionRegister::Factory> SessionRegistry;
static SessionRegistry* GlobalSessionRegistry() {
  static SessionRegistry* global_session_registry 
    = new SessionRegistry();
  return global_session_registry;
}
void SessionRegister::InitInternal(
    const string& name, Factory factory) {
  GlobalSessionRegistry()->insert(std::make_pair(name, factory));
}

} //namespace session_factory

SessionBase* GetSession(const string& name, int opt) {
  if (session_factory::GlobalSessionRegistry()->count(name) == 0)
    return NULL;
  else
    return session_factory::GlobalSessionRegistry()->at(name)(opt);
}

} //namespace midend
