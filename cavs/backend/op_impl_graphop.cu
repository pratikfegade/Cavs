#include "cavs/backend/op_impl.h"
#include "cavs/backend/functor_batched_memcpy.cuh"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/midend/tensor.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_util.h"

using ::midend::Tensor;
using ::midend::GraphSchedulerBase;
using std::vector;

namespace backend {

template <typename T>
class GraphGatherOp : public OpImpl {
 public:
  explicit GraphGatherOp(const OpDef& def) :
    OpImpl(def), count_(1), stream_(cudaStreamDefault)  {

    CHECK(def.input_size()  == 0);
    CHECK(def.output_size() == 1);
    CHECK(def.shape_size()  == 1);
    for (auto d : def.shape(0).dim())
      count_ *= d;
    child_offset_ = GetSingleArg<int>(def, "Child");
    CHECK(child_offset_ >= 0);
  }

  void Compute(OpContext* context) override {
    //LOG(FATAL) << "Gather Operator needs further runtime support";
    Tensor* out = context->Output(0);
    GraphSchedulerBase* gs = context->graph_scheduler();
    const Tensor& inp = gs->GetMessagePasser(0);

    const vector<int>& gids = gs->GetJobId();
    context->SetDynDim(gids.size());
    context->ScaleOutputTensor();
    int stride = out->count()/out->dims(0);
    CHECK(stride == count_) << out->debug_info() << op_def_.DebugString();
    VLOG(V_DEBUG) << "Batching jobs of this round: " << gids.size();

    if (gs->HasChild(gids[0])) {
      const vector<int>& child_tensor_ids = gs->GatherTensorIds(gids, child_offset_);
      checkCudaError(cudaMemcpy(gs->gpu_idx_buf(), child_tensor_ids.data(),
                     child_tensor_ids.size()*sizeof(int), cudaMemcpyHostToDevice));
      int blocksPerGrid = gids.size();
      int threadsPerBlock = stride;
      BatchedDynamicSelectedInputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
              out->mutable_data<T>(), stride, inp.data<T>(), stride, gs->gpu_idx_buf(), stride);
    }else {
      checkCudaError(cudaMemset(out->mutable_data<T>(), 0, gids.size()*stride*sizeof(T)));
    }

    /*for (int local_id = 0; local_id < gids.size(); local_id++) {*/
      /*int gid = gids[local_id];*/
      /*if (gs->HasChild(gid)) {*/
        /*CHECK(gs->ChildIds(gid).size() > child_offset_);*/
        /*int child_gid = gs->ChildIds(gid)[child_offset_];*/
        /*VLOG(V_DEBUG) << "Gathering Child_gid:" << child_gid;*/

        /*CHECK(gs->GatherTensorIds(gids, child_offset_).size() > local_id);*/
        /*int child_tensor_id = gs->GatherTensorIds(gids, child_offset_)[local_id];*/
        /*VLOG(V_DEBUG) << "internal gather offset: " << child_tensor_id;*/
        /*//const Tensor& inp = gs->GetMessagePasser(gs->JobIdToInternalTensorId(child_gid));*/
        /*const Tensor& inp = gs->GetMessagePasser(child_tensor_id);*/
        /*CHECK(out->count() >= inp.count())*/
              /*<< "Input count:\t" << inp.count()*/
              /*<< "\t" << inp.debug_size() << "Bytes\n"*/
              /*<< "Output count:\t" << out->count() */
              /*<< "\t" << out->debug_size() << "Bytes";*/
        /*checkCudaError(cudaMemcpy(out->mutable_data<T>()+local_id*stride,*/
                                  /*inp.data<T>(),*/
                                  /*stride*sizeof(T),*/
                                  /*cudaMemcpyDeviceToDevice));*/
      /*}else {*/
        /*VLOG(V_DEBUG) << "[Gathering] No Child_id, Setting Zero";*/
        /*checkCudaError(cudaMemset(out->mutable_data<T>()+local_id*stride,*/
                                  /*0,*/
                                  /*stride*sizeof(T)));*/
      /*}*/
    /*}*/
    out->DebugNumerical<T>();
  }

 private:
  int count_;
  int child_offset_;
  cudaStream_t stream_;
};

template <typename T>
class GraphScatterOp : public OpImpl {
 public:
  explicit GraphScatterOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {

    child_offset_ = GetSingleArg<int>(def, "Child");
    CHECK(child_offset_ >= 0);
  }

  void Compute(OpContext* context) override {
    //LOG(FATAL) << "Scatter Operator needs further runtime support";
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    CHECK(out->count() == inp.count())
          << "Input count:\t" << inp.count()
          << "\t" << inp.debug_size() << "Bytes\n"
          << "Output count:\t" << out->count() 
          << "\t" << out->debug_size() << "Bytes";
    CHECK(inp.IsDynamicShape());
    CHECK(out->IsDynamicShape());
    CHECK(out->dims(0) == inp.dims(0));
    int stride = out->count()/out->dims(0);
    CHECK(stride == inp.count()/inp.dims(0));

    out->SetOffsetWithId(0);
    GraphSchedulerBase* gs = context->graph_scheduler();
    const vector<int>& gids = gs->GetJobId();
    VLOG(V_DEBUG) << "Batching jobs of this round: " << gids.size();

    const vector<int>& output_tensor_ids = gs->ScatterTensorIds(gids, child_offset_);
    checkCudaError(cudaMemcpy(gs->gpu_idx_buf(), output_tensor_ids.data(),
                   output_tensor_ids.size()*sizeof(int), cudaMemcpyHostToDevice));
    int blocksPerGrid = gids.size();
    int threadsPerBlock = stride;
    BatchedDynamicSelectedOutputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
            out->mutable_data<T>(), stride, gs->gpu_idx_buf(), inp.data<T>(), stride, stride);

    /*out->SetOffsetWithId(0);*/
    /*for (int local_id = 0; local_id < gids.size(); local_id++) {*/
      /*int gid = gids[local_id];*/
      /*if (gs->ScatterTensorIds({gid}, child_offset_).size() > 0) {*/
        /*CHECK(gs->ScatterTensorIds({gid}, child_offset_).size() == 1);*/
        /*int output_tensor_id = gs->ScatterTensorIds({gid}, child_offset_)[0];*/
        /*checkCudaError(cudaMemcpy(out->mutable_data<T>() + output_tensor_id*stride,*/
                                  /*inp.data<T>() + local_id*stride,*/
                                  /*stride*sizeof(T),*/
                                  /*cudaMemcpyDeviceToDevice));*/
        /*VLOG(V_DEBUG) << "internal scatter id: " << output_tensor_id;*/
      /*}else {*/
        /*VLOG(V_DEBUG) << "[Scattering] No need to scatter";*/
      /*}*/
    /*}*/

    out->DebugNumerical<T>();
  }

 private:
  int child_offset_;
  cudaStream_t stream_;
};

template <typename T>
class GraphPushOp : public OpImpl {
 public:
  explicit GraphPushOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //LOG(FATAL) << "Push Operator needs further runtime support";
    GraphSchedulerBase* gs = context->graph_scheduler();
    CHECK_NOTNULL(gs);
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    //CHECK(out->count() >= inp.count())
    VLOG(V_DEBUG)      << "Input count:\t" << inp.count()
          << "\t" << inp.debug_size() << "Bytes\n"
          << "Output count:\t" << out->count() 
          << "\t" << out->debug_size() << "Bytes";

    T* out_ptr = out->mutable_data<T>();
    CHECK(!out->IsFullShape());
    checkCudaError(cudaMemcpy(out_ptr,
                              inp.data<T>(),
                              inp.count()*sizeof(T),
                              cudaMemcpyDeviceToDevice));
    gs->SetFuncRet(*out);

    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }
};

template <typename T>
class GraphPullOp : public OpImpl {
 public:
  explicit GraphPullOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //LOG(FATAL) << "Pull Operator needs further runtime support";
    GraphSchedulerBase* gs = context->graph_scheduler();
    CHECK_NOTNULL(gs);
    const Tensor& inp = gs->GetFuncArg();
    Tensor* out = context->Output(0);
    CHECK(inp.count() >= out->count())
          << "Input count:\t" << inp.count()
          << "\t" << inp.debug_size() << "Bytes\n"
          << "Output count:\t" << out->count() 
          << "\t" << out->debug_size() << "Bytes";

    //out tensor must be local
    //if in tensor is a global tensor(in the backward of pull)
    //CHECK(inp.IsFullShape());
    const vector<int>& job_ids = gs->GetJobId();
    context->SetDynDim(job_ids.size());
    context->ScaleOutputTensor();
    int stride = out->count()/out->dims(0);
    CHECK(out->dims(0) == job_ids.size());
    VLOG(V_DEBUG) << out->debug_info() << "\t" << out->debug_size();
    VLOG(V_DEBUG) << inp.debug_info() << "\t" << inp.debug_size();
    for (int local_id = 0; local_id < job_ids.size(); local_id++) {
      int gid = job_ids[local_id];
      VLOG(V_DEBUG) << "job_ids[" << local_id << "] = " << gid;
      //const T* inp_ptr = inp.data<T>() + out->count()*gs->GetJobId();
      checkCudaError(cudaMemcpy(out->mutable_data<T>() + local_id*stride,
                                inp.data<T>() + gid*stride,
                                stride*sizeof(T),
                                cudaMemcpyDeviceToDevice));
    }
    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }
};

template <typename T>
class FunctionPushArgOp : public OpImpl {
 public:
  explicit FunctionPushArgOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //LOG(FATAL) << "here";
    const Tensor& inp = context->Input(0);
    GraphSchedulerBase* gs = context->graph_scheduler();
    CHECK_NOTNULL(gs);
    gs->SetFuncArg(inp);
    inp.DebugNumerical<T>();
  }
};

template <typename T>
class FunctionPopRetOp : public OpImpl {
 public:
  explicit FunctionPopRetOp(const OpDef& def) : OpImpl(def) {}
  void Compute(OpContext* context) override {
    //LOG(FATAL) << "here";
    GraphSchedulerBase* gs = context->graph_scheduler();
    CHECK_NOTNULL(gs);
    const Tensor& inp = gs->GetFuncRet();
    Tensor* out = context->Output(0);
    VLOG(V_DEBUG) << inp.debug_info();
    VLOG(V_DEBUG) << out->debug_info();
    CHECK(inp.count() <= out->count())
      << inp.count() << "\t" << out->count();
    CHECK(inp.debug_size() >= out->debug_size())
        << inp.debug_size() << "\t" << out->debug_size();
    VLOG(V_DEBUG) << inp.debug_info();
    VLOG(V_DEBUG) << out->debug_info();
    if (inp.IsDynamicShape()) {
      //CHECK(!out->IsDynamicShape() || out->IsFullShape());
      int stride = inp.count()/inp.dims(0);
      //CHECK(inp.count()/inp.dims(0) == stride);
      for (int tid = 0; tid < out->dims(0); tid++) {
        int gid = gs->InternalTensorIdToJobId(tid);
        VLOG(V_DEBUG) << "tid: " << tid;
        VLOG(V_DEBUG) << "gid: " << gid;
        VLOG(V_DEBUG) << "stride: " << stride;
        checkCudaError(cudaMemcpy(out->mutable_data<T>()+gid*stride,
                                  inp.data<T>()+tid*stride,
                                  stride*sizeof(T),
                                  cudaMemcpyDeviceToDevice));
      }
    }else {
      checkCudaError(cudaMemcpy(out->mutable_data<T>(),
                                inp.data<T>(),
                                out->count()*sizeof(T),
                                cudaMemcpyDeviceToDevice));
    }
    out->DebugNumerical<T>();
  }
};

REGISTER_OP_IMPL_BUILDER(Key("Pull").Device("GPU"),    GraphPullOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Push").Device("GPU"),    GraphPushOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Scatter").Device("GPU"), GraphScatterOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Gather").Device("GPU"),  GraphGatherOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("FunctionPushArg").Device("GPU"), FunctionPushArgOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("FunctionPopRet").Device("GPU"), FunctionPopRetOp<float>);

} //namespace backend