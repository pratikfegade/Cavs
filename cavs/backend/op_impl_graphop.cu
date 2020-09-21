#include "cavs/backend/op_impl.h"
#include "cavs/backend/functor_batched_memcpy.cuh"
#include "cavs/midend/graph_scheduler.h"
#include "cavs/midend/tensor.h"
#include "cavs/util/macros_gpu.h"
#include "cavs/util/op_util.h"

#include <iostream>
#include <string>

using ::midend::Tensor;
using ::midend::GraphSchedulerBase;
using std::vector;
using std::string;

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
    if (VLOG_IS_ON(V_DEBUG)) {
      string out;
      for (int id : gids) out += std::to_string(id) + "\t";
      VLOG(V_DEBUG) << out;
    }

    const vector<int>& tensor_ids_for_gather = gs->CurrentRoundTensorIdsForGather(child_offset_);
    if (VLOG_IS_ON(V_DEBUG)) {
      string out;
      for (int id : tensor_ids_for_gather) out += std::to_string(id) + "\t";
      VLOG(V_DEBUG) << out;
    }
    //for skewed batched trees, in the backward pass,
    //the root of one tree does not need to gather,
    //but the inode of other tree have to gather
    //so we loose this constraint
    /*CHECK(gids.size() == tensor_ids_for_gather.size() || tensor_ids_for_gather.empty());*/
    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Unary] Assign new stream with ID " << context->GetStreamID();
    }

    if (!tensor_ids_for_gather.empty()) {
      checkCudaError(cudaMemcpyAsync(gs->gpu_idx_buf(), tensor_ids_for_gather.data(),
                     tensor_ids_for_gather.size()*sizeof(int), cudaMemcpyHostToDevice, stream_));
      /*int blocksPerGrid = gids.size();*/
      int blocksPerGrid = tensor_ids_for_gather.size();
      /*int threadsPerBlock = stride;*/
      const int MAX_THREADS_IN_BLOCK = 1 << 10;
      int threadsPerBlock = (MAX_THREADS_IN_BLOCK > stride)? stride : MAX_THREADS_IN_BLOCK;
      BatchedDynamicSelectedInputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
              out->mutable_data<T>(), stride, inp.data<T>(), stride, gs->gpu_idx_buf(), stride);
    }else {
      /*checkCudaError(cudaMemset(out->mutable_data<T>(), 0, gids.size()*stride*sizeof(T)));*/
      int blocksPerGrid = gs->CurrentRoundTensorIdsForGatherInitialization().size();
      checkCudaError(cudaMemcpyAsync(gs->gpu_idx_buf(), gs->CurrentRoundTensorIdsForGatherInitialization().data(),
                     blocksPerGrid*sizeof(int), cudaMemcpyHostToDevice, stream_));
      const int MAX_THREADS_IN_BLOCK = 1 << 10;
      int threadsPerBlock = (MAX_THREADS_IN_BLOCK > stride)? stride : MAX_THREADS_IN_BLOCK;
      BatchedDynamicSelectedAssignZeroKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
              out->mutable_data<T>(), stride, gs->gpu_idx_buf(), stride);
    }
    checkCudaError(cudaGetLastError());

    out->DebugNumerical<T>();

    // {
    //   std::cout << "[GATHERED] " << std::endl;
    //   vector<float> res(out->count());
    //   if (out->device_type() == GPU) {
    // 	checkCudaError(cudaMemcpy(res.data(), out->data<float>(),
    // 				  out->count()*sizeof(float), cudaMemcpyDeviceToHost));
    //   } else {
    // 	checkCudaError(cudaMemcpy(res.data(), out->data<float>(),
    // 				  out->count()*sizeof(float), cudaMemcpyHostToHost));
    //   }
    //   std::cout << "[PULL_OP] Gathered " << out->count() << std::endl;
    //   for (int i = 0; i < out->count(); ++i) {
    // 	std::cout << res[i] << " ";
    //   }
    //   std::cout << std::endl;
    // }
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
    if (VLOG_IS_ON(V_DEBUG)) {
      string out;
      for (int id : gids) out += std::to_string(id) + "\t";
      VLOG(V_DEBUG) << out;
    }

    const vector<int>& tensor_ids_for_scatter = gs->CurrentRoundTensorIdsForScatter(child_offset_);
    VLOG(V_DEBUG) << "tensor ids for scatter: " << tensor_ids_for_scatter.size();
    if (VLOG_IS_ON(V_DEBUG)) {
      string out;
      for (int id : tensor_ids_for_scatter) out += std::to_string(id) + "\t";
      VLOG(V_DEBUG) << out;
    }
    //for skewed batched trees, the root of one tree does not need to scatter,
    //but the inode of other tree have to scatter
    //so we loose this constraint
    /*CHECK(gids.size() == tensor_ids_for_scatter.size() || tensor_ids_for_scatter.empty());*/
    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Unary] Assign new stream with ID " << context->GetStreamID();
    }
    if (!tensor_ids_for_scatter.empty()) {
      checkCudaError(cudaMemcpyAsync(gs->gpu_idx_buf(), tensor_ids_for_scatter.data(),
                     tensor_ids_for_scatter.size()*sizeof(int), cudaMemcpyHostToDevice, stream_));
      int blocksPerGrid = tensor_ids_for_scatter.size();
      /*int threadsPerBlock = stride;*/
      const int MAX_THREADS_IN_BLOCK = 1 << 10;
      int threadsPerBlock = (MAX_THREADS_IN_BLOCK > stride)? stride : MAX_THREADS_IN_BLOCK;
      BatchedDynamicSelectedOutputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
              out->mutable_data<T>(), stride, gs->gpu_idx_buf(), inp.data<T>(), stride, stride);
    }

    checkCudaError(cudaGetLastError());
    out->DebugNumerical<T>();
  }

 private:
  int child_offset_;
  cudaStream_t stream_;
};

template <typename T>
class GraphPushOp : public OpImpl {
 public:
  explicit GraphPushOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {}
  void Compute(OpContext* context) override {
    //LOG(FATAL) << "Push Operator needs further runtime support";
    GraphSchedulerBase* gs = context->graph_scheduler();
    CHECK_NOTNULL(gs);
    const Tensor& inp = context->Input(0);
    Tensor* out = context->Output(0);
    //CHECK(out->count() >= inp.count())
    VLOG(V_DEBUG) << "Input count:\t" << inp.count()
                  << "\t" << inp.debug_size() << "Bytes\n"
                  << "Output count:\t" << out->count()
                  << "\t" << out->debug_size() << "Bytes";

    T* out_ptr = out->mutable_data<T>();
    CHECK(!out->IsFullShape());

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Unary] Assign new stream with ID " << context->GetStreamID();
    }
    /*LOG(FATAL) << context->GetStreamID();*/
    checkCudaError(cudaMemcpyAsync(out_ptr, inp.data<T>(),
                                   inp.count()*sizeof(T),
                                   cudaMemcpyDeviceToDevice, stream_));
    /*ContinuousMemcpyKernel<<<BLOCKS_PER_GRID(inp.count()), THREADS_PER_BLOCK, 0, stream_>>>(*/
        /*out_ptr, inp.data<T>(), inp.count());*/
    gs->SetFuncRet(*out);

    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }

 private:
  cudaStream_t stream_;
};

template <typename T>
class GraphPullOp : public OpImpl {
 public:
  explicit GraphPullOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault)  {}

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
    const vector<int>& gids = gs->GetJobId();

    // {
    //   std::cout << "[PULL_OP] Pulling for gids " << std::endl;
    //   for (auto id: gids) {
    // 	std::cout << id << " " << std::endl;
    //   }
    //   std::cout << std::endl;
    //   std::cout << "Input count:\t" << inp.count()
    // 	     << "\t" << inp.debug_size() << "Bytes\n"
    // 	     << "Output count:\t" << out->count()
    // 	     << "\t" << out->debug_size() << "Bytes" << std::endl;
    // }




    context->SetDynDim(gids.size());
    context->ScaleOutputTensor();
    int stride = out->count()/out->dims(0);
    CHECK(out->dims(0) == gids.size());
    /*VLOG(V_DEBUG) << out->debug_info() << "\t" << out->debug_size();*/
    /*VLOG(V_DEBUG) << inp.debug_info() << "\t" << inp.debug_size();*/

    if (!stream_ && context->GetStreamID() != -1) {
      stream_ = StreamEventHandlePool::GetCudaStream(context->GetStreamID());
      VLOG(V_DEBUG) << "[Unary] Assign new stream with ID " << context->GetStreamID();
    }
    checkCudaError(cudaMemcpyAsync(gs->gpu_idx_buf(), gids.data(),
                   gids.size()*sizeof(int), cudaMemcpyHostToDevice, stream_));
    int blocksPerGrid = gids.size();
    /*int threadsPerBlock = stride;*/
    const int MAX_THREADS_IN_BLOCK = 1 << 10;
    int threadsPerBlock = (MAX_THREADS_IN_BLOCK > stride)? stride : MAX_THREADS_IN_BLOCK;
    checkCudaError(cudaGetLastError());
    BatchedDynamicSelectedInputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
            out->mutable_data<T>(), stride, inp.data<T>(), stride, gs->gpu_idx_buf(), stride);
    checkCudaError(cudaGetLastError());

    // {
    //   vector<float> res(inp.count());
    //   if (inp.device_type() == GPU) {
    // 	checkCudaError(cudaMemcpy(res.data(), inp.data<float>(),
    // 				  inp.count()*sizeof(float), cudaMemcpyDeviceToHost));
    //   } else {
    // 	checkCudaError(cudaMemcpy(res.data(), inp.data<float>(),
    // 				  inp.count()*sizeof(float), cudaMemcpyHostToHost));
    //   }
    //   std::cout << "[PULL_OP] Pulled for gids " << std::endl;
    //   for (auto id: gids) {
    // 	std::cout << id << " " << res[id] << " " << std::endl;
    //   }
    // }

    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }

 private:
  cudaStream_t stream_;
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
  explicit FunctionPopRetOp(const OpDef& def) :
    OpImpl(def), stream_(cudaStreamDefault) {}

  void Compute(OpContext* context) override {
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

    CHECK(inp.IsDynamicShape());
    //for the backward, the gradient of lower layer output may not be dynamic
    //for example, the placeholder of layer0
    /*CHECK(out->IsDynamicShape());*/
    int stride = inp.count()/inp.dims(0);
    int out_dyn_dim = out->dims(0);
    //for the backward, the shape of lower layer output is arbitrary
    //for example, the placeholder may be {2, 4} (batch, time_step)
    //here, the inp shape may be {1, 1} (serial model) or {2, 1} (batch mode)
    /*CHECK(stride == out->count()/out_dyn_dim);*/
    const vector<int>& tids2gids= gs->TensorIdsToJobIds();
    for (int i = 0; i < tids2gids.size(); i++) {
      VLOG(V_DEBUG) << "i: " << i << "\tgid: " << tids2gids[i];
    }
    checkCudaError(cudaMemcpyAsync(gs->gpu_idx_buf(), tids2gids.data(),
                   tids2gids.size()*sizeof(int), cudaMemcpyHostToDevice, stream_));
    int blocksPerGrid = tids2gids.size();
    /*int threadsPerBlock = stride;*/
    const int MAX_THREADS_IN_BLOCK = 1 << 10;
    VLOG(V_DEBUG) << blocksPerGrid;
    VLOG(V_DEBUG) << stride;
    checkCudaError(cudaGetLastError());
    int threadsPerBlock = (MAX_THREADS_IN_BLOCK > stride)? stride : MAX_THREADS_IN_BLOCK;
    BatchedDynamicSelectedOutputSliceCopyKernel<T><<<blocksPerGrid, threadsPerBlock, 0, stream_>>>(
            out->mutable_data<T>(), stride, gs->gpu_idx_buf(), inp.data<T>(), stride, stride);
    checkCudaError(cudaGetLastError());

    inp.DebugNumerical<T>();
    out->DebugNumerical<T>();
  }

 private:
  cudaStream_t stream_;
};

REGISTER_OP_IMPL_BUILDER(Key("Pull").Device("GPU"),    GraphPullOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Push").Device("GPU"),    GraphPushOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Scatter").Device("GPU"), GraphScatterOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("Gather").Device("GPU"),  GraphGatherOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("FunctionPushArg").Device("GPU"), FunctionPushArgOp<float>);
REGISTER_OP_IMPL_BUILDER(Key("FunctionPopRet").Device("GPU"), FunctionPopRetOp<float>);

} //namespace backend
