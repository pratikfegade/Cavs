#include "cavs/backend/op_impl.h"
#include "cavs/backend/cuda_common.h"
#include "cavs/backend/cublas_wrapper.h"
#include "cavs/midend/devices.h"
#include "cavs/proto/tensor_shape.pb.h"
#include "cavs/util/macros_gpu.h"

namespace backend {

using ::midend::Tensor;

template <typename T>
class AxpyOpCublas : public OpImpl {
 public:
  explicit AxpyOpCublas(const OpDef& def)
    : OpImpl(def) {}
  void Compute(OpContext* context) override;

 private:
};

template <typename T>
void AxpyOpCublas<T>::Compute(OpContext* context) {
  const Tensor& x = context->Input(0);
  const Tensor& alpha = context->Input(1);
  Tensor* y = context->Output(0);

  int N = x.dims(0);
  CHECK(alpha.dims() == 1);
  T a = *(alpha.data<T>());

  AxpyCublasWrapper<T>(
      N, 1.f, x.data<T>(), y->mutable_data<T>());
}

template <typename T>
class ScalOpCublas : public OpImpl {
 public:
  explicit ScalOpCublas(const OpDef& def)
    : OpImpl(def) {
    alpha = GetSingleArg<T>("alpha", 1.f);
  }
  void Compute(OpContext* context) override;

 private:
  T alpha;
};

//REGISTER_OP_IMPL_BUILDER(Key("Scal").Device("GPU"), CudaUnaryOpInstance(math::Abs, float));

} //namespace backend
