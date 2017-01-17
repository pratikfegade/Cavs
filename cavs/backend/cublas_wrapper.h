#ifndef CAVS_BACKEND_CUBLAS_WRAPPER_H_
#define CAVS_BACKEND_CUBLAS_WRAPPER_H_

namespace backend {

//cublas only supports FORTRAN order, fuck
//so we have to wrap a new interface

//level3
template <typename T>
void MatMulMatCublasWrapper(
    const bool TransA, const bool TransB, 
    const int M, const int N, const int K, 
    const T alpha, const T* A, const T* B,
    const T beta, T* C);

//level2
template <typename T>
void MatMulVecCublasWrapper(
    const bool TransA,
    const int M, const int N,
    const T alpha, const T* A, const T* B,
    const T beta, T* C);

//level1
template <typename T>
void AxpyCublasWrapper(
    const int N, const T alpha,
    const T* x, T* y);

} //namespace backend

#endif