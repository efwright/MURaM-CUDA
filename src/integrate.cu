#include <iostream>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)

namespace{
template <typename T>
void check(T err, const char* const func, const char* const file, const int line)
{
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
              << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
  }
}

void checkLast(const char* const file, const int line)
{
  cudaError_t err{cudaGetLastError()};
  if (err != cudaSuccess)
  {
    std::cerr << "CUDA Runtime Error at: " << file << ":" << line
              << std::endl;
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
}
}; // End anon namespace

/*__global__ void integrate_kernel(
  double *I_n, double *coeff, int nx, int ny, int nz,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3,
  int y
)
{
  const int n = nx*ny*nz;

  int x = blockIdx.x + 1;
  if(x < nx) {
    for(int z = threadIdx.x + 1; z < nz; z += blockDim.x) {

      int ind = y*nx*nz + x*nz + z;
      int i_nu = (y-1)*(nx-1)*(nz-1) + (x-1)*(nz-1) + z-1;

      double I_upw = c0*I_n[ind-off0] +
                     c1*I_n[ind-off1] +
                     c2*I_n[ind-off2] +
                     c3*I_n[ind-off3];

      I_n[ind] = I_upw*coeff[i_nu] + coeff[i_nu+n];
    }
  }
*/
__global__ void integrate_kernel(
  double *I_n, double *coeff, const int n,
  const int bound0, const int bound1, const int bound2,
  const int i0, const int i1, const int i2,
  const int step0, const int step1, const int step2,
  const int str0, const int str1, const int str2,
  const int istr0, const int istr1, const int istr2,
  const double c0, const double c1, const double c2, const double c3,
  const int off0, const int off1, const int off2, const int off3,
  int i
)
{
  int b0 = i0 + (i*step0);
  //int b0off = b0*str0;
  //int b0inu = i*istr0;

  int b1 = i1 + (blockIdx.x*step1);
  int b1off = b0*str0 + b1*str1;
  int b1inu = i*istr0 + blockIdx.x*istr1;


  for(int k = threadIdx.x; k < bound2; k+=blockDim.x) {
    int b2 = i2 + (k*step2);
    int ind = b1off + b2*str2;
    int i_nu = b1inu + k*istr2;

    double I_upw = c0*I_n[ind-off0] +
                   c1*I_n[ind-off1] +
                   c2*I_n[ind-off2] +
                   c3*I_n[ind-off3];
    I_n[ind]=I_upw*coeff[i_nu] + coeff[i_nu + n];
  }
}

extern "C" void integrate_cuda(
  double *I_n, double *coeff, const int n,
  const int bound0, const int bound1, const int bound2,
  const int i0, const int i1, const int i2,
  const int step0, const int step1, const int step2,
  const int str0, const int str1, const int str2,
  const int istr0, const int istr1, const int istr2,
  const double c0, const double c1, const double c2, const double c3,
  const int off0, const int off1, const int off2, const int off3
)
{
  int blocks = bound1;
  int threads = 128;

  for(int i = 0; i < bound0; i++) {
    integrate_kernel<<<blocks, threads>>>(
      I_n, coeff, n, 
      bound0, bound1, bound2,
      i0, i1, i2,
      step0, step1, step2,
      str0, str1, str2,
      istr0, istr1, istr2,
      c0, c1, c2, c3,
      off0, off1, off2, off3,
      i
    );
  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

