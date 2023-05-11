#include <iostream>
#include <stdio.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

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

__global__ void integrate_kernel_cg(
  double *I_n, double *coeff, int n,
  int bound0, int bound1, int bound2,
  int i0, int i1, int i2,
  int step0, int step1, int step2,
  int str0, int str1, int str2,
  int istr0, int istr1, int istr2,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3
)
{
  auto g = cg::this_grid();

  for(int i = 0; i < bound0; i++) {
    int b0 = i0 + (i*step0);
    int b0off = b0*str0;
    int b0inu = i*istr0;

    for(int j = blockIdx.x; j < bound1; j+=gridDim.x) {
      int b1 = i1 + (j*step1);
      int b1off = b0off + b1*str1;
      int b1inu = b0inu + j*istr1;


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
    g.sync();
  }
}

/*__global__ void integrate_kernel_cg(
  double *I_n, double *coeff, int nx, int ny, int nz,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3
)
{
  auto g = cg::this_grid();

  const int n = nx*ny*nz;

  for(int y = 1; y < ny; y++) {

    for(int x = blockIdx.x + 1; x < nx; x += gridDim.x) {
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

    g.sync();
  }

}*/

extern "C" void integrate_cuda_cg(
/*
  double *I_n, double *coeff, int nx, int ny, int nz,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3
*/

  double *I_n, double *coeff, int n,
  int bound0, int bound1, int bound2,
  int i0, int i1, int i2,
  int step0, int step1, int step2,
  int str0, int str1, int str2,
  int istr0, int istr1, int istr2,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3


)
{
  int BLOCKS, THREADS;
  CHECK_CUDA_ERROR(cudaOccupancyMaxPotentialBlockSize(&BLOCKS, &THREADS, integrate_kernel_cg, 0, 0));

  //void *args[] = {&I_n, &coeff, &nx, &ny, &nz, &c0, &c1, &c2, &c3, &off0, &off1, &off2, &off3};
  void *args[] = {&I_n, &coeff, &n, 
                  &bound0, &bound1, &bound2,
                  &i0, &i1, &i2,
                  &step0, &step1, &step2,
                  &str0, &str1, &str2,
                  &istr0, &istr1, &istr2,
                  &c0, &c1, &c2, &c3,
                  &off0, &off1, &off2, &off3};
  CHECK_CUDA_ERROR(cudaLaunchCooperativeKernel((void*)integrate_kernel_cg, BLOCKS, 128, args));
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

