#include <iostream>
#include <omp.h>

extern void mhd_openacc();

int main()
{
  std::cout << "Running OpenACC test...\n";
  mhd_openacc();

  std::cout << "MHD Cuda has finished running.\n";
  return 0;
}

