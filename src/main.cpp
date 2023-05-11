#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <omp.h>

//extern void integrate_openacc(
//  double *I_n, double *coeff,
//  const int nx, const int ny, const int nz,
//  const double c0, const double c1, const double c2, const double c3,
//  const int off0, const int off1, const int off2, const int off3
//);

extern void integrate_openacc(
  double *I_n, double *coeff, const int n,
  const int bound0, const int bound1, const int bound2,
  const int i0, const int i1, const int i2,
  const int step0, const int step1, const int step2,
  const int str0, const int str1, const int str2,
  const int istr0, const int istr1, const int istr2,
  const double c0, const double c1, const double c2, const double c3,
  const int off0, const int off1, const int off2, const int off3
);

extern "C" void integrate_cuda(
  double *I_n, double *coeff, const int n,
  const int bound0, const int bound1, const int bound2,
  const int i0, const int i1, const int i2,
  const int step0, const int step1, const int step2,
  const int str0, const int str1, const int str2,
  const int istr0, const int istr1, const int istr2,
  const double c0, const double c1, const double c2, const double c3,
  const int off0, const int off1, const int off2, const int off3
);

extern "C" void integrate_cuda_cg(
  double *I_n, double *coeff, int n,
  int bound0, int bound1, int bound2,
  int i0, int i1, int i2,
  int step0, int step1, int step2,
  int str0, int str1, int str2,
  int istr0, int istr1, int istr2,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3
);

extern "C" void integrate_cuda_graphs(
  double *I_n, double *coeff, int n,
  int bound0, int bound1, int bound2,
  int i0, int i1, int i2,
  int step0, int step1, int step2,
  int str0, int str1, int str2,
  int istr0, int istr1, int istr2,
  double c0, double c1, double c2, double c3,
  int off0, int off1, int off2, int off3
);


#define OFF(y, x, z) \
  (y)*nx*nz + (x)*nz + (z)

// Serial reference kernel
void integrate(
  double *I_n, double *coeff,
  const int nx, const int ny, const int nz,
  const double c0, const double c1, const double c2, const double c3,
  const int off0, const int off1, const int off2, const int off3
)
{
  const int n = nx*ny*nz;

  for(int y = 1; y < ny; y++) {
    int yoff = y*nx*nz;
    int ycoeff = (y-1)*(nx-1)*(nz-1);
    for(int x = 1; x < nx; x++) {
      int xoff = yoff + x*nz;
      int xcoeff = ycoeff + (x-1)*(nz-1);
      for(int z = 1; z < nz; z++) {
        int ind = xoff + z;
        int i_nu = xcoeff + z-1;

        double I_upw = c0*I_n[ind-off0] +
                       c1*I_n[ind-off1] +
                       c2*I_n[ind-off2] +
                       c3*I_n[ind-off3];
        I_n[ind]=I_upw*coeff[i_nu] + coeff[i_nu + n];
      }
    }
  }
}


int main()
{

  const int nx = 256;
  const int ny = 256;
  const int nz = 256;

  const double c[] = {0.0, 1.0, 2.0, 3.0};
  const int off[] = {OFF(1, 0, 0),
                     OFF(1, 1, 0),
                     OFF(1, 0, 1),
                     OFF(1, 1, 1)};

  double *coeff = new double[nx*ny*nz*2];
  srand(0);
  for(int i = 0; i < nx*ny*nz*2; i++)
    coeff[i] = ((double) (rand()%10))/10.0;

  double *In_ref = new double[nx*ny*nz];
  double *In_acc = new double[nx*ny*nz];
  double *In_cuda = new double[nx*ny*nz];
  double *In_cuda_cg = new double[nx*ny*nz];
  double *In_cuda_graphs = new double[nx*ny*nz];
  for(int i = 0; i < nx*ny*nz; i++) {
    double val = ((double) (rand()%10))/10.0;
    In_ref[i] = val;
    In_acc[i] = val;
    In_cuda[i] = val;
    In_cuda_cg[i] = val;
    In_cuda_graphs[i] = val;
  }

  integrate(In_ref, coeff, nx, ny, nz,
            c[0], c[1], c[2], c[3],
            off[0], off[1], off[2], off[3]);
  std::cout << "CPU run finished\n";

  int bound[] = {ny-1, nx-1, nz-1};
  int str[] = {nx*nz, nz, 1};
  int istr[] = {(nx-1)*(nz-1), nz-1, 1};
  int ini[] = {1, 1, 1};
  int step[] = {1, 1, 1};

#pragma acc data copy(In_acc[:nx*ny*nz], In_cuda[:nx*ny*nz], In_cuda_cg[:nx*ny*nz], In_cuda_graphs[:nx*ny*nz]) copyin(coeff[:nx*ny*nz*2])
{
#pragma acc host_data use_device(In_acc, In_cuda, In_cuda_cg, In_cuda_graphs, coeff)
{
  integrate_openacc(
    In_acc, coeff, nx*ny*nz,
    bound[0], bound[1], bound[2],
    ini[0], ini[1], ini[2],
    step[0], step[1], step[2],
    str[0], str[1], str[2],
    istr[0], istr[1], istr[2],
    c[0], c[1], c[2], c[3],
    off[0], off[1], off[2], off[3]
  );
  std::cout << "OpenACC run finished\n";

  integrate_cuda(
    In_cuda, coeff, nx*ny*nz,
    bound[0], bound[1], bound[2],
    ini[0], ini[1], ini[2],
    step[0], step[1], step[2],
    str[0], str[1], str[2],
    istr[0], istr[1], istr[2],
    c[0], c[1], c[2], c[3],
    off[0], off[1], off[2], off[3]
  );
  std::cout << "CUDA run finished\n";

  integrate_cuda_cg(
    In_cuda_cg, coeff, nx*ny*nz,
    bound[0], bound[1], bound[2],
    ini[0], ini[1], ini[2],
    step[0], step[1], step[2],
    str[0], str[1], str[2],
    istr[0], istr[1], istr[2],
    c[0], c[1], c[2], c[3],
    off[0], off[1], off[2], off[3]
  );
  std::cout << "CUDA CG run finished\n";

  integrate_cuda_graphs(
    In_cuda_graphs, coeff, nx*ny*nz,
    bound[0], bound[1], bound[2],
    ini[0], ini[1], ini[2],
    step[0], step[1], step[2],
    str[0], str[1], str[2],
    istr[0], istr[1], istr[2],
    c[0], c[1], c[2], c[3],
    off[0], off[1], off[2], off[3]
  );
  std::cout << "CUDA Graphs run finished\n";


  std::cout << "Running performance analysis\n";
  const int NumRuns = 5;
  std::cout << " Using the average of " << NumRuns << " runs.\n";

  double OpenACCTime = 0.0;
  double CUDATime = 0.0;
  double CUDACGTime = 0.0;
  double CUDAGraphsTime = 0.0;

  std::cout << "  OpenACC ";
  for(int i=0; i<NumRuns; i++) {
    double st, et;
    st = omp_get_wtime();
    integrate_openacc(
      In_acc, coeff, nx*ny*nz,
      bound[0], bound[1], bound[2],
      ini[0], ini[1], ini[2],
      step[0], step[1], step[2],
      str[0], str[1], str[2],
      istr[0], istr[1], istr[2],
      c[0], c[1], c[2], c[3],
      off[0], off[1], off[2], off[3]
    );
    et = omp_get_wtime();
    OpenACCTime += (et-st);
    std::cout << "*";
  }
  std::cout << "\n";

  std::cout << "  CUDA ";
  for(int i=0; i<NumRuns; i++) {
    double st, et;
    st = omp_get_wtime();
    integrate_cuda(
      In_cuda, coeff, nx*ny*nz,
      bound[0], bound[1], bound[2],
      ini[0], ini[1], ini[2],
      step[0], step[1], step[2],
      str[0], str[1], str[2],
      istr[0], istr[1], istr[2],
      c[0], c[1], c[2], c[3],
      off[0], off[1], off[2], off[3]
    );
    et = omp_get_wtime();
    CUDATime += (et-st);
    std::cout << "*";
  }
  std::cout << "\n";

  std::cout << "  CUDA CG ";
  for(int i=0; i<NumRuns; i++) {
    double st, et;
    st = omp_get_wtime();
    integrate_cuda_cg(
      In_cuda_cg, coeff, nx*ny*nz,
      bound[0], bound[1], bound[2],
      ini[0], ini[1], ini[2],
      step[0], step[1], step[2],
      str[0], str[1], str[2],
      istr[0], istr[1], istr[2],
      c[0], c[1], c[2], c[3],
      off[0], off[1], off[2], off[3]
    );
    et = omp_get_wtime();
    CUDACGTime += (et-st);
    std::cout << "*";
  }
  std::cout << "\n";

  std::cout << "  CUDA Graphs ";
  for(int i=0; i<NumRuns; i++) {
    double st, et;
    st = omp_get_wtime();
    integrate_cuda_graphs(
      In_cuda_graphs, coeff, nx*ny*nz,
      bound[0], bound[1], bound[2],
      ini[0], ini[1], ini[2],
      step[0], step[1], step[2],
      str[0], str[1], str[2],
      istr[0], istr[1], istr[2],
      c[0], c[1], c[2], c[3],
      off[0], off[1], off[2], off[3]
    );
    et = omp_get_wtime();
    CUDAGraphsTime += (et-st);
    std::cout << "*";
  }
  std::cout << "\n";

  std::cout << "Performance analysis results:\n";
  std::cout <<   "OpenACC:\t" << (OpenACCTime/((double)NumRuns))*1000.0 << "ms\n";
  std::cout <<   "CUDA:\t\t" << (CUDATime/((double)NumRuns))*1000.0 << "ms\n";
  std::cout <<   "CUDACG:\t\t" << (CUDACGTime/((double)NumRuns))*1000.0 << "ms\n";
  std::cout <<   "CUDAGraphs:\t" << (CUDAGraphsTime/((double)NumRuns))*1000.0 << "ms\n";

} // End use_devices
} // End data

  std::cout << "Finished\n";

  return 0;

}

