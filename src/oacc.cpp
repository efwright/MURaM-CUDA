#include <openacc.h>

//void integrate_openacc(
//  double *I_n, double *coeff,
//  const int nx, const int ny, const int nz,
//  const double c0, const double c1, const double c2, const double c3,
//  const int off0, const int off1, const int off2, const int off3
//)
void integrate_openacc(
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
  for(int i = 0; i < bound0; i++) {
    int b0 = i0 + (i*step0);
    int b0off = b0*str0;
    int b0inu = i*istr0;

    //int yoff = y*nx*nz;
    //int ycoeff = (y-1)*(nx-1)*(nz-1);
    #pragma acc parallel loop gang async \
     deviceptr(I_n, coeff) default(present)
    for(int j = 0; j < bound1; j++) {
      int b1 = i1 + (j*step1);
      int b1off = b0off + b1*str1;
      int b1inu = b0inu + j*istr1;


      //int xoff = yoff + x*nz;
      //int xcoeff = ycoeff + (x-1)*(nz-1);
      #pragma acc loop vector
      for(int k = 0; k < bound2; k++) {
        int b2 = i2 + (k*step2);
        int ind = b1off + b2*str2;
        int i_nu = b1inu + k*istr2;


        //int ind = xoff + z;
        //int i_nu = xcoeff + z-1;



        double I_upw = c0*I_n[ind-off0] +
                       c1*I_n[ind-off1] +
                       c2*I_n[ind-off2] +
                       c3*I_n[ind-off3];
        I_n[ind]=I_upw*coeff[i_nu] + coeff[i_nu + n];

      }
    }
  }
  #pragma acc wait

}

