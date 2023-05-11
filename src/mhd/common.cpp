#include <iostream>

#include "common.h"

GridData::GridData(
  int nx, int ny, int nz
){
  lsize[0]=nz; lsize[1]=nx; lsize[2]=ny;
  for(int i=0;i<3;i++) ghosts[i]=2;
  for(int i=0;i<3;i++) lbeg[i]=ghosts[i];
  for(int i=0;i<3;i++) lend[i]=lbeg[i]+lsize[i]+ghosts[i]-1;
  stride[0] = 1;
  stride[1] = lsize[0]+(2*ghosts[0]);
  stride[2] = stride[1]*(lsize[1]+(2*ghosts[i]));
  vsize = max(max(nx, ny), nz);
  nvar = 8;
  bufsize = (lsize[0]+(2*ghosts[0]))*
            (lsize[1]+(2*ghosts[1]))*
            (lsize[2]+(2*ghosts[2]));

  U = new cState[bufsize];
  Res = new cState[bufsize];

  v_amb = new Vector[bufsize];
  R_amb = new Vector[bufsize];

  curlB = new Vector[bufsize];
  curlBxB = new Vector[bufsize];

}

