struct Vector
{
  double x, y, z;

  Vector() :x(0.0), y(0.0), z(0.0) { }
  Vector(double _x, double _y, double _z):
    x(_x), y(_y), z(_z) { };
  ~Vector(){}
};

struct cState
{
  double d;
  Vector M;
  double e;
  Vector B;

  cState() :d(0.0), M(Vector()), e(0.0), B(Vector()) { }
  ~cState(){}
}

struct GridData
{
  int lsize[3], ghosts[3], lbeg[3], lend[3], stride[3];
  int vsize, nvar, bufsize;

  GridData(int nx, int ny, int nz)
  {
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
  }
}

