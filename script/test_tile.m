m = 10;
n =  m;

nb = 3;

i  = 2;
j  = 2;

A = reshape(1:m*n, [m,n])

Aij = tile(A,i,j,m,n,nb)