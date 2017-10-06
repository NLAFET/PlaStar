m = 10;
n =  m;

nb = 2;

A = reshape(1:m*n, [m, n])

[L, U, P] = lut(A, nb);