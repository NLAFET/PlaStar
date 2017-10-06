function Aij = tile(A, i, j, m, n, nb)

% Extracts tile i,j of size 'nb' from matrix A

i0 = (i-1)*nb+1;
j0 = (j-1)*nb+1;

i1 = i*nb; if i1 > m, i1 = m; end
j1 = j*nb; if j1 > n, j1 = n; end

Aij = A(i0:i1, j0:j1);

end