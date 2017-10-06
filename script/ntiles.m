function [m, n, p, q] = ntiles(A, nb)

% Detects number of rows and columns,
% and number of row and column tiles of size 'nb' in matrix A.

[m, n] = size(A);

p = ceil(m/nb);
q = ceil(n/nb);

end