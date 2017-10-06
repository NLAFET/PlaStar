function [x] = solv(A, P, b)

m = rows(A)
n = columns(A)

% fill vector x with pivoted values of b
x = b(P)

% forward substitution Ld = b
for i = 1:m
  for j = 1:i

    x(i) -= A(i,j) * x(j);

  endfor
endfor

x

% back substitution Ux = d
for i = m:-1:1
  for j = i:n

    x(i) -= A(i,j) * x(j);

  endfor
endfor

x

% normalise, divide x by pivots
x ./= diag(A)';

x

endfunction
