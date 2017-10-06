function [Q] = lux(A, tol)

[m, n] = size(A);

% permutation vector
Q = 1:n

% for each column
for j = 1:n

  % find maximum of |Aj| and its index
  [maxv, maxi] = max(abs(A(j:m,j)));

  % failure: matrix is degenerate
  if (maxv < tol)
    return;
  end

  % perform pivoting if necessary
  if (maxi ~= j)

    % swap values of permutation vector
    Q([maxi j]) = Q([j maxi])

    % swap rows of matrix A
    A([maxi j],:) = A([j maxi],:)

  end

%   % for each consecutive row
%   for i = (j+1):m
% 
%     % printf("normalising %d,%d with %d,%d:\n", i,j, j,j);
% 
%     % normalise
%     A(i,j) /= A(j,j);
% 
%     % for each consecutive column
%     for k = (j+1):n
% 
%       printf("elliminating %d,%d with %d,%d * %d,%d:\n", i,k, i,j, j,k);
% 
%       % elliminate
%       A(i,k) -= A(i,j) * A(j,k)
% 
%     endfor
% 
%   endfor

end

end
