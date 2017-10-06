function [P, L, U, sign] = splu(A)

[m, n] = size(A);

if m ~= n
  error('Matrix must be square!');
end

% init matrices P, L, U
P = eye(n, n);
L = eye(n, n);
U = zeros(n, n);
tol = sqrt(eps);

sign = 1;

% for each column
for k = 1:n

  % perform pivoting, if necessary
  if abs(A(k,k)) < tol

    % for each row of k
    for r = k:n

      % pivot is found
      if abs(A(r,k)) >= tol
        break
      end

      % pivot not found, exit
      if r == n
	    if nargout == 4
          sign = 0;
	      return
        else
	      disp('A is singular within tolerance.')
	      error(['No pivot in column ' int2str(k) '!'])
        end
      end

      % swap rows of A, L, P
      A([r k], 1:n) = A([k r], 1:n);
      if k > 1, L([r k], 1:n) = L([k r], 1:n); end
      P([r k], 1:n) = P([k r], 1:n);
      sign = -sign;
    end
  end

  % fill in value of L for each row below k
  L(k+1:n,k) = A(k+1:n,k)/A(k,k);

  % update trailing matrix, each row below k, each column beyond k
  A(k+1:n,k+1:n) = A(k+1:n,k+1:n) - L(k+1:n,k)*A(k,k+1:n);

  % fill in row of U, each column incl. k
  U(k,k:n) = A(k,k:n) .* (abs(A(k,k:n)) >= tol);
end

if nargout < 4
  roworder = P*(1:n)';
  disp('Pivots in rows:'), disp(roworder');
end

end
