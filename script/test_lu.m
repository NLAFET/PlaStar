n  = 3;
% A  = rand(n,n)
A = [1 2 3; 4 5 6; 7 8 9]
A0 = A;

[L0, U0, P0] = lu(A)

% [Q] = lux(A, eps)
[P1, L1, U1, sign] = splu(A)

% b = [10 11 12]
% 
% x = solv(A, P, b);



% Pm = zeros(n,n);
% 
% for i = 1:n
%   Pm(i,P(i)) = 1;
% endfor
% 
% P
% Pm
% 
% PmA0 = Pm*A0
% LU   = L*U

% B = L*U
% C = P*A
