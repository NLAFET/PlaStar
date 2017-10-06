function [L, U, P] = lut(A, nb)

% Tiled version of LU factorization.

[m, n, p, q] = ntiles(A, nb);

for k = 1:min(p,q)
    
    % extract tile kk
    Akk = tile(A,k,k,m,n,nb);
    
    % perform LU factorisation of tile kk (getrf)
    [Lkk, Ukk, Pkk] = lu(Akk);
    
    % pre-calculate product Lkk^(-1) Pkk (gessm)
    LPkk = Lkk^(-1) * Pkk;
    
    % for each column tile kj
    for j = k+1:q
        
        % exctract column tile kj
        Akj = tile(A,k,j,m,n,nb);
        
        % perform multiplications Lkk^(-1) Pkk Akj
        Uk{j} = LPkk * Akj;
        
    end
    
    % for each row tile i
    for i = k+1:p
        
        % extract tile Aik
        Aik = tile(A,i,k,m,n,nb);
        
        % perform LU factorisation of matrix [Ukk; Aik] (tstrf)
        [Lk{i}, Uik, Pk{i}] = lu([Ukk; Aik]);

        % for each column tile j
        %   update trailing submatrix
        for j = k+1:q
            
            % extract tile ij
            Aij = tile(A,i,j,m,n,nb);
            
            % perform multiplications Lik^(-1) Pik [Ukj; Aij]
            B{i,j} = ssssm(Lk{i}, Pk{i}, Uk{j}, Aij);
            
        end
    end
end

end