function Aij = ssssm(Lik, Pik, Ukj, Aij)

% Performs multiplication Lik^(-1) Pik [Ukj; Aij]

Aij = pinv(Lik) * Pik * [Ukj; Aij];

end