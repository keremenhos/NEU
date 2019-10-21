% Function used in question 1 - Used from googleDrive folder of the course
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*((Sigma)\(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
