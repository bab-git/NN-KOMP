function [x,resnorm,exitflag,output,lambda] = knnls(A,Kyy,Kzy)
%Kernel LSQNONNEG Linear least squares with nonnegativity constraints.
%
%   This code solves the following problem with Cholesky decomposition: 
% 
%   X=argmin_X ||phi(y)-phi(Y) A X||_2^2
%   subject to X >= 0. 
%
% Input:
%     Kyy: The similarity based kernel matrix of all data Y
%           Kyy=Phi(Y)'*Phi(Y);
%     Kzy: The similarity of sample y to all data samples Y
%           Kzy=Phi(y)'*Phi(Y);
%
%
%   [X,RESNORM] = LSQNONNEG(...) also returns the value of the squared 2-norm of
%   the residual: ||phi(y)-phi(Y)AX||_2^2.
%
%   [X,RESNORM,EXITFLAG] = LSQNONNEG(...) returns an EXITFLAG that
%   describes the exit condition of LSQNONNEG. Possible values of EXITFLAG and
%   the corresponding exit conditions are
%
%    1  LSQNONNEG converged with a solution X.
%    0  Iteration count was exceeded. Increasing the tolerance
%       (OPTIONS.TolX) may lead to a solution.
%
%   [X,RESNORM,EXITFLAG,OUTPUT] = LSQNONNEG(...) returns a structure
%   OUTPUT with the number of steps taken in OUTPUT.iterations, the type of
%   algorithm used in OUTPUT.algorithm, and the exit message in OUTPUT.message.
%
%   [X,RESNORM,EXITFLAG,OUTPUT,LAMBDA] = LSQNONNEG(...) returns
%   the dual vector LAMBDA  where LAMBDA(i) <= 0 when X(i) is (approximately) 0
%   and LAMBDA(i) is (approximately) 0 when X(i) > 0.
%
%   This code is resulted by modifying and kernelizing the matlab lsqnonneg
%   algorithm related to:
%   Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.
% 
%
% 
%   Modified and kernelized by Babak Hosseini 
%   <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
% 
%   Related paper:
%   "Non-negative kernel sparse coding for the analysis of motion data.", B. Hosseini, et al. 
%
%================================================================
verbosity = 1;
n = size(A,2);
% Initialize vector of n zeros and Infs (to be used later)
nZeros = zeros(n,1);
wz = nZeros;
tol = 10*eps*norm(Kyy,1)*length(Kyy);
 
% Initialize set of non-active columns to null
P = false(n,1);
% Initialize set of active columns to all and the initial point to zeros
Z = true(n,1);
x = nZeros;

AkA=A'*Kyy*A;
kA=Kzy*A;

w=kA'-AkA*x;

% Set up iteration criterion
outeriter = 0;
iter = 0;
itmax = 3*n;
exitflag = 1;

% Outer loop to put variables into set to hold positive coefficients
i_chol=0;
while any(Z) && any(w(Z) > tol)
    outeriter = outeriter + 1;
    % Reset intermediate solution z
    z = nZeros;
    % Create wz, a Lagrange multiplier vector of variables in the zero set.
    % wz must have the same size as w to preserve the correct indices, so
    % set multipliers to -Inf for variables outside of the zero set.
    wz(P) = -Inf;
    wz(Z) = w(Z);
    % Find variable with largest Lagrange multiplier
    [~,t] = max(wz);
    % Move variable t from zero set to positive set
    P(t) = true;
    Z(t) = false;
    % Compute intermediate solution using only variables in positive set
    if i_chol        
        i_p=find(P-P0);
        i_ph=[i_ph i_p];
        Ak=[Ak;A(:,i_p)'*Kyy];
        temp=Ak*A(:,i_p);        
        v=temp(1:end-1);
        c=temp(end);
        w=L_1\v;
        L=[L_1 zeros(size(L_1,1),1) ; w' sqrt(c-w'*w)];
        kApt=kA(:,i_ph);
        z(i_ph)=(L')\(L\kApt');
        L_1=L;        
    else
        A_1=AkA(P,P);
        L_1=chol(A_1,'lower');
        Ap=A(:,P);
        Ak=Ap'*Kyy;
        i_ph=find(P)';
        z(P)=(A_1)\(Ap'*Kzy');
    end
    i_chol=1;
    P0=P;
    % inner loop to remove elements from the positive set which no longer belong
    while any(z(P) <= 0)
        iter = iter + 1;
        if iter > itmax
            if verbosity
                disp('MATLAB:lsqnonneg:IterationCountExceeded');
            end
            exitflag = 0;
            output.iterations = outeriter;
            output.message = msg;
            output.algorithm = 'active-set';
            x = z;
            x(x<0)=0;
            lambda = w;
            resnorm=1+(A*x)'*Kyy*(A*x)-2*Kzy*(A*x);
            return
        end
        % Find indices where intermediate solution z is approximately negative
        Q = (z <= 0) & P;
        % Choose new x subject to keeping new x nonnegative
        alpha = min(x(Q)./(x(Q) - z(Q)));
        x = x + alpha*(z - x);
        % Reset Z and P given intermediate values of x
        Z = ((abs(x) < tol) & P) | Z;
        P = ~Z;
        z = nZeros;           % Reset z        
        Ap=A(:,P);        
        AkAp=AkA(:,P);AkAp=AkAp(P,:);
        kAp=kA(:,P);
        z(P)=(AkAp)^-1*(kAp');
        i_chol=0;
        
    end
    x = z;
    w=kA'-AkA*x;
    
end

lambda = w;
resnorm=1+(A*x)'*Kyy*(A*x)-2*Kzy*(A*x);
output.iterations = outeriter;
output.algorithm = 'active-set';
msg = 'OptimizationTerminated';
if verbosity > 1
    disp(msg)
end
output.message = msg;
