function [X, res_x]= NNKOMP(A, Kyy, Kzy,Kzz,T0)
% The Non-Negative Kernel-based Orthogonal Matching Pursuit
%
% This code solves the following problem:
%
%   x = argmin_x ||phi(y)-phi(Y) A x||_2^2
%   subject to x >= 0 , ||x||_0 < T0
%
% The detail of the algorithm is described in the following paper:
%   "Non-negative kernel sparse coding for the analysis of motion data.", B. Hosseini, et al.
%
% Written by Babak Hosseini <bhosseini@techfak.uni-bielefeld.de> or <bbkhosseini@gmail.com>
% Copyright 2019 by Babak Hosseini
%-------------------------
%
%
% Input:
%     Kyy: The similarity based kernel matrix of all data Y
%           Kyy=Phi(Y)'*Phi(Y);
%     Kzy: The similarity of sample y to all data samples Y
%           Kzy=Phi(y)'*Phi(Y);
%     A  : Dictionary matrix
%     T0 : cardinality constraint
%
%
% ===========================

e_tol=1;
[~,N] = size(A);
X = sparse(zeros(N,1));
S = []; % positions indexes of components of s
R = 1:size(A,2); % positions indexes of components of s
R(sum(A,1)==0)=[];
res_phi = Kzy*A; % first r*d
x_est_pre=0;
res_x=err_kern(A,X,Kyy,Kzz,Kzy);
t=1;
while (t<=T0)
    [~,j]=min((res_phi(R)-1).^2);
    j=R(j);
    S = [S j];
    R=R(R~=j);
    S=sort(S);
    Ai=A(:,S);
    x_est = knnls(Ai,Kyy,Kzy);
    fx(x_est);
    
    res_x2=err_kern(Ai,x_est,Kyy,Kzz,Kzy);
    
    if (res_x2-res_x) < -res_x*0.0001
        res_x=res_x2;
        t=t+1;
        if sum(x_est==0)
            t=t-sum(x_est==0);
            z=find(x_est==0);
            S(z)=[];
            x_est(z)=[];
            Ai=A(:,S);
        end
    else
        S=S(S~=j);
        Ai=A(:,S);
        if isempty(Ai)
            Ai=zeros(size(Ai,1),1);
        end
        x_est=x_est_pre;
    end
    
    vs = Ai*x_est;
    res_phi=(Kzy-(vs)'*Kyy)*A; %r'*D
    
    x_est_pre=x_est;
    
    if norm(res_x) < e_tol  || isempty(R)
        break
    end
end
X(S)=x_est;
X((X/max(X))<1e-2)=0;

% ===============================
function [MSE,MSE_abs] = err_kern(A,X,Kyy,k_zz,k_zy)

MSE_abs=trace(abs(-2*k_zy*A*X+k_zz+X'*A'*Kyy*A*X));
MSE=MSE_abs / trace(k_zz)*100;