%========= sample implementation of nn-komp algorithm
n=100;  % data points
Y=rand(10,n);
T0=3;
D=pdist2(Y',Y');
Kyy=exp(-D.^2/mean(mean(D))^2);
A=rand(n,floor(n)/2)-0.5;
i_z=5;
Kzy=Kyy(i_z,:);
Kzz=Kyy(i_z,i_z);
[x,res_x] = NN_KOMP(A,Kyy,Kzy,Kzz,T0)