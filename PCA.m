% Question 1 
% Initialisation
clear all;
load data/partitionedData.mat; 
[D, N] = size(x_train); % D=W*H of each image, N is no. of samples

% Compute mean face 
sum_x_train = x_train * ones([N,1]); 
mean_face = sum_x_train.*(1/N); 

% Compute covariance matrix S 
A = x_train - mean_face*ones([1,N]); 
S = (1/N)*(A*A'); 
[u,evals] = eig(S); % A*u = u*D
evals_u = diag(evals); 
u = normc(u); % Normalise eigenvectors 

non_zero_evals = size(find(evals_u(:)>0.001)); % no. of evecs with non zero evals <- check this! 

% Compute (1/N)ATA (PCA when D >> N) 
S2 = (1/N)*(A'*A); 
[v,evals] = eig(S2); 
evals_v = diag(evals); 
v = normc(v); 

% Determine if v and evals_v are identical to u and evals_u 

% The M evals_v of ATA correspond to the M largest evals_u of AAT

% Accuracy vs number of eigenvector plot 