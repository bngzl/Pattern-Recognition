% Question 1 
% Compute mean face 
load partitionedData.mat 
N = 52*8; 
sum_x_train = x_train * ones([416,1]); 
mean_face = sum_x_train.*(1/N); 

% Compute covariance matrix S 
A = x_train - mean_face*ones([1,416]); 
S = (1/N)*(A*A'); 
[V,D] = eig(S); % A*V = V*D
evals = diag(D); 

% No. of zero eigenvalues: 
no_of_zero_evals = size(find(evals));  