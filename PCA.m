% Question 1 
% Compute mean face 
N = 52*8; 
sum_x_train = x_train * ones([416,1]); 
mean_face = sum_x_train.*(1/N); 

% Compute covariance matrix S 
A = x_train - mean_face*ones([1,416]); 
S = A*1/N