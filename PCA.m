% Question 1 
% Initialisation
clear all;
load data/face_split_0.7.mat;

% Unpack data: 
x_train = data('x_train');
x_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

% mean face
mean_face = mean(x_train,2);
A = x_train - mean_face * ones(1,nTrainSamples); % Matrix of normalised training faces in each col 

% cov matrix of nFeatures * nFeatures
Sf = A * A' ./ double(nTrainSamples);

% cov matrix of nTrainSamples * nTrainSamples 
St = A' * A ./ double(nTrainSamples);

[u, Du] = eig(Sf); 
[v, Dv] = eig(St); 

u = fliplr(u); 
evals_u = flipud(diag(Du)); 

% Plot eigenvalues against index 
% plot(evals_u); 
% title('Eigenvalues against index'); 
% ylabel('Eigenvalues'); 
% xlabel('Index'); 
% xlim([0,250]); 

v = fliplr(v); 
evals_v = flipud(diag(Dv)); 

% Nearest Neighbour Classification: 
[u_m, ~] = eigs(Sf, M_pca); 
w_train = (A'*u_m)';

A_test = x_test - mean_face * ones(1, nTestSamples); 
w_test = (A_test'*u_m)'; 





% Accuracy vs number of eigenvector plot 