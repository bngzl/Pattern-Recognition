% Question 1 
% Initialisation
clear all;
load data/face_split_0.7.mat;

x_train = data('x_train');

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

% Pull out the largest M eigenvectors: 

% for M_pca = 1:50 
%     [u_m, Du_m] = eigs(Sf, M_pca); 
%     u_m = fliplr(u_m); 
%     a = (A'*u_m); 
%     res = (a*u_m')'+ mean_face*ones(1,364); 
%     showImage(res(:,1));
% end
M_pca = 1
[u_m, Du_m] = eigs(Sf, M_pca); 
a = (A'*u_m); % Project training face onto eigenvector
res = (a*u_m')'+ mean_face*ones(1,364); % Reconstruct as linear comb of faces 
figure; 
showImage(res(:,1));

M_pca = 250
[u_m, Du_m] = eigs(Sf, M_pca); 
a = (A'*u_m); 
res = (a*u_m')'+ mean_face*ones(1,364); 
figure; 
showImage(res(:,1));
 


% [u_m, Du_m] = eigs(Sf, M_pca);
% u_m = fliplr(u_m); 
% 
% comb = (A'*u_m);
% res = mean_face*ones(1,364) + comb'; 


% Accuracy vs number of eigenvector plot 