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
v = fliplr(v); 

evals_u = flipud(diag(Du)); 
evals_v = flipud(diag(Dv)); 

% Check u(i) = Av(i): 
for i = 1:nTrainSamples
    tf = isequal(u(:,i),normc(A*v(:,i)));  
    if tf == 1
        fprintf('same') 
    end
end

% Pull out the largest M eigenvectors: 
[u_m, Du_m] = eigs(Sf, M_pca)


% Determine if v and evals_v are identical to u and evals_u 

%c = intersect(evals_v, evals_u); 

% Accuracy vs number of eigenvector plot 