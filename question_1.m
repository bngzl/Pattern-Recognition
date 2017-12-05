% Question 1: 
% x_normalised: A
% Sf: Cov S for AAT 
% St: Cov S for ATA 

clear all

% Unpack data: 
load data/face_split_0.7.mat;
x_train = data('x_train');
x_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

% PCA: 
x_mean = mean(x_train, 2); 
x_normalised =  x_train - x_mean * ones(1,nTrainSamples); 

% Cov matrix S for AAT  
Sf = x_normalised * x_normalised' ./ double(nTrainSamples); 

% Cov matrix S for ATA
St = x_normalised' * x_normalised ./ double(nTrainSamples);

% Analysis of Eigenvectors and Eigenvalues of Sf and St 
[u, Du] = eig(Sf); 
[evals_u, index] = sort(diag(Du), 'descend'); 

[v, Dv] = eig(St); 
evals_v = sort(diag(Dv), 'descend'); 

% Plot eigenvalue to index
% plot(evals_u); 
% hold on; 
% plot(evals_v); 
% title('Magnitude of Eigenvalues');
% ylabel ('Eigenvalues');
% xlabel ('Index');
% legend('Eigenvalues u','Eigenvalues v'); 
% xlim([0,250]); 

nonzero_evals_u = size(find(abs(evals_u) > 0.00001)); 
nonzero_evals_v = size(find(abs(evals_u) > 0.00001));

% Reconstruction error: 
for M_pca = 1:250
    x_estimate = reconstruct(M_pca, nTrainSamples, x_normalised, Sf, x_mean); 
    
end

M_pca = 150; % Largest M Eigenvalues 
[u_m, Du_m] = eigs (Sf, M_pca); 

