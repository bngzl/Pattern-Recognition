% Question 1: 
% x_normalised: A
% Sf: Cov S for AAT 
% St: Cov S for ATA 

clear all;

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
evals_u = sort(diag(Du), 'descend'); 

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

M_pca = 150;
[u_m, Du_m] = eigs(Sf, M_pca); 
x_estimate = reconstruct(u_m, nTrainSamples, x_normalised, x_mean); 


train_error = zeros(1, M_pca); 
% Training/Theoretical Reconstruction error: 
% for i = 1:M_pca
%     train_error(i) = sum(evals_u(i+1));   
% end
% plot(train_error);

% reconstruct(M_pca, 1, nTrainSamples, x_normalised, Sf, x_mean); 


% Test Reconstruction error:  
x_normalised_test = x_test - x_mean*ones(1,nTestSamples); 
x_test_estimate = reconstruct(u_m, nTestSamples, x_normalised_test, x_mean);

test_error = (vecnorm(x_test - x_test_estimate).^2) * ones(nTestSamples, 1);
test_error = test_error/double(nTestSamples); 



