% Question 1: 
% x_normalised: A
% Sf: Cov S for AAT 
% St: Cov S for ATA 

clear all;

% Unpack data: 
load data/face_split_0.7.mat;
X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
X_normalised_test = X_test - x_mean*ones(1,nTestSamples); 
M_pca = 100; 

%Part 1a/b: Computing Eigenfaces from AAT and ATA 
% Cov matrix S for AAT  
Sf = X_normalised_train * X_normalised_train' ./ double(nTrainSamples); 

% Cov matrix S for ATA
St = X_normalised_train' * X_normalised_train ./ double(nTrainSamples);

% Analysis of Eigenvectors and Eigenvalues of Sf and St 
[u, Du] = eig(Sf); 
evals_u = sort(diag(Du), 'descend'); 

[v, Dv] = eig(St); 
evals_v = sort(diag(Dv), 'descend'); 

%% Plot Eigenvalue to Index (1a)
plot(evals_u); 
hold on; 
plot(evals_v); 
title('Magnitude of Eigenvalues');
ylabel ('Eigenvalues');
xlabel ('Index');
legend('Eigenvalues u','Eigenvalues v'); 
xlim([0,250]); 

%% Determining No. of Eigenvectors with non-zero Eigenvalues (1a)

nonzero_evals_u = size(find(abs(evals_u) > 0.00001)); 
nonzero_evals_v = size(find(abs(evals_u) > 0.00001));

%% Part 2: Application of Eigenfaces 
% Reconstruction Error with varying M_pca 
theoretical_error = zeros(1,M_pca); 
test_error = zeros(1, M_pca); 
train_error = zeros(1, M_pca);

for i = 1:M_pca
    theoretical_error(i) = evals_u(i+1:end)'*ones(2576-i,1);
    [v_m, ~] = eigs(St, i); 
    u_m = normc(X_normalised_train*v_m);
    
    X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
    X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);
    
    % Reconstruction error:  
    test_error(i) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
    train_error(i) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
end
plot(theoretical_error);
hold on; 
plot(train_error);
hold on;
plot(test_error); 

title('Error with Varying M Eigenvectors');
ylabel ('Reconstruction Error');
xlabel ('M');
legend('Theoretical error','Train error','Test error'); 
