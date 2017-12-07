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


M_pca = 200;
theoretical_error = zeros(1,M_pca); 
test_error = zeros(1, M_pca); 
train_error = zeros(1, M_pca);

x_normalised_test = x_test - x_mean*ones(1,nTestSamples); 
x_normalised_train = x_train - x_mean*ones(1,nTrainSamples); 

% Training/Theoretical Reconstruction error: 
for i = 1:M_pca
    theoretical_error(i) = evals_u(i+1:end)'*ones(2576-i,1);
    
    [u_m, Du_m] = eigs(Sf, i); % Extract M largest evals and corresponding evecs
    x_test_estimate = reconstruct(u_m, nTestSamples, x_normalised_test, x_mean);
    x_train_estimate = reconstruct(u_m, nTrainSamples, x_normalised_train, x_mean);
    
    % Reconstruction error:  
    test_error(i) = (vecnorm(x_test - x_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
    train_error(i) = (vecnorm(x_train - x_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
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



