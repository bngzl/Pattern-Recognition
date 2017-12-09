%%%%%%%%%%%%%%%%%%%%% Alternative Classification %%%%%%%%%%%%%%%%%%%%
% Unpack data: 
clear all; 
load data/face_split_0.7.mat; 


reconstruction_error = zeros(data('nClass'),nTestSamples);
M_pca = 3; 
%I = zeros(data('nClass'),nTestSamples); 
labels = y_test'; 

for i = 1:nClass
    % Separate data for all of i class: 
    index = find(y_train == i); 
    nTrainSamples_subset = length(index); 
        
    % PCA: 
    X_subset_train = X_train(:, index); 
    [u_m_subset, ~] = doPCA(X_subset_train, M_pca, nTrainSamples_subset); 
    
    % Reconstruction Error: 
    x_subset_mean = mean(X_subset_train, 2); 
    X_test_normalised = X_test - x_subset_mean * ones(1,nTestSamples); 
    X_estimate = reconstruct(u_m_subset, nTestSamples, X_test_normalised, x_subset_mean); 
    reconstruction_error(i,:) = sqrt(vecnorm(X_test - X_estimate).^2);  
    % Cols of reconstruction_error are the test samples, rows are the
    % errors for corresponding classes 
end

[reconstruction_error, I] = sort(reconstruction_error); 
correct_prediction_alternate = sum(I(1,:)' == labels); 
accuracy_alternate = double (correct_prediction_alternate)/ double(nTestSamples);
