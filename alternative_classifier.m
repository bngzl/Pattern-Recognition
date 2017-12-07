%%%%%%%%%%%%%%%%%%%%% Alternative Classification %%%%%%%%%%%%%%%%%%%%
% Unpack data: 
load data/face_split_0.7.mat; 
x_train = data('x_train');
x_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
nClass = data('nClass'); 

reconstruction_error = zeros(data('nClass'),nTestSamples);
M_pca = 150; 

for i = 1:nClass
    % Separate data for all of i class: 
    index = find(y_train == i); 
    nTrainSamples_subset = length(index); 
        
    % PCA: 
    X_subset_train = x_train(:,index);
    X_subset_mean = mean(X_subset_train, 2); 
    X_subset_normalised = X_subset_train - X_subset_mean* ones(1,nTrainSamples_subset);
    Sf_subset = X_subset_normalised * X_subset_normalised' ./ double(nTrainSamples_subset); 
    [u_m_subset, Du_m_subset] = eigs(Sf_subset, M_pca);
    
    % Reconstruction Error: 
    x_test_normalised = x_test - X_subset_mean*ones(1,nTestSamples); 
    x_estimate = reconstruct(u_m_subset, nTestSamples, x_test_normalised, X_subset_mean);
    reconstruction_error(i,:) = vecnorm(x_test - x_estimate).^2 * ones(nTestSamples, 1)/double(nTestSamples); 
end