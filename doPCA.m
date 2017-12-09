function[eigenfaces, eigenvalues]=doPCA(X_train_dataset, M_pca, nTrainSamples)
% M_pca < N

x_mean = mean(X_train_dataset, 2); 
X_normalised = X_train_dataset - x_mean * ones (1, nTrainSamples);

% Cov S for ATA 
St = X_normalised' * X_normalised ./double(nTrainSamples); 
[v, Dv] = eigs(St, M_pca); 
eigenfaces = normc(X_normalised*v); 
eigenvalues = diag(Dv); 

end


