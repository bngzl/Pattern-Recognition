function[x_estimate] = reconstruct(M_pca, nTrainSamples, x_normalised, Sf, x_mean)
% sample i 
[u_m, ~] = eigs(Sf, M_pca); 
w = (x_normalised'*u_m)';
x_estimate = x_mean*ones(1,nTrainSamples) + (w'*u_m')'; 
end