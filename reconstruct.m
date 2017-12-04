function[w, x_estimate] = reconstruct(M_pca, i, nTrainSamples, A, Sf, mean_face)
% sample i 

[u_m, ~] = eigs(Sf, M_pca); 
w = (A'*u_m)';
x_estimate = mean_face*ones(1,nTrainSamples) + (w'*u_m')'; 
showImage(x_estimate(:,i));

end