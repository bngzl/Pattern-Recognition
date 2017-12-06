function[x_estimate] = reconstruct(u_m, nSamples, normalised, mean)
% sample i  
w = (normalised'*u_m)';
x_estimate = mean*ones(1,nSamples) + (w'*u_m')'; 
end