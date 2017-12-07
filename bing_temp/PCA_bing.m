<<<<<<< HEAD
function [P] = PCA_bing(data, M_pca)
%because I need it to do my shit bng, I have no time to wait
=======
function [] = PCA_bing(data, M_pca)
>>>>>>> 6986fb4c642db504fa4a67fc22a430e943cca502

% unpack data
x_train = data('x_train');
x_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

% mean face
mean_face = mean(x_train,2);
x_centered = x_train - mean_face * ones(1,nTrainSamples);

% cov matrix of nFeatures * nFeatures
Sf = x_centered * x_centered' ./ double(nTrainSamples);

% cov matrix of nTrainSamples * nTrainSamples
St = x_centered' * x_centered ./ double(nTrainSamples);

[Vf, Df] = eigs(Sf,M_pca);

% Take M_pca largest eigenvectors and transform our data set
P = Vf(:,1:M_pca);

x_pca_mean = P'*mean_face;
x_pca_train = P' * x_train + x_pca_mean*ones(1,nTrainSamples);
x_pca_test = P' * x_test + x_pca_mean*ones(1,nTestSamples); 

data('x_pca_train') = x_pca_train;
data('x_pca_test') = x_pca_test;
data('M_pca') = M_pca;

end

