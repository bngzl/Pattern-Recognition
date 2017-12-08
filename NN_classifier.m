%%%%%%%%% Nearest Neighbours Classifier: %%%%%%%%%
clear all; 

% Unpack Data: 
load data/face_split_0.7.mat;
X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

% User Input: 
M_pca = 150;

% Preprocessing: 
x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
X_normalised_test = X_test - x_mean*ones(1,nTestSamples);  

St = X_normalised_train' * X_normalised_train ./ double(nTrainSamples);
[v_m, ~] = eigs(St, M_pca); 
u_m = normc(X_normalised_train*v_m);

W_train = (X_normalised_train'*u_m)';
W_test = (X_normalised_test'*u_m)';

y_train = data('y_train'); 
y_test = data('y_test'); 

%% Create Confusion Matrix: 
% labels = zeros(nTestSamples, data('nClass')); 
% predicted_class = zeros(nTestSamples, data('nClass')); 
% class_error = zeros(nTrainSamples,1); 
% 
% for i = 1:nTestSamples
%     labels(i,y_test(i)) = 1; 
% end
% 
% for i = 1:nTestSamples
%     W_diff = W_test(:,i)*ones(1,nTrainSamples) - W_train; 
%     class_error(:,1) = (sqrt(vecnorm(W_diff).^2))'; 
%     [class_error, index] = sort(class_error, 'ascend'); 
%     predicted_class(i,y_train(index(1))) = 1; 
% end
% plotconfusion(labels',predicted_class')

%% Measure Accuracy: 
class_error = zeros(nTrainSamples,1); 
labels = y_test';
predicted_class = zeros(nTestSamples,1); 

for i = 1:nTestSamples
    W_diff = W_test(:,i)*ones(1,nTrainSamples) - W_train; 
    class_error(:,1) = (sqrt(vecnorm(W_diff).^2))';
    [class_error, index] = sort(class_error, 'ascend'); 
    predicted_class(i)=y_train(index(1)); 
end

correct_prediction_NN = sum(predicted_class == labels); 
accuracy_NN = double (correct_prediction_NN)/ double(nTestSamples);