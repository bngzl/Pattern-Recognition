% Report script: 
clear all; 
load data/face.mat; 

%% Creating multiple images for class 1: 
for i = 1:10
    filename = sprintf('class_1_%d', i); 
    imwrite(mat2gray(showImage(X(:,i))),[filename, '.jpg']); 
end

%% Checking best training/test split 
clear all; 

for i = 0.1:0.1:0.5
    name = sprintf('data/face_split_%d.mat',i); 
    load name;  
    
    X_train = data('x_train');
    X_test = data('x_test');
    nTrainSamples = data('nTrainSamples');
    nTestSamples = data('nTestSamples');
    nFeatures = data('nFeatures');
    nClass = data('nClass'); 
    y_train = data('y_train'); 
    y_test = data('y_test'); 
    
    
end

%% Eigenfaces 
clear all; 
load data/face_split_0.7.mat; 
X_train = data('x_train'); 
nTrainSamples = data('nTrainSamples');

[eigenfaces, eigenvalues] = doPCA(X_train, 250, nTrainSamples);

% for i = 1:10
%     filename = sprintf('eigenvector_%d', i); 
%     imwrite(mat2gray(showImage(eigenfaces(:,i))),[filename, '.jpg']);
%     eigenvalues(i)
% end

%imwrite(mat2gray(showImage(mean(X_train,2))), 'mean_image.jpg')

%%  Reconstruction error 

clear all; 
load data/face_split_0.7.mat; 
X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');

x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
X_normalised_test = X_test - x_mean*ones(1,nTestSamples); 

St = X_normalised_train' * X_normalised_train ./ double(nTrainSamples);

M_pca = 363; 
train_error = zeros(1, M_pca); 
test_error = zeros(1, M_pca); 
time_taken = zeros(1, M_pca); 

for i = 1:M_pca
    tic; 
    [v_m, ~] = eigs(St, i); 
    u_m = normc(X_normalised_train*v_m);
    
    X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
    X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);
    
    test_error(i) = (sqrt(vecnorm(X_test - X_test_estimate).^2)) * ones(nTestSamples, 1)/double(nTestSamples);
    train_error(i) = (sqrt(vecnorm(X_train - X_train_estimate).^2)) * ones(nTrainSamples, 1)/double(nTrainSamples);
    time_taken(i) = toc;
end

yyaxis left; 
plot(train_error);
hold on;

plot(test_error); 
hold on;

yyaxis right;
plot (time_taken); 
xlim([0,363]); 

title('Reconstruction RMS Error and Time taken/s with Varying M Eigenvectors'); 
xlabel('M'); 

yyaxis left; 
ylabel ('Reconstruction Root Mean Sq Error');
yyaxis right; 
ylabel ('Time Taken/s'); 

legend ('Train Error', 'Test Error', 'Time Taken'); 

%% NN Classifier Error with Various M_pca 

clear all; 
load data/face_split_0.7.mat; 

load data/face_split_0.7.mat;
X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
y_train = data('y_train'); 
y_test = data('y_test'); 

x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
X_normalised_test = X_test - x_mean*ones(1,nTestSamples);  
St = X_normalised_train' * X_normalised_train ./ double(nTrainSamples);

class_error = zeros(nTrainSamples,1); 
labels = y_test';
predicted_class = zeros(nTestSamples,1);

M_pca = 363; 
accuracy_NN = zeros(M_pca,1); 

for a = 1:M_pca
    [v_m, ~] = eigs(St, a); 
    u_m = normc(X_normalised_train*v_m);
    
    W_train = (X_normalised_train'*u_m)';
    W_test = (X_normalised_test'*u_m)';
    
    for i = 1:nTestSamples
        W_diff = W_test(:,i)*ones(1,nTrainSamples) - W_train; 
        class_error(:,1) = (sqrt(vecnorm(W_diff).^2))';
        [class_error, index] = sort(class_error, 'ascend'); 
        predicted_class(i)=y_train(index(1)); 
    end
    
    correct_prediction_NN = sum(predicted_class == labels); 
    accuracy_NN(a) = double (correct_prediction_NN)/ double(nTestSamples);
end

plot(accuracy_NN); 
title('NN Classifier Accuracy with Varying M')
xlabel('M');
ylabel('NN Classifier Accuracy'); 
xlim([0,363]); 

%% NN Classifier Error with Various M_pca 

clear all; 
load data/face_split_0.6.mat;

X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
y_train = data('y_train'); 
y_test = data('y_test'); 
nClass = data('nClass'); 

reconstruction_error = zeros(data('nClass'),nTestSamples);
labels = y_test'; 

M_pca = 6; 
%I = zeros(data('nClass'),nTestSamples);
accuracy_alternate = zeros(M_pca, 1);

for a = 1:M_pca
    for i = 1:nClass
        % Separate data for all of i class: 
        index = find(y_train == i); 
        nTrainSamples_subset = length(index); 

        % PCA: 
        X_subset_train = X_train(:, index); 
        [u_m_subset, ~] = doPCA(X_subset_train, a, nTrainSamples_subset); 

        % Reconstruction Error: 
        x_subset_mean = mean(X_subset_train, 2); 
        X_test_normalised = X_test - x_subset_mean * ones(1,nTestSamples); 
        X_estimate = reconstruct(u_m_subset, nTestSamples, X_test_normalised, x_subset_mean); 
        reconstruction_error(i,:) = sqrt(vecnorm(X_test - X_estimate).^2);  
    end
    
    [reconstruction_error, I] = sort(reconstruction_error); 
    correct_prediction_alternate = sum(I(1,:)' == labels); 
    accuracy_alternate(a) = double (correct_prediction_alternate)/ double(nTestSamples);
end

plot(accuracy_alternate); 
title('Alternate Classifier Accuracy with Varying M')
xlabel('M');
ylabel('NN Classifier Accuracy'); 
xlim([1,M_pca]); 