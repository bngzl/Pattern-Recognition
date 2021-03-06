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
theoretical_error = zeros(1, M_pca); 

Sf = X_normalised_train * X_normalised_train' ./ double(nTrainSamples); 
[~, Du] = eig(Sf); 
evals_u = sort(diag(Du),'descend'); 

for i = 1:M_pca
    tic; 
    [v_m, Dv_m] = eigs(St, i); 
    u_m = normc(X_normalised_train*v_m);
    
    X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
    X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);
        
    test_error(i) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
    train_error(i) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
    time_taken(i) = toc;  
    
    theoretical_error(i) = evals_u(i+1:end)'*ones(2576-i,1);
end

yyaxis left; 
plot(theoretical_error); 
hold on; 

plot(train_error);
hold on;

plot(test_error); 
hold on;

yyaxis right;
plot (time_taken); 
xlim([0,363]); 

title('Reconstruction Error and Time taken/s with Varying M Eigenvectors'); 
xlabel('M'); 

yyaxis left; 
ylabel ('Reconstruction Error');
yyaxis right; 
ylabel ('Time Taken/s'); 

legend ('Theoretical Error', 'Training Error', 'Test Error', 'Time Taken'); 

%% NN Classifier Error with Various M_pca 

clear all; 
load data/face_split_0.7.mat; 

load data/face_split_0.7.mat;
X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nClass = data('nClass');
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

reconstruction_error = zeros(data('nClass'),nTestSamples); 

M_pca = 6; 
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

subplot(2,1,1); 
plot(accuracy_NN); 
title('NN Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([0,363]);
subplot(2,1,2); 
plot(accuracy_alternate,'red'); 
title('Alternate Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([1,M_pca]); 


%% Alternate Classifier Error with Various M_pca 
reconstruction_error = zeros(data('nClass'),nTestSamples); 

M_pca = 7; 
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

subplot(2,1); 
plot(accuracy_NN); 
title('NN Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([0,363]);
subplot(2,2); 
plot(accuracy_alternate); 
title('Alternate Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([1,M_pca]); 

%% Alternate Classifier Error with Various M_pca 

clear all; 
load data/face_split_0.7.mat;

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

M_pca = 7; 
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
ylabel('Accuracy'); 
xlim([1,M_pca]); 

%% Measuring time for AAT vs ATA 
 
clear all; 
load data/face_split_0.7.mat;

X_train = data('x_train');
nTrainSamples = data('nTrainSamples');
M_pca = 250; 

tic; 
x_mean = mean(X_train, 2); 
X_normalised = X_train - x_mean * ones (1, nTrainSamples);

% Cov S for ATA 
St = X_normalised' * X_normalised ./double(nTrainSamples); 
[v, Dv] = eigs(St, M_pca);
ATA_time = toc; 

tic; 
x_mean = mean(X_train, 2); 
X_normalised = X_train - x_mean * ones (1, nTrainSamples);

% Cov S for AAT
Sf = X_normalised * X_normalised' ./double(nTrainSamples); 
[u, Du] = eigs(Sf, M_pca);
AAT_time = toc; 

%% Plot Eigenvalue to Index 

clear all; 
load data/face_split_0.7.mat
X_train = data('x_train');
nTrainSamples = data('nTrainSamples');

x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
Sf = X_normalised_train * X_normalised_train' ./ double(nTrainSamples);
St = X_normalised_train' * X_normalised_train ./ double(nTrainSamples);

% Analysis of Eigenvectors and Eigenvalues of Sf and St 
[u, Du] = eig(Sf); 
evals_u = sort(diag(Du), 'descend'); 

[v, Dv] = eig(St); 
evals_v = sort(diag(Dv), 'descend'); 

plot(evals_u); 
hold on; 
plot(evals_v); 
title('Magnitude of Eigenvalues');
ylabel ('Eigenvalues');
xlabel ('Index');
legend('Eigenvalues u_m','Eigenvalues v_m'); 
xlim([0,250]); 

%% Displaying 150th Eigenvector of u_m 

clear all; 
load data/face_split_0.7.mat
X_train = data('x_train');
nTrainSamples = data('nTrainSamples');

x_mean = mean(X_train, 2); 
X_normalised_train = X_train - x_mean*ones(1,nTrainSamples); 
Sf = X_normalised_train * X_normalised_train' ./ double(nTrainSamples);
[u_m, Du_m] = eigs(Sf, 250); 

imwrite(mat2gray(showImage(u_m(:,150))),'150_eigenvector.jpg');

%% Reconstruction Error for 3 images of your choice 

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
 
train_error = zeros(1,4); 
test_error = zeros(1, 4); 

imwrite(mat2gray(showImage(X_train(:,1))),'class_1_original.jpg'); 
imwrite(mat2gray(showImage(X_train(:,15))),'class_2_original.jpg'); 
imwrite(mat2gray(showImage(X_test(:,5))),'test_class_1_original.jpg'); 
imwrite(mat2gray(showImage(X_test(:,11))),'test_class_2_original.jpg'); 

% M_pca = 10 
i = 10; 
[v_m, Dv_m] = eigs(St, i); 
u_m = normc(X_normalised_train*v_m);
X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);

test_error(1) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
train_error(1) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
imwrite(mat2gray(showImage(X_train_estimate(:,1))),'class_1_M10.jpg');
imwrite(mat2gray(showImage(X_train_estimate(:,15))),'class_2_M10.jpg');
imwrite(mat2gray(showImage(X_test_estimate(:,5))),'test_class_1_M10.jpg'); 
imwrite(mat2gray(showImage(X_test_estimate(:,11))),'test_class_2_M10.jpg'); 

i = 50; 
[v_m, Dv_m] = eigs(St, i); 
u_m = normc(X_normalised_train*v_m);
X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);

test_error(2) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
train_error(2) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
imwrite(mat2gray(showImage(X_train_estimate(:,1))),'class_1_M50.jpg');
imwrite(mat2gray(showImage(X_train_estimate(:,15))),'class_2_M50.jpg');
imwrite(mat2gray(showImage(X_test_estimate(:,5))),'test_class_1_M50.jpg'); 
imwrite(mat2gray(showImage(X_test_estimate(:,11))),'test_class_2_M50.jpg'); 

i = 200; 
[v_m, Dv_m] = eigs(St, i); 
u_m = normc(X_normalised_train*v_m);
X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);

test_error(3) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
train_error(3) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
imwrite(mat2gray(showImage(X_train_estimate(:,1))),'class_1_M200.jpg');
imwrite(mat2gray(showImage(X_train_estimate(:,15))),'class_2_M200.jpg');
imwrite(mat2gray(showImage(X_test_estimate(:,5))),'test_class_1_M200.jpg'); 
imwrite(mat2gray(showImage(X_test_estimate(:,11))),'test_class_2_M200.jpg'); 

i = 363; 
[v_m, Dv_m] = eigs(St, i); 
u_m = normc(X_normalised_train*v_m);
X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);

test_error(4) = (vecnorm(X_test - X_test_estimate).^2) * ones(nTestSamples, 1)/double(nTestSamples);
train_error(4) = (vecnorm(X_train - X_train_estimate).^2) * ones(nTrainSamples, 1)/double(nTrainSamples);
imwrite(mat2gray(showImage(X_train_estimate(:,1))),'class_1_M363.jpg');
imwrite(mat2gray(showImage(X_train_estimate(:,15))),'class_2_M363.jpg');
imwrite(mat2gray(showImage(X_test_estimate(:,5))),'test_class_1_M363.jpg'); 
imwrite(mat2gray(showImage(X_test_estimate(:,11))),'test_class_2_M363.jpg'); 

%% Checking Classifier Accuracy with Train/Test Split 
clear all; 
accuracy_NN = zeros(9, 1); 
accuracy_alternate = zeros(9,1); 

for a = 2:9 
    name = sprintf('data/face_split_0.%d.mat',a); 
    load (name);  
    
    X_train = data('x_train');
    X_test = data('x_test');
    nTrainSamples = data('nTrainSamples');
    nTestSamples = data('nTestSamples');
    nClass = data('nClass');
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
    
    M_NN = 100; 
    M_AC = 5; 
    
    %NN: 
    [v_m, ~] = eigs(St, M_NN); 
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
    
    reconstruction_error = zeros(data('nClass'),nTestSamples); 
    
    for i = 1:nClass 
        % Separate data for all of i class: 
        index = find(y_train == i); 
        nTrainSamples_subset = length(index); 

        % PCA: 
        X_subset_train = X_train(:, index); 
        [u_m_subset, ~] = doPCA(X_subset_train, a-1, nTrainSamples_subset); 

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

subplot(2,1,1); 
plot(accuracy_NN); 
title('NN Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([0,9]);
subplot(2,1,2); 
plot(accuracy_alternate,'red'); 
title('Alternate Classifier Accuracy with Varying M')
xlabel('M');
ylabel('Accuracy'); 
xlim([1,9]); 

%% Confusion Matrix for 10 classes 
clear all; 
load data/face_split_10_classes_0.7.mat; 

X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
nClass = data('nClass'); 

M_pca = 69; 
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

labels = zeros(nTestSamples, data('nClass')); 
predicted_class = zeros(nTestSamples, data('nClass')); 
class_error = zeros(nTrainSamples,1); 

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

% Alternate Classifier 
% M_pca_alternate = 5; 
% reconstruction_error = zeros(data('nClass'),nTestSamples);
% predicted_class_alternate = zeros(nTestSamples, data('nClass')); 
% 
% for a = 1:nTestSamples 
%     for i = 1:nClass 
%         index = find(y_train == i); 
%         nTrainSamples_subset = length(index); 
% 
%         X_subset_train = X_train(:,index); 
%         [u_m_subset, ~] = doPCA(X_subset_train, M_pca_alternate, nTrainSamples_subset); 
%         x_subset_mean = mean(X_subset_train, 2); 
%         X_test_normalised = X_test - x_subset_mean * ones(1,nTestSamples); 
%         X_estimate = reconstruct(u_m_subset, nTestSamples, X_test_normalised, x_subset_mean); 
%         reconstruction_error(i,:) = sqrt(vecnorm(X_test - X_estimate).^2);  
%     end 
% 
%     [~, I] = sort(reconstruction_error); 
%     predicted_class_alternate(a,I(1,a)) = 1; 
% end 
% figure(); 
% plotconfusion(labels',predicted_class');
% figure();   
% plotconfusion(labels', predicted_class_alternate') 

%% Show Success and Failure Cases 
%% Confusion Matrix for 10 classes 
clear all; 
load data/face_split_10_classes_0.7.mat; 

X_train = data('x_train');
X_test = data('x_test');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
nClass = data('nClass'); 

M_pca = 69; 
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

class_error = zeros(nTrainSamples,1); 
labels = y_test';
predicted_class = zeros(nTestSamples,1); 

% NN Classifier:
% Success: Target Class 10, Output Class 10 
% Target class: 2, Output class: 5 

X_test_estimate = reconstruct(u_m, nTestSamples, X_normalised_test, x_mean);
X_train_estimate = reconstruct(u_m, nTrainSamples, X_normalised_train, x_mean);

for i = 1:nTestSamples
    W_diff = W_test(:,i)*ones(1,nTrainSamples) - W_train; 
    class_error(:,1) = (sqrt(vecnorm(W_diff).^2))';
    [class_error, index] = sort(class_error, 'ascend'); 
    predicted_class(i)=y_train(index(1));
end

for i = 1:nTestSamples
    filename = sprintf('class_%d', i);
    if predicted_class(i) ~= labels(i)  
        originalfilename = sprintf('original_class_%d_%d', i, floor(i./3));
        imwrite(mat2gray(showImage(X_test(:,i))),[originalfilename,'.jpg']);
        imwrite(mat2gray(showImage(X_test_estimate(:,i))),[filename,'.jpg']);
        predictedfilename = sprintf('predicted_class_%d_%d', i, predicted_class(i));
        imwrite(mat2gray(showImage(X_train(:,7*predicted_class(i)))),[predictedfilename,'.jpg']);
    end 
end

% for i = 1:nTestSamples
%     filename = sprintf('class_%d', i);
%     if predicted_class(i) == labels(i)  
%         originalfilename = sprintf('original_class_%d_%d', i, uint16(i/3));
%         imwrite(mat2gray(showImage(X_test(:,i))),[originalfilename,'.jpg']);
%         imwrite(mat2gray(showImage(X_test_estimate(:,i))),[filename,'.jpg']);
%         predictedfilename = sprintf('predicted_class_%d_%d', i, predicted_class(i));
%         imwrite(mat2gray(showImage(X_train(:,7*predicted_class(i)))),[predictedfilename,'.jpg']);
%     end 
% end
