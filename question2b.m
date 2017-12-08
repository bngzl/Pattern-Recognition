load data/face_split_0.7.mat

% Unpacking variables from data model
x_train = data('x_train');
y_train = data('y_train');
x_test = data('x_test');
y_test = data('y_test');
nTrain = data('nTrain');
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');
nFeatures = data('nFeatures');
nClass = data('nClass');

nModels = 10;

% Define randomness parameter and dimensions
% randomness parameter, rho, will be the ratio of M1:M_pca
rho = 0.2;
M_pca = 100;
M_lda = nClass-1;
M1 = round(M_pca * rho);
M0 = M_pca - M1;


% RANDOM SAMPLING IN FEATURE SPACE

% PCA to obtain nTrainSamples-1 eigenfaces
[efaces, evalues] = doPCA(x_train, nTrainSamples-1, nTrainSamples);

% Initialise data matrices
P_pca = zeros(nFeatures, M_pca, nModels);
P_lda = zeros(M_pca, M_lda, nModels);
x_train_pca = zeros(M_pca, nTrainSamples,nModels);
x_train_lda = zeros(M_lda, nTrainSamples,nModels);
y_pred = zeros(nTestSamples, nModels);

for n = 1:nModels
    % For each model, use M0 eigenfaces corresponding to M0 largest eignevalues
    % and M1 random eigenfaces corresponding from (M0+1)th to (nTrainSamples-1)th 
    % largest eigenvalues such that M0+M1 = M_pca <= nTrainSamples - nClass 
    
    % Generate random indices for M1 eigenfaces
    inds = sort(randsample(nTrainSamples-1-M0, M1)) + M0;
    P_pca(:,1:M0,n) = efaces(:,1:M0);
    for i =1:M1
        P_pca(:,M0+i,n)= efaces(:,inds(i));
    end

    % Project training data into PCA subspace
    x_train_pca(:,:,n) = P_pca(:,:,n)'*x_train;
    
    % Perform LDA on PCA-transformed training data
    P_lda(:,:,n) = LDA(x_train_pca(:,:,n),nTrain, M_lda);
    x_train_lda(:,:,n) = P_lda(:,:,n)' * x_train_pca(:,:,n);

    % Perform PCA-LDA on test data
    x_test_lda = P_lda(:,:,n)'* P_pca(:,:,n)'*x_test;

    
    % Testing phase
    % Test data is projected onto PCA-LDA subspace,
    % NN w/ euclidean distance is used to predict
    % for this model
    for i = 1:nTestSamples
        diffs = x_test_lda(:,i)*ones(1,nTrainSamples) - x_train_lda(:,:,n);
        dists = vecnorm(diffs);
        [min_val, min_ind] = min(dists); 
        y_pred(i,n) = y_train(min_ind);
    end
end


% Testing phase:
% Test data is projected onto n random subspaces and fed into n LDA
% classifers in parallel
% Output of n classifiers are combined through some fusion scheme (TBD)

% Fusion Scheme (more schemes TBI)
y_pred_combined = mode(y_pred,2); % Majority voting

% Test Accuracy
diff = y_test'-y_pred_combined;
random_feature_accuracy = sum(diff==0)/length(diff)