load data/face_split_0.7.mat

% RANDOM BAGGING OF TRAINING SET

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

% Define dimensions
M_pca = 100;
M_lda = nClass-1;

% Initialise data matrices
x_bag = zeros(nFeatures, nTrainSamples,nModels);
x_train_pca = zeros(M_pca, nTrainSamples,nModels);
x_train_lda = zeros(M_lda, nTrainSamples,nModels);
P_pca = zeros(nFeatures, M_pca, nModels);
P_lda = zeros(M_pca, M_lda, nModels);
y_pred = zeros(nTestSamples, nModels);

for n = 1:nModels
    % Generate a random bag of nTrainSamples-1 samples for current model
    % Training data is sampled on per-class basis w/ replacement, otherwise
    % there is a possibility of omitting some classes in our bag
    for c = 1:nClass
        for i = 1:nTrain
            ind = randi(nTrain);
            x_bag(:,(c-1)*nTrain + i,n) = x_train(:,(c-1)*nTrain + ind);
        end
    end
    
    % Perform PCA on bagged sample and obtain nTrainSamples-1 eigenfaces
    [efaces, evalues] = doPCA(x_bag(:,:,n), nTrainSamples-1, nTrainSamples);
    P_pca(:,:,n) = efaces(:,1:M_pca);
    x_train_pca(:,:,n) = P_pca(:,:,n)' * x_train;
    
    % Perform LDA on PCA-transformed bagged sample
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

% Fusion Scheme (more schemes TBI)
y_pred_combined = mode(y_pred,2); % Majority voting

% Test Accuracy
diff = y_test'-y_pred_combined;
random_feature_accuracy = sum(diff==0)/length(diff)










