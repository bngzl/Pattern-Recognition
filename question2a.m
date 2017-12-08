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

% PCA to obtain nTrainSamples-1 eigenfaces
[efaces, evalues] = doPCA(x_train, nTrainSamples-1, nTrainSamples);

% Initialise accuracy matrix
accuracy = zeros(length(2:2:312),length(2:2:51));



% Varying M_pca and M_lda
for M_pca = 2:2:312
    for M_lda = 2:2:min(M_pca,51)
        % Initialise data matrices
        P_pca = zeros(nFeatures, M_pca);
        P_lda = zeros(M_pca, M_lda);
        x_train_pca = zeros(M_pca, nTrainSamples);
        x_train_lda = zeros(M_lda, nTrainSamples);
        y_pred = zeros(1,nTestSamples);


        % Select eigenfaces based on M_pca largest eigenvalues
        P_pca = efaces(:,1:M_pca);

        % Project training data into PCA subspace
        x_train_pca = P_pca'*x_train;

        % Perform LDA on PCA-transformed training data
        P_lda = LDA(x_train_pca,nTrain, M_lda);
        x_train_lda = P_lda' * x_train_pca;

        % Perform PCA-LDA on test data
        x_test_lda = P_lda'* P_pca'*x_test;


        % Test data is projected onto PCA-LDA subspace,
        % NN w/ euclidean distance is used to predict
        for i = 1:nTestSamples
            diffs = x_test_lda(:,i)*ones(1,nTrainSamples) - x_train_lda;
            dists = vecnorm(diffs);
            [min_val, min_ind] = min(dists); 
            y_pred(i) = y_train(min_ind);
        end
        % Test Accuracy
        diff = y_test-y_pred;
        accuracy(M_pca/2,M_lda/2) = sum(diff==0)/length(diff);
    end
end

v_cols = max(accuracy,[],1);
v_rows = max(accuracy,[],2);

[v,j] = max(v_cols);
[v,i] = max(v_rows);

fprintf('Max accuracy: %f, (M_pca,M_lda) = (%d,%d)\n',v, i*2,j*2);

figure;
[xx,yy] = meshgrid(2:2:312,2:2:51);
surf(xx,yy,accuracy','FaceAlpha',0.5)
ylabel("M\_lda")
xlabel("M\_pca")
zlabel('Accuracy')
title('2D plot of M\_pca and M\_lda against Accuracy')


% Repeat model for max accuracy scenario
M_pca = i*2;
M_lda = j*2;

% Initialise data matrices
P_pca = zeros(nFeatures, M_pca);
P_lda = zeros(M_pca, M_lda);
x_train_pca = zeros(M_pca, nTrainSamples);
x_train_lda = zeros(M_lda, nTrainSamples);
y_pred = zeros(1,nTestSamples);


% Select eigenfaces based on M_pca largest eigenvalues
P_pca = efaces(:,1:M_pca);

% Project training data into PCA subspace
x_train_pca = P_pca'*x_train;

% Perform LDA on PCA-transformed training data
[P_lda,Sw,Sb] = LDA(x_train_pca,nTrain, M_lda);
x_train_lda = P_lda' * x_train_pca;

% Perform PCA-LDA on test data
x_test_lda = P_lda'* P_pca'*x_test;


% Test data is projected onto PCA-LDA subspace,
% NN w/ euclidean distance is used to predict
for i = 1:nTestSamples
    diffs = x_test_lda(:,i)*ones(1,nTrainSamples) - x_train_lda;
    dists = vecnorm(diffs);
    [min_val, min_ind] = min(dists); 
    y_pred(i) = y_train(min_ind);
end

% Ranks of Sw and Sb
fprintf('Rank of Between-class Scatter Matrix = %d\n',rank(Sb))
fprintf('Rank of Within-class Scatter Matrix = %d\n',rank(Sw))

% Confusion matrix goes here (TODO, idk how many classes do you want to talk about)



