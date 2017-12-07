<<<<<<< HEAD:PCA_LDA_run.m
load data/face_split_0.7.mat

% M_pca = 200;
% M_lda = 10;

accuracy = zeros(10,50);
for i = 1:10
    M_pca = i*50;
    % PCA goes here
    PCA_bing(data,M_pca);
    for j = 1:50
        % LDA goes here
        y_pred = LDA(data,M_lda);

        % accuracy
        y_test = data('y_test');
        diff = y_pred - y_test;
        accuracy(i,j) = sum(diff==0)/length(diff);

        % confusion matrix 
        % confusionMat = confusionmat(y_pred,y_test)
    end
end
axis = 1:50
plot(axis,accuracy(1,:),axis,accuracy(2,:),axis,accuracy(3,:),...
     axis,accuracy(4,:),axis,accuracy(5,:),axis,accuracy(6,:),...
     axis,accuracy(7,:),axis,accuracy(8,:),axis,accuracy(9,:),...
     axis,accuracy(10,:))
=======
load data/face_split_0.7.mat

M_pca = 200;
M_lda = 10;

% PCA goes here
PCA_bing(data,M_pca);
% NN prediction of PCA data
nTrainSamples = data('nTrainSamples');
nTestSamples = data('nTestSamples');

x_train = data('x_pca_train');
y_train = data('y_train');
x_test = data('x_pca_test');
y_test = data('y_test');

mins = zeros(1,nTestSamples);
y_pca_pred = zeros(1,nTestSamples);
for i =1:nTestSamples
    diff = x_test(:,i)*ones(1,nTrainSamples) - x_train;
    dist = vecnorm(diff);
    [val, ind] = min(dist);
    y_pca_pred(i) = y_train(ind);
end
y_pca_pred

diff = y_pca_pred - y_test;

accuracy = sum(diff==0)/double(length(diff))


% 
% % LDA goes here
% y_pred = LDA(data,M_lda);
% 
% % accuracy
% y_test = data('y_test');
% 
% diff = y_pred - y_test;
% accuracy = sum(diff==0)/length(diff)
% 
% % confusion matrix 
% confusionMat = zeros(52,52);
% 
% for i = 1:data('nTestSamples')
%     pred = y_pred(i);
%     actual = y_test(i);
%     
%     confusionMat(pred,actual) = confusionMat(pred,actual) + 1;
% end
% 
>>>>>>> 6986fb4c642db504fa4a67fc22a430e943cca502:bing_temp/PCA_LDA_run.m
