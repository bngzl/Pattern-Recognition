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