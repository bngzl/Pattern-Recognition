%One-vs-rest svm training
clear
load data/data_raw_0.7.mat

models = cell(1,nClass);
testAccuracy = zeros(1,nClass);

parpool(2);

parfor t = 1:nClass
    fprintf("Training SVM for face %d\n",t);
    % preprocess training results
    y = zeros(1,nTrain*nClass);
    for i = 1:nTrain
        y((t-1)*nTrain+i) = 1;
    end
    
   % train svm
    svm = fitcsvm(transpose(x_train),y, 'Standardize',true, 'KernelScale','auto','KernelFunction', 'rbf','OptimizeHyperparameters', 'auto');
    
    % find test accuracy
    total = nClass*nTest;
    correct = 0.0;
    for i = 1:total
        
        y_true = 0;
        y_pred = svm.predict(transpose(x_test(:,i)));
        if (i == 2*t-1) || (i==2*t)
            y_true = 1;
        end
        
        if y_pred == y_true
            correct = correct + 1;
        end
    end
    
    testAccuracy(t) = correct/total;
    models(t) = {svm};
end

mkdir('svm_models');
save('svm_models/svm_raw_one_vs_rest.mat','models');

% one-vs-rest assignment based on confidence value
% confidenceMat = zeros(nClass,nTest*nClass);
% 
% for c=1:nClass
%     [p,s] = models{c}.predict(x_test');
%     confidenceMat(c,:) = s(2)';
% end
% [v,i] = max(confidenceMat,[],1);
% y_pred = i';


