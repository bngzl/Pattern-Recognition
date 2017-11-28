%One-vs-rest svm training


models = cell(52);
testAccuracy = zeros(1,52);

for t = 1:52
    % preprocess training results
    y = zeros(1,416);
    for i = 1:8
        y((t-1)*8+i) = 1;
    end
    
   % train svm
    svm = fitcsvm(transpose(x_train),y, 'Standardize',true, 'KernelScale','auto','KernelFunction', 'rbf','OptimizeHyperparameters', 'auto');
    
    % find test accuracy
    total = 104.0;
    correct = 0;
    for i = 1:104
        
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

save("rawsvm_one_vs_rest.mat","models");
transpose(testAccuracy)
