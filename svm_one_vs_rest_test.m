%One-vs-rest svm training
clear
load svm_models/svm_raw_one_vs_rest.mat
load data/data_raw_0.7.mat

% one-vs-rest assignment based on confidence value
confidenceMat = zeros(nClass,nTest*nClass);

for c=1:nClass
    [p,s] = models{c}.predict(x_test');
    confidenceMat(c,:) = s(:,2)';
end
[v,i] = max(confidenceMat,[],1);
diff = i-y_test;
accuracy = sum(diff(i)==0)/length(diff)

