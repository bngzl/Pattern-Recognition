function [x_train, y_train, x_test, y_test] = partitionData(X,labels,train_frac)
% Partitions data into training and tset sets
%   Splits data randomly, we should only need to do this once

test_frac = 1-train_frac;
nTrain = uint16(train_frac * 10);
nTest = uint16(test_frac * 10);


x_train = zeros(2576,nTrain*52);
y_train = zeros(1,nTrain*52);
x_test = zeros(2576,nTest*52);
y_test = zeros(1,nTest*52);

for classes = 1:52
    % for each class, take 8 random images as training data
    % and the remaining two as test data
    [ind_train, ind_val, ind_test] = dividerand(10,train_frac,0.0,test_frac);
    
    for i = 1:length(ind_train)
        x_train(:,i+(classes-1)*nTrain) = X(:,ind_train(i)+(classes-1)*10);
        y_train(i+(classes-1)*nTrain) = labels(ind_train(i)+(classes-1)*10);
    end
    for i = 1:length(ind_test)
        x_test(:,i+(classes-1)*nTest) = X(:,ind_test(i)+(classes-1)*10);
        y_test(i+(classes-1)*nTest) = labels(ind_test(i)+(classes-1)*10);
    end
    
end

save("partitionedData.mat","x_train","y_train","nTrain","x_test","y_test","nTest");

