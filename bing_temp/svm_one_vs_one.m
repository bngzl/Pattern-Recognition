%One-vs-rest svm training

clear;
load partitionedData.mat;

max_class = 52;

models = cell(max_class,max_class);
testAccuracy = zeros(max_class,max_class);
parpool(2);


for c1=1:max_class   
    fprintf("Parallel training, c1 = %d\n",c1);
    parfor c2 = c1+1:max_class
        fprintf("Now training for pair (%d,%d)...\n",c1,c2);
        % comparing face c1 against c2
        % c1 < c2
        % c1 will have labels 1, c2 with labels 0
        
        % preprocess training data
        x = [x_train(:,(c1-1)*8+1:(c1-1)*8+8) x_train(:,(c2-1)*8+1:(c2-1)*8+8)];
        y = [ones(1,8) zeros(1,8)];
        
        % train svm
        svm = fitcsvm(x',y, 'Standardize',true, 'KernelScale','auto','KernelFunction', 'rbf','OptimizeHyperparameters', 'auto');
        models(c1,c2) = {svm}; 
    end
end

mkdir("svm_models");
save("svm_models/svm_raw_one_vs_one.mat","models");