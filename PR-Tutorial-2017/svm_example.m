%%%% Pattern Recognition  %%%%%
%%%%%% Matlab Tutorial  %%%%%%%                       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% SVM example %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% More details about this library can be found in LIBSVM FAQ
% (http://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html) and LIBSVM
% implementation document
% (http://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf).

%%
clear all
close all
clc

addpath(genpath('libsvm'))

%% Read data
[heart_scale_label, heart_scale_inst] = libsvmread('heart_scale');

%% Access data and understand the "nature"
vector_1 = heart_scale_inst(1,:);
vector_2 = heart_scale_inst(2,:);

vector_1 = full (vector_1)';
vector_2 = full (vector_2)';
train_set = [vector_1 vector_2];

%% Split Data Cross validation - Spliting data to training - validation - testing sets...
train_data = heart_scale_inst(1:150,:);
train_label = heart_scale_label(1:150,:);
test_data = heart_scale_inst(151:270,:);
test_label = heart_scale_label(151:270,:);

%% Train model
% Returned Model Structure
% ========================
% 
% The 'svmtrain' function returns a model which can be used for future
% prediction.  It is a structure and is organized as [Parameters, nr_class,
% totalSV, rho, Label, ProbA, ProbB, nSV, sv_coef, SVs]:
% 
%         -Parameters: parameters
%         -nr_class: number of classes; = 2 for regression/one-class svm
%         -totalSV: total #SV
%         -rho: -b of the decision function(s) wx+b
%         -Label: label of each class; empty for regression/one-class SVM
%         -ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
%         -ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
%         -nSV: number of SVs for each class; empty for regression/one-class SVM
%         -sv_coef: coefficients for SVs in decision functions
%         -SVs: support vectors
% 
% If you do not use the option '-b 1', ProbA and ProbB are empty
% matrices. If the '-v' option is specified, cross validation is
% conducted and the returned model is just a scalar: cross-validation
% accuracy for classification and mean-squared error for regression.


model_linear = svmtrain(train_label, train_data, '-t 0');
model_precomputed = svmtrain(train_label, [(1:150)', train_data*train_data'], '-t 4');

%% Prediction
% The function 'svmpredict' has three outputs. The first one,
% predictd_label, is a vector of predicted labels. The second output,
% accuracy, is a vector including accuracy (for classification), mean
% squared error, and squared correlation coefficient (for regression).
% The third is a matrix containing decision values or probability
% estimates (if '-b 1' is specified). If k is the number of classes, for decision values, 
% each row includes results of predicting k(k-1/2) binary-class SVMs. For probabilities, 
% each row contains k values indicating the probability that the testing instance is in
% each class. Note that the order of classes here is the same as 'Label'
% field in the model structure.


[predict_label_L, accuracy_L, dec_values_L] = svmpredict(test_label, test_data, model_linear);
[predict_label_P, accuracy_P, dec_values_P] = svmpredict(test_label, [(1:120)', test_data*train_data'], model_precomputed);

accuracy_L
accuracy_P