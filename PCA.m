% Question 1 
% Computing the average face vector 
N = 52*8; 
sum_x_train = x_train * ones([416,1]); 
ave_x_train = sum_x_train.*(1/N); 

showImage(ave_x_train(:,1)) 
