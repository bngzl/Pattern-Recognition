% Report script: 
clear all; 
load data/face.mat; 

% Creating multiple images for class 1: 
for i = 1:10
    filename = sprintf('class_1_%d', i); 
    imwrite(mat2gray(showImage(X(:,i))),[filename, '.jpg']); 
end