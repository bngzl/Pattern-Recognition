function [] = showImage(c)
% converts a column vector into an appropriate greyscale image format
% image is a 56 x 46 matrix
img = reshape(c,[56,46]);
imshow(mat2gray(img))
end

% face.mat data: 
% X contains all the face data
% - Each column is the data of 1 picture (2576 = 56 x 46) 
% l contains all the label data 
% - 52 faces, 10 samples each 
% showImage(X(:,column)) shows the image in each column 
