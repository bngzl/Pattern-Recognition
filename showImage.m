function [] = showImage(c)
% converts a column vector into an appropriate greyscale image format
% image is a 56 x 46 matrix
img = reshape(c,[56,46]);
imshow(mat2gray(img))
end

