%%%% Pattern Recognition  %%%%%
%%%%%% Matlab Tutorial  %%%%%%%                       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Operations with images %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% Read image
I=imread('imperial.jpg');

%% Show image
imshow(I)

%% Size?
[height, width, dim] = size(I);

%% RGB - 3 channels - Show Red Channel only
I_red(:,:,1)=I(:,:,1);
I_red(:,:,2)=zeros(height,width); % Removing Green Channel by replacing info with 0s
I_red(:,:,3)=zeros(height,width); % Removing Blue channel by replacing info with 0s 
figure(1); imshow(I_red);

%% Show green channel only
I_green(:,:,1)=uint8(zeros(height,width));
I_green(:,:,2)=uint8(I(:,:,2));
I_green(:,:,3)=uint8(zeros(height,width));
figure(2); imshow(I_green);

%% Show blue channel only
I_blue(:,:,1)=uint8(zeros(height,width));
I_blue(:,:,2)=uint8(zeros(height,width));
I_blue(:,:,3)=uint8(I(:,:,3));
figure(3); imshow(I_blue);

%% RGB to gray...
gray = sum(I,3)/3; % sum(I,3) sums along the 3rd dimension 
figure(4); imshow(gray);

%% or maybe...
I_gray = uint8(sum(I,3)/3); % imshow requires uint8 as input 
figure(5); imshow(I_gray);

%% rgb2gray - I_gray_matlab=0.299Red + 0.587Green + 0.114Blue
I_gray_matlab = rgb2gray(I); 
figure(6); imshow(I_gray_matlab);

%% Reshape function
I_vectorised = I_gray_matlab(:); % Now the image is a column vector
figure; imshow(I_vectorised);
size(I_vectorised)
I_reshaped = reshape(I_vectorised,height,width); % We transform the column vector to a matrix again
figure;imshow(I_reshaped);
size(I_reshaped)

%% Patches.... 32x32 pixels
num_patches = 20;
patch_size = 32;
[height, width, dim] = size(I_gray);
patch_height_idx = randperm(height, num_patches); % vector containing 20 random height coordinates 
patch_width_idx = randperm(width, num_patches); % vector containing 20 random width coordinates 
result_matrix = zeros(patch_size,patch_size,num_patches,'uint8'); % Initialise matrix  

for i=1:num_patches
    % Choose previously random coordinate 
    patch_y_idx = patch_height_idx(i); 
    patch_x_idx = patch_width_idx(i);      
    % Check if it exceeds the image limits 
    % i.e if cropping the height with the patch size will exceed given
    % height of image 
    if patch_y_idx + patch_size > height
        patch_y_idx = height - patch_size;
    end
    if patch_x_idx + patch_size > width
        patch_x_idx = width - patch_size;
    end 
    patch = imcrop(I_gray, [patch_x_idx, patch_y_idx, patch_size-1, patch_size-1]); 
    % I2 = imcrop(I, RECT) where RECT is a 4-element vector in form [XMIN
    % YMIN WIDTH HEIGHT] 
    figure; imshow(patch)  
    result_matrix(:,:,i) = patch;
end
close all
%% Mean?
my_mean=sum(result_matrix,3)./num_patches;
figure; imshow(uint8(my_mean));

%% Matlab mean
matlab_mean=mean(result_matrix,3); % Take the mean of all 32x32 randomly chosen patches one on top of the other 
figure; imshow(uint8(matlab_mean));

%% Test accuracy, are they the same? % Yes 
test=abs(matlab_mean - my_mean);
max(test)

%% Covariance matrix
%(Linear Algebra) S = 1/(N-1)(result_matrix-my_mean_over_patches)(result_matrix-my_mean_over_patches)'

X=reshape(result_matrix,[patch_size*patch_size,num_patches]); % Since results_matrix is a 32x32x20 matrix (3D)
% Want to convert a 3D matrix into a 2D matrix where the number of cols is
% the number of patches 
% we first transform a matrix of size patch_size x patch_size x num_patches to
% patch_size*patchsize x num_patches where each image is a column vector
mu_hat = my_mean(:)*ones(1,num_patches); % 32x32 converted into a single col vector * row vector of 1s 
% mu hat rows: 32*32 = 1024, cols: 20 
Z = double(X) - mu_hat;
R_hat = (Z*Z')/(num_patches-1);
img_my_cov = uint8(R_hat / max(R_hat(:)) * 255);

figure; imshow(img_my_cov);

%% PCA
[eig_vec, eig_val] = eig(R_hat);

%% After classification - confusion matrix
labels = [1 0 0 0;
    1 0 0 0;
    1 0 0 0;
    0 1 0 0;
    0 1 0 0;
    0 1 0 0;
    0 0 1 0;
    0 0 1 0;
    0 0 1 0;
    0 0 0 1;
    0 0 0 1;
    0 0 0 1];

predictions = [1 0 0 0;
    1 0 0 0;
    1 0 0 0;
    0 1 0 0;
    1 0 0 0; % Actual was 0100
    0 1 0 0;
    0 0 0 1; % Actual was 0010 
    0 0 1 0;
    0 0 1 0;
    0 0 0 1;
    0 0 0 1;
    0 0 0 1];

plotconfusion(labels',predictions')
% Used to show prediction rate


