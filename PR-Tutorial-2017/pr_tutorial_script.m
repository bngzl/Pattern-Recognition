%%%% Pattern Recognition  %%%%%
%%%%%% Matlab Tutorial  %%%%%%%                       

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Basics of Matlab %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Run line by line and try to understand the commands
% Tip: try using Ctrl+Enter to run different sections

%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% VECTORS %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc

%% define a 2-dimensional vector
x = [9 4];

%% define a 6-dimensioanl vector
y = [2 4 6 2 1 5]

%% transpose of a vector
y'

%% create vector from a pre-defined pattern
z = [1:5]

%% or
w = [1:0.5:4]

%% access specific element of vector
w(3)
z(3)

%% create nameless vector
0:1:10

%% transpose it
ans'

%% or
(0:1:10)'

%% access sub-vector
w(1:3)

%% or
w(1:3:6)

%% perform operations
w(1:3)-w(2:4)

%% stepping back
y = [0:-1:-6]

%% w + y
w+y

%% -2*y
-2*y

%% w/3
w/3


%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% MATRICES %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
%clear all
%close all
%clc

%% Define 3x3 matrix
A = [ 1 2 3; 3 4 5; 6 7 8]

%% Define 3x3 matrix as combination of vectors
B = [[1 4 7]' [2 5 8]' [3 6 9]']

%% You've lost track?
whos

%% Multiplying
x = [1:3:9]
A*x

%% But
A*x'

%% Slicing
A(1:2,3:4)

%% But
A(1:2,2:3) % Rows 1 to 2 and Cols 2 to 3 of Matrix A
 
%% Transpose
A(1:2,2:3)'

%% Access row and columns
A(1,:) % Row 1 and all the cols of Matrix A
A(:,1)

%% Diagonal
diag(A)

%% create random 3x3 matrix
A = rand(3,3)

%% Determinant
det(A)

%% Furthermore Bx=v find x
v = [1 3 5]'
B = [[1 3 5]' [2 4 8]' [4 6 8]']

x=B\v %%solves Bx=v
B*x

%% or?...
x1 = v'/B %%solves xB=v
x1*B

%%%%%%%%%%%%%%%%%%%%%
%%%%%%% PLOTS %%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% example
x = 0:2*pi/100:2*pi;
y = sin(x);
z = cos(x);
size(y)
max(size(y))
y(1)
plot(x,y)
grid on

%% example subplot
subplot(1,2,1);
plot(x,y)
grid on;
subplot(1,2,2);
plot(x,z)
grid on;

%% print result (try other formats!)
print -dps plot.ps

%% more examples - contour
[c,h] = contour(peaks); clabel(c,h), colorbar

%% more examples - histogram

x = -4:0.1:4; 
y = randn(1000000,1); 
hist(y,x)

%% more examples - surface 3D plot
[X,Y] = meshgrid(-2:.2:2, -2:.2:2);
Z = X .* exp(-X.^2 - Y.^2);
surf(X,Y,Z)


%%%%%%%%%%%%%%%%%%%%%
%%%%%%% LOOPS %%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% for
for j=1:4
    j
end

%% or
v = [1:3:10]

for j=1:3
    v(j)=3*j;
end
v

%% operations on the rows of a matrix
A = [ [1 2 3]' [3 2 1]' [2 1 3]']
B = A;
for j=2:3
    A(j,:) = A(j,:) - A(j-1,:) % 2nd row - 1st row, 3rd row - resultant 2nd row 
end

%% more examples
for j=2:3,
    for i=j:3,
        B(i,:) = B(i,:) - B(j-1,:)*B(i,j-1)/B(j-1,j-1)
    end
end

%% while loops
k = 100
while (k < 101) && (k>50)
    k = k / 1.5
end

%% vectorize always
N = 1000;
A = rand(N);
tic() % start time counter
d1 = diff(A);
t1 = toc() % stop timer counter
tic()
for i = 1:N-1
    d2(i,:) = A(i+1,:) - A(i,:); % Differencing by col vectors
end
t2 = toc()
fprintf(1,'delay: %f%%\n', 100*t2/t1);

tic()
for i = 1:N-1
    for j = 1:N
        d3(i,j) = A(i+1,j) - A(i,j); % Differencing by 2 for loops
    end
end
t3 = toc()
fprintf(1,'Delay: %f%%\n', 100*t3/t1);

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% FUNCTIONS %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% first create and then call
myfunc([-1 1 2 3])

r = myfunc([-1 1 2 3])

[r1, r2] = myfunc([-1 1 2 3])

%%%%%%%%%%%%%%%%%%%%
%%%%%%% DATA %%%%%%%
%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% saving
x = [1 2 3];
y = [6 5 4];
whos
save vars.mat
dir

%% loading
clear
whos
load vars.mat
whos

%% loading whatever you want
clear
whos
load vars.mat x % Loading only variable x 
whos

%% deleting
delete vars.mat
dir

%% save session
diary save.txt % From start diary to next diary, all the command window info will be saved in the file defined
r = 5;
z = 2;
diary % Must remember to close it

%%%%%%%%%%%%%%%%%%%%%
%%%%%%% FILES %%%%%%%
%%%%%%%%%%%%%%%%%%%%%
%% clear all, close all, clc
clear all
close all
clc

%% create temp.log offline and then
fp = fopen('temp.log','r');
tmp_10 = fscanf(fp,'%d,%d',10);% read 10 items
tmp_all = fscanf(fp, '%d,%d') %read all items
fclose(fp)
%% Structs... sometimes they are useful
student(1).Name = 'Sarah';
student(1).Year = 4;
student(1).Course = 'Pattern Recognition';
student(1).Final_Marks = [85 82 78;  % courseworks
                          80 81 80]; % interviews

student(2).Name = 'Mark';
student(2).Year = 4;
student(2).Course = 'Pattern Recognition';
student(2).Final_Marks = [45 52 58;  % courseworks
                          39 41 40]; % interviews


student(3).Name = 'Guillermo';
student(3).Year = 4;
student(3).Course = 'Pattern Recognition';
student(3).Final_Marks = [99 99 45;  % courseworks
                          20 80 99]; % interviews


figure; bar(student(1).Final_Marks); title(['Coursework and Interview marks for ', student(1).Name]);
figure; bar(student(2).Final_Marks); title(['Coursework and Interview marks for ', student(2).Name]);
figure; bar(student(3).Final_Marks); title(['Coursework and Interview marks for ', student(3).Name]);


figure; subplot(1,3,1); bar(student(1).Final_Marks); title(['Coursework and Interview marks for ', student(1).Name]);
subplot(1,3,2); bar(student(2).Final_Marks); title(['Coursework and Interview marks for ', student(2).Name]);
subplot(1,3,3); bar(student(3).Final_Marks); title(['Coursework and Interview marks for ', student(3).Name]);

average_courseworks_marks_interviews = student(1).Final_Marks;
average_courseworks_marks_interviews = [average_courseworks_marks_interviews student(2:end).Final_Marks];

mean(average_courseworks_marks_interviews,2)
