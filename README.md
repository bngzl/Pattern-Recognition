# Pattern-Recognition
## Q1 

### Eigenfaces (10 Marks)  
a. Partition the provided face data into your training and testing data, in a way you choose.
Explain briefly the way you partitioned. Apply PCA to your training data, by computing the
eigenvectors and eigenvalues of the covariance matrix S=(1/N)AAT directly. Show and
discuss the results, including: the eigenvectors, the eigenvalues, and the mean image,
how many eigenvectors with non-zero eigenvalues are obtained and how many
eigenvectors are to be used for face recognition. Give insights and reasons behind your
answers.

b. Apply PCA to your training data, using the eigenvectors and eigenvalues of (1/N)ATA.
Show and discuss the results in comparison to the above, including: if the eigenvectors
and eigenvalues obtained are identical, what are the pros/cons of each method. Show
respective measurements for your answers.

### Application of Eigenfaces (15 Marks) 
Hereinafter, we use a more efficient PCA technique among the two methods in the above. Use
the data partition, which you used in Q1, into training and testing.

a. Perform the face image reconstruction using the PCA bases learnt. Show and discuss the
results, while varying the number of bases to use, including: if the reconstruction error (or
the distortion measure) obtained is same as in the theory, how good the reconstruction
results are for at least 3 images of your choice (e.g. from both the training and testing
dataset).

b. Perform the PCA-based face recognition by either the NN classification method or
alternative method learnt in the PCA lecture. Report and discuss, including: the recognition
accuracy (success rates), example success and failure cases, the confusion matrices,
time/memory (and any other aspects you observe), by varying the parameter
values/experimental settings you used. Give insights and reasons behind your answers.
