function [y_lda_pred, Sw, Sb] = LDA(data, M_lda)
% Calculates the accuracy of LDA after projecting onto M_lda eigenvectors
% Takes in the data container (hint: type in 'data.keys')
    
    % Unpack variables from data
    if isKey(data, 'x_pca_train')
        x_train = data('x_pca_train');
        x_test = data('x_pca_test');    
        nFeatures = data('M_pca');
        fprintf("Using PCA transformed data (M_pca = %d)\n", nFeatures)
    else 
        x_train = data('x_train');
        x_test = data('x_test');
        nFeatures = data('nFeatures');
        fprintf("Using original face data...\n")
    end
    y_train = data('y_train');
    y_test = data('y_test');
    
    nClass = data('nClass');
    nTrain = data('nTrain');
    nTrainSamples = data('nTrainSamples');
    nTestSamples = data('nTestSamples');

    % Initialize matrices
    means = zeros(nFeatures,nClass);
    Sw = zeros(nFeatures,nFeatures);
    Sb = zeros(nFeatures,nFeatures);

    % Computing mean of classes
    for c=1:nClass
        means(:,c) = x_train(:,(c-1)*nTrain+1:(c-1)*nTrain+nTrain)* ...
            ones(nTrain,1)./ double(nTrain);
    end
    mean = x_train*ones(nTrainSamples,1)./double(nTrainSamples);

    % computing Within-class scatter matrix
    for c= 1:nClass
        for i = 1:nTrain
            diff = x_train(:,(c-1)*nTrain+i)-means(:,c);
            Sw = Sw + diff*diff';
        end
    end

    %Computing Between-class scatter matrix
    for c = 1:nClass
       Sb = Sb + (means(:,c)-mean)*(means(:,c)-mean)';
    end
    [Ws, vals] = eig(inv(Sw)*Sb);

    % Obtain projection matrix from M_lda eigenvectors
    % based on M_lda eigenvalues
    P = Ws(:,1:M_lda)';

    % Transform feature set
    x_lda_train = P*x_train;
    x_lda_test = P*x_test;

    % NN computation, comparing each test sample with each train sample
    % Euclidean distance metric
    mins = zeros(1,nTestSamples);
    y_lda_pred = zeros(1,nTestSamples);
    for i = 1:nTestSamples
        curr_min = Inf;
        curr_pred = 0;
        for j = 1:nTrainSamples
            dist = norm(x_lda_test(i)-x_lda_train(j));
            if dist < curr_min
                curr_min = dist;
                curr_pred = y_train(j);
            end
        end
        mins(i) = curr_min;
        y_lda_pred(i) = curr_pred;
    end
    
end