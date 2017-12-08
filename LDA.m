function [P, Sw, Sb] = LDA(x_train, nTrain, M_lda)
% Calculates the accuracy of LDA after projecting onto M_lda eigenvectors
% Takes in the data container (hint: type in 'data.keys')

    % Initialize matrices
    [nFeatures, nTrainSamples] = size(x_train);
    nClass = 52;
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

    % Obtain projection matrix from M_lda eigenvectors
    % based on M_lda eigenvalues
    [P, vals] = eigs(inv(Sw)*Sb,M_lda);

end