clear;
load data/data_raw_0.7.mat

means = zeros(nFeatures,nClass);
Sw = zeros(nFeatures,nFeatures);
Sb = zeros(nFeatures,nFeatures);

% computing mean of classes
for c=1:nClass
    means(:,c) = x_train(:,(c-1)*nTrain+1:(c-1)*nTrain+nTrain)*ones(nTrain,1)./ double(nTrain);
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