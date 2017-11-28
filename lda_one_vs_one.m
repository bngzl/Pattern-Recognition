clear;
load data/data_raw_0.7.mat

means = zeros(nFeatures,nClass);
Sclass = zeros(nFeatures,nFeatures,nClass);

% computing mean of classes
for c=1:nClass
    means(:,c) = x_train(:,(c-1)*nTrain+1:(c-1)*nTrain+nTrain)*ones(nTrain,1)./ double(nTrain);
end

% computing scatter matrices of classes
for c= 1:nClass
    for i = 1:nTrain
        diff = x_train(:,(c-1)*nTrain+i)-means(:,c);
        Sclass(:,:,c) = Sclass(:,:,c) +diff*diff';
    end
end

Ws = zeros(nClass,nClass,nFeatures);

% for each pair (c1,c2)
for c1 = 1:nClass
    for c2 = c1+1:nClass
        Sw = Sclass(:,:,c1) + Sclass(:,:,c2);               % Within-class scatter matrix
%         Sb = (means(c1)-means(c2))*(means(c1)-means(c2))';  % Between-class scatter matrix
        w = inv(Sw)*(means(:,c1)-means(:,c2));
        Ws(c1,c2,:) = w./norm(w);
    end
end