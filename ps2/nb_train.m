
clear all;
[spmatrix, tokenlist, trainCategory] = readMatrix('MATRIX.TRAIN');

trainMatrix = full(spmatrix);
numTrainDocs = size(trainMatrix, 1);
numTokens = size(trainMatrix, 2);

% YOUR CODE HERE

numSpam = nnz(trainCategory);
numNotSpam = numTrainDocs - numSpam;

% Compute parameter vectors
trainCategoryRev = ones(1,numTrainDocs)-trainCategory;

% Denominators for phi_1 and phi_0 
denom_1 = trainCategory * (trainMatrix * ones(numTokens,1)) + numTokens;
denom_0 = trainCategoryRev * (trainMatrix * ones(numTokens,1)) + numTokens;

% phi_1(k) = p(xj = k|y = 1) (for any j)
phi_1 = ((trainCategory * trainMatrix)' + 1) ./ denom_1;
% phi_0(k) = p(xj = k|y = 0) (for any j)
phi_0 = ((trainCategoryRev * trainMatrix)' + 1) ./ denom_0;
% phi_y_1 = p(y = 1)
phi_y_1 = numSpam / numTrainDocs;

% Find the top five tokens
prob = log(phi_1) - log(phi_0);
[probSorted, indices] = sort(prob, 'descend');
top5 = indices(1:5);
