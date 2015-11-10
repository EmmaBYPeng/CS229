addpath('liblinear-1.7/matlab');  % add LIBLINEAR to the path
[sparseTestMatrix, tokenlist, testCategory] = readMatrix('MATRIX.TEST');

numTestDocs = size(sparseTestMatrix, 1);
numTokens = size(sparseTestMatrix, 2);

output = zeros(numTestDocs, 1);

%---------------
% YOUR CODE HERE
for i = 1:numTestDocs
    if (testCategory(i) == 0)
      testCategory(i) = -1;
    end
end

[output, accuracy, prob] = predict(testCategory', sparseTestMatrix, model);
%---------------


% Compute the error on the test set
error=0;
for i=1:numTestDocs
  if (testCategory(i) ~= output(i))
    error=error+1;
  end
end

%Print out the classification error on the test set
error/numTestDocs

% Error for each training set
trainNums = [50,100,200,400,800,1400];
error = [0.0525,0.0288,0.0125,0.0150,0.0125,0.0100];

plot(trainNums, error);
xlabel('Training set size');
ylabel('Error');
