% Q1 (b)
clear all;

% Compute gradient and Hessian of log likelihood function
function [dL, H] = computederivatives(theta, xTrain, yTrain)
  m = size(xTrain, 1);
  X = [xTrain, ones(m,1)];
  % Output of sigmoid function
  h = ones(1,m)' ./ (1 + exp((-1)*(X*theta)));
  % Gradient
  dL = X' * (yTrain - h);
  % Hessian
  g = (h - 1) .* h;
  W = diag(g);
  H = X' * W * X;
end

% Load training set
xTrain = load("q1x.dat");
yTrain = load("q1y.dat");

m = size(xTrain, 1);
dimension = size(xTrain, 2);

% Initialize theta
theta = zeros(dimension+1, 1);
threshold = 1e-6 * ones(dimension+1, 1);
diff = ones(dimension+1, 1);

% User Newton's method to update theta until converge
while (diff > threshold)
  [dL, H] = computederivatives(theta, xTrain, yTrain);
  oldTheta = theta;
  theta = theta - inv(H)*dL;
  diff = abs(theta - oldTheta);
end

% Separate training dat into class 1 and 0
label1Index = find(yTrain);
xTrain1 = xTrain(label1Index(1):label1Index(size(label1Index,1)), :);

label0Index = find(ones(m,1) - yTrain);
xTrain0 = xTrain(label0Index(1):label0Index(size(label0Index,1)), :);

% Decision boundary
x = linspace(0,10);
y = -theta(1)/theta(2)*x - theta(3)/theta(2);

hold on;
plot(xTrain1(:,1), xTrain1(:,2), '*');
plot(xTrain0(:,1), xTrain0(:,2), 'o');
plot(x, y, 'r');
xlabel('x1');
ylabel('y2');
title('Logistic Regression');
legend('Data points in class 1', 'Data points in class 2', 'Fitted curve');



