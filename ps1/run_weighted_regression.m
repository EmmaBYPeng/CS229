%Q2 (d) (ii) (iii) Weighted Linear Regression
clear all;

% Load training set
xTrain = load("q2x.dat");
yTrain = load("q2y.dat");

% Include intercept
m = size(xTrain, 1);
n = size(xTrain, 2);
X = [xTrain, ones(m,1)];
theta = zeros(m,n+1);
band = [0.8, 0.1, 0.3, 2, 10];

% For different bandwidth
for k = 1:size(band,2)
  % Make prediction for each data point in range -6 to 13
  x = (-6:0.1:13)';
  m2 = size(x);
  y = zeros(m2,1);

  for j = 1:m2
    xJ = x(j) * ones(m,1);
    wJ = exp(-0.5*(band(k)^2)*((xTrain-xJ) .* (xTrain-xJ)));
    W = diag(wJ);
    % Compute theta for each point
    theta(j,:) = inv(X'*W*X) * X'*W*yTrain;
    % Compute the prediction value y for each point
    y(j) = theta(j,1)*x(j) + theta(j,2);  
  end

  figure;
  hold on;
  plot(xTrain, yTrain, 'o');
  plot(x, y, 'r');
  xlabel('x');
  ylabel('y');
  str = sprintf('Locally Weighted Regression with bandwith = %2.2f\n', band(k));
  title(str);
  legend('Data points', 'Fitted curve');
end

