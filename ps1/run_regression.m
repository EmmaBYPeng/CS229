% Q2 (d)(i) Unweighted Linear Regression
clear all;

% Load training set
xTrain = load("q2x.dat");
yTrain = load("q2y.dat");

% Include intercept
m = size(xTrain, 1);
X = [xTrain, ones(m,1)];

% Theta for linear regression
theta = inv(X' * X) * X' * yTrain;

x = linspace(-10,15);
y = theta(1) * x + theta(2);

hold on;
plot(xTrain, yTrain, 'o');
plot(x, y, 'r');


