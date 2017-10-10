%% ==================== CSL 603, Machine Learning - Question 2: Logistic Regression ====================

%% Initialization
clear; close all; clc;

%% Loading Data
%  The first 2 columns contain value for attributes
%  The third column contains the label

%  Training Data
data = load('../credit.txt');

X = data(:, [1,2]); 
Y = data(:, 3);

%  Test Data
test_data = load('../credit2.txt');

test_X = test_data(:, [1,2]); 
test_Y = test_data(:, 3);

%% ==================== Part 1: Plotting ====================

fprintf('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples\n');

plotData(X, Y);
 
hold on;
% Labels and Legend
xlabel('Attribute x1 value')
ylabel('Attribute x2 value')

legend('Credit Card Issued', 'Credit Card Not Issued')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 2: Regularized Logistic Regression ====================

%% Implementing Regularized Logistic Regression using Gradient Decent
fprintf('\nRegularized Logistic Regression using Gradient Decent\n\n');
%  Setting the data matrix appropriately, by adding ones for the intercept term
[m, n] = size(X);
X = [ones(m, 1) X];

%  Initializing fitting parameters
W = zeros(n + 1, 1);
%  Setting regularization parameter
lambda = 10;
%  Initializing learning rate
alpha = 0.1;
%  Upper bounding the number of iterations
iterations = 5000;

[W, J] = gradientDescent(X, Y, W, alpha, lambda, iterations); 
 
% Printing Resulting Values
fprintf('Cost at W found by Gradient Descent: %f\n', J(iterations));
fprintf('W: \n');
fprintf(' %f \n', W);

%% Checking Accuracy

%  Training Accuracy
training_accuracy = checkAccuracy(X,W,Y);
m1 = length(test_X);
%  Setting the test data matrix appropriately, by adding ones for the intercept term
test_X = [ones(m1, 1) test_X];
%  Test Accuracy
test_accuracy = checkAccuracy(test_X,W,test_Y);

fprintf('\nAccuracy on Training Data = %f\n', training_accuracy);
fprintf('Accuracy on Test Data = %f\n', test_accuracy);

%% Plotting Boundary
plotDecisionBoundary(W, X, Y);
hold on;
title(sprintf('Decision Boundary using Gradient Descent'))

% Labels and Legend
xlabel('Attribute x1 value')
ylabel('Attribute x2 value')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% Implementing Regularized Logistic Regression using Newton Raphson

fprintf('\nRegularized Logistic Regression using Newton Raphson\n\n');
%  Initializing fitting parameters
W = zeros(n + 1, 1);
%  Setting regularization parameter
lambda = 1;
%  Upper bounding the number of iterations
iterations = 5000;

[W, J] = NewtonRaphson(X, Y, W, lambda, iterations); 
 
% Printing Resulting Values
fprintf('Cost at W found by Newton Raphson Method: %f\n', J(iterations));
fprintf('W: \n');
fprintf(' %f \n', W);

%% Checking Accuracy

%  Training Accuracy
training_accuracy = checkAccuracy(X,W,Y);

%  Test Accuracy
test_accuracy = checkAccuracy(test_X,W,test_Y);

fprintf('\nAccuracy on Training Data = %f\n', training_accuracy);
fprintf('Accuracy on Test Data = %f\n', test_accuracy);

%% Plotting Boundary
plotDecisionBoundary(W, X, Y);
hold on;
title(sprintf('Decision Boundary using Newton Raphson Method'))

% Labels and Legend
xlabel('Attribute x1 value')
ylabel('Attribute x2 value')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 3: Feature Transformation ====================
fprintf('\nRegularized Logistic Regression after Feature Transformation using Newton Raphson\n\n');

%  Degree = 2
degree = 2;
fprintf('Feature Transformation using Degree = %d\n\n', degree);
logisticRegression(degree);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Degree = 3
degree = 3;
fprintf('\nFeature Transformation using Degree = %d\n\n', degree);
logisticRegression(degree);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Degree = 4
degree = 4;
fprintf('\nFeature Transformation using Degree = %d\n\n', degree);
logisticRegression(degree);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ==================== Part 4: Varying Lambda ====================

%  Overfitting
lambda = 0;
degree = 2;
fprintf('\nShowing Overfitting using lambda = %d\n', lambda);
Overfitting(lambda, degree);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Underfitting

lambda = 20;
degree = 2;
fprintf('\nShowing Underfitting using lambda = %d\n', lambda);
Overfitting(lambda, degree);