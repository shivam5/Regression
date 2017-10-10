function [] = logisticRegression(degree)

%% Loading Data
data = load('../credit.txt');

X = data(:, [1,2]); 
Y = data(:, 3);

%  Test Data
test_data = load('../credit2.txt');

test_X = test_data(:, [1,2]); 
test_Y = test_data(:, 3);

%% Regularized Logistic Regression using Newton Raphson

%  Transforming Features
X = featureTransform(X, degree);
test_X = featureTransform(test_X, degree);

%  Initializing fitting parameters
theta = zeros(size(X,2), 1);
%  Setting regularization parameter
lambda = 0;
%  Upper bounding the number of iterations
iterations = 7;

[theta, J] = NewtonRaphson(X, Y, theta, lambda, iterations); 
 
% Printing Resulting Values
fprintf('Cost at W found by Newton Raphson Method: %f\n', J(iterations));
fprintf('W: \n');
fprintf(' %f \n', theta);

%% Checking Accuracy

%  Training Accuracy
training_accuracy = checkAccuracy(X,theta,Y);

%  Test Accuracy
test_accuracy = checkAccuracy(test_X,theta,test_Y);

fprintf('\nAccuracy on Training Data = %f\n', training_accuracy);
fprintf('Accuracy on Test Data = %f\n', test_accuracy);

%% Plotting Decision Boundary

% Plot Boundary
plotDecisionBoundary(theta, X, Y, degree);
hold on;
title(sprintf('Degree = %d',degree))

% Labels and Legend
xlabel('Attribute x1 value')
ylabel('Attribute x2 value')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;
