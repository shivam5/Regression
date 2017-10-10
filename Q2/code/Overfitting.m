function [] = Overfitting(lambda, degree)

%% Loading Data
data = load('../credit.txt');

X = data(:, [1,2]); 
Y = data(:, 3);

%% Regularized Logistic Regression using Newton Raphson

%  Transforming Features
X = featureTransform(X, degree);

%  Initializing fitting parameters
theta = zeros(size(X,2), 1);
%  Upper bounding the number of iterations
iterations = 7;

[theta, J] = NewtonRaphson(X, Y, theta, lambda, iterations); 
 
%% Plotting Decision Boundary

% Plot Boundary
plotDecisionBoundary(theta, X, Y, degree);
hold on;
title(sprintf('Lambda = %d',lambda))

% Labels and Legend
xlabel('Attribute x1 value')
ylabel('Attribute x2 value')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

end