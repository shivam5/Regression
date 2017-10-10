function [J, grad] = computeCost2(theta, X, y, lambda)

%   COMPUTECOST1 Computes cost for logistic regression with regularization
%   J = COSTFUNCTIONREG(W, X, y, lambda) computes the cost of using
%   W as the parameter for regularized logistic regression

hx = sigmoid(X * theta);
m = length(X);

J = (sum(-y' * log(hx) - (1 - y')*log(1 - hx)) / m) + lambda * sum(theta(2:end).^2) / (2*m);
grad =((hx - y)' * X / m)' + lambda .* theta .* [0; ones(length(theta)-1, 1)] ./ m ;

end
