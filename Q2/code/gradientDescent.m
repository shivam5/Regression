function [W, J_history] = gradientDescent2(X, y, W, alpha, lambda, num_iters)
%   GRADIENTDESCENT Performs gradient descent to learn W
%   W = GRADIENTDESCENT(x, y, W, alpha, num_iters) updates W by
%   taking num_iters gradient steps with learning rate alpha

% Initializing some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp = 1 - ((alpha * lambda)/m);
row = length(W);
for iter = 1:1:num_iters
    % Performing a single gradient step on the parameter vector
    S = W;
    T = (sigmoid(X * W) - y);
    T = X' * T;
    for i = 1:row
        if(i == 1)
            W(i,1) = S(i,1) - ((alpha / m) * T(i,1));
        else
            W(i,1) = (S(i,1) * temp) - ((alpha / m) * T(i,1));
        end
    end

    % Saving the cost J in every iteration    
    J_history(iter) = computeCost1(W, X, y, lambda);
    if(J_history(iter) == 0)
        break;
    end;

end

end