function [W, J_history] = NewtonRaphson2(X, y, W, lambda, num_iters)
%   Newton Raphson Performs optimization to learn W
%   W = NEWTONRAPHSON(x, y, W, alpha, num_iters) updates W by
%   taking num_iters steps with learning rate alpha

% Initializing some useful values
[m,n] = size(X); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:1:num_iters
    % Performing a single gradient step on the parameter vector
    S = W;
    T = sigmoid(X * W);
    P = T - y;
    R = zeros(m,m);
    % Defining R = N X N Diagnol Matrix
    for i = 1:m
        R(i,i) = T(i,1) * (1 - T(i,1));
    end
    H = (X' * R * X) + (lambda/m) * eye(n);
    W = S - pinv(H)*((X' * P) + (lambda/m) * S);  
  
    % Saving the cost J in every iteration    
    J_history(iter) = computeCost1(W, X, y, lambda);
    if(J_history(iter) == 0)
        break;
    end;

end

end