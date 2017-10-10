function out = featureTransform (X, degree)
%   FEATURETRANSFORM Feature mapping function to polynomial features
%
%   FEATURETRANSFORM(X, degree) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

X1 = X(:,1);
X2 = X(:,2);
out = ones(size(X1(:,1)));
for i = 1:degree
    for j = 0:i
        out(:, end+1) = (X1.^(i-j)).*(X2.^j);
    end
end

end