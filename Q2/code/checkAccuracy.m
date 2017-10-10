function [accuracy] = checkAccuracy(X, W, Y)
%   CHECKACCURACY returns accuracy of parameters learned on given dataset

m = length(X);
correct = 0;

T = sigmoid(X * W);
for i = 1:1:m
    if(T(i,1) >= 0.5)
        T(i,1) = 1;
    else
        T(i,1) = 0;
    end
    if(T(i,1) == Y(i,1))
        correct = correct + 1;
    end
end

accuracy = (correct/m)*100;

end
