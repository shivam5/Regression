function [J] = computeCost1(W, X, y, lambda)

%   COMPUTECOST1 Computes cost for logistic regression with regularization
%   J = COSTFUNCTIONREG(W, X, y, lambda) computes the cost of using
%   W as the parameter for regularized logistic regression 
hx = sigmoid(X * W);
kx = 1-hx;

len = length(hx);
for i = 1:len
    if (hx(i,1) ~= 0)
        hx(i,1) = log(hx(i,1));
    end
    if (kx(i,1) ~= 0)
        kx(i,1) = log(kx(i,1));
    end
end
%disp(hx);
m = length(X);

J = (sum(-y' * hx - (1 - y')*kx) / m) + lambda * sum(W(2:end).^2) / (2*m);
%grad =((hx - y)' * X / m)' + lambda .* W .* [0; ones(length(W)-1, 1)] ./ m ;
%%
% J = 0;
% temp_theta = [];
% m = length(Y);
% 
% %get the regularization term
% 
% for jj = 2:length(W)
%     temp_theta(jj) = W(jj)^2;
% end
% 
% theta_reg = lambda/(2*m)*sum(temp_theta);
% 
% temp_sum =[];
% 
% %for the sum in the cost function
% 
% for ii =1:m
%    temp_sum(ii) = -Y(ii)*log(sigmoid(W'*X(ii,:)'))-(1-Y(ii))*log(1-sigmoid(W'*X(ii,:)'));
% end
% 
% tempo = sum(temp_sum);
% 
% J = (1/m)*tempo+theta_reg;

%% Computing Cost
% m = length(Y); % Number of training examples
% H = sigmoid(X*W);
% K = ones(size(H));
% K = K - H;
% Z = ones(m,1);
% Z = Z-Y;
% L = log(H);
% M = log(K);
% 
% J = (Y'*L) + (Z'*M);
% J = J/(2*m);
% J = -J;
% 
% temp = 0;
% for i = 2:size(W)
%     temp = temp + (W(i,1)*W(i,1));
% end
% 
% J = J + (temp*(lambda/(2*m)));

%%
% H = sigmoid(X*W); % Computing sigmoid function of X * W
% K = ones(size(H));
% K = K - H;
% 
% H = log(H);
% K = log(K);
% 
% Z = ones(size(Y));
% Z = Z - Y;
% 
% J = (Y' * H) + (Z' * K);
% 
% J = J / (2*m);
% J = J * (-1);
% 
% sum = 0;
% for i = 2:1:length(W)
%     sum = sum + (W(i,1)*W(i,1));
% end
% 
% J = J + ((sum*lambda)/(2*m));

end