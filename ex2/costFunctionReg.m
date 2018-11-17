function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

l = length(theta);
h = sigmoid(X * theta);

% Calcurate cost without regulation as first.
J = 1/m * (-y' * log(h) - (1 - y)' * log(1 - h )); 
% Add regulation term for theta_j (j > 2)
J = J + sum(lambda / (2 * m) * theta(2:l).^2);

% Calcurate gradient without regulation as first.
grad = 1/m * (h - y)' * X; % Current size of grad is 1 x 28
grad = grad'; % Identify grad size to theta (28 x 1)
% Add regulation term for theta_j (j > 2)
grad(2:l) = grad(2:l) + lambda / m * theta(2:l);

% =============================================================

end
