function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Get unregularized cost and gradient
[J, grad] = costFunction(theta, X, y);
% Zero out theta_0
theta(1) = 0;
% Add regularization value to original cost
J = J + lambda/(2*m)*(theta'*theta);

% n x 1 vector
reg_grad = lambda/m * theta;
grad = grad + reg_grad;
% =============================================================

end
