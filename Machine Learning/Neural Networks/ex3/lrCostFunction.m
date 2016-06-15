function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
[m n] = size(X); % # of training samples, # of features
assert(size(y), [m 1]); % m x 1;
assert(size(theta), [n 1]); % n x 1

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));
onez = ones(size(y));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%

% h: m x 1; predictions for x using weights theta
h = sigmoid(X * theta);
assert(size(h), [m, 1]);
% y_pos: 1 x 1; cost when y_actual is 1
y_pos = -y'*log(h);
assert(size(y_pos), [1 1]);
% y_neg: 1 x 1; cost when y_actual is 0
y_neg = (onez-y)'*log(onez-h);
assert(size(y_neg), [1 1]);
% J: 1 x 1; scalar
J = 1/m .* (y_pos - y_neg);
assert(size(J), [1 1]);

% Hint: When computing the gradient of the regularized cost function,
%       there are many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%
grad = 1/m .* (X' * (h-y));

% Now the regularization
theta(1) = 0; % Leave out the bias weight
J = J + lambda/(2*m)*(theta'*theta);
grad = grad + (lambda/m) .* theta;


% =============================================================

% Convert to column vector
grad = grad(:);

end

%!test
%! theta = [-2; -1; 1; 2];
%! X = [ones(3,1) magic(3)];
%! y = [1; 0; 1] >= 0.5;       % creates a logical array
%! lambda = 3;
%! [J grad] = lrCostFunction(theta, X, y, lambda);
%! assert(J, 7.6832, 10^-4);
%! assert(grad, [0.31722; -0.12768; 2.64812; 4.23787], 10^-5);
