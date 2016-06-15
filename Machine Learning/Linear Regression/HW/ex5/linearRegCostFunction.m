function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
h = X * theta;
err = h - y;
J = (1/(2*m)) * (err'*err);
assert(size(J) == [1, 1],
       sprintf("J: %f\n", J));
       %sprintf('h: %f\nerr: %f\nJ: %f\nX: %f\ny: %f\n',
       %        h(:), err, J, X, y));
J = J + (lambda/(2*m)) * (theta(2:end)' * theta(2:end));
assert(size(J), [1, 1]);
% =========================================================================
grad = (1/m) .* (X' * err);
theta(1) = 0;
grad = grad + (lambda/m) .* theta;
% =========================================================================
grad = grad(:);

end

%!test
%! X = [[1 1 1]' magic(3)];
%! y = [7 6 5]';
%! theta = [0.1 0.2 0.3 0.4]';
%! lambda = 0;
%! [J grad] = linearRegCostFunction(X, y, theta, lambda);
%! assert(J, 1.3533, 1e-4);
%! assert(grad, [
%!   -1.4000
%!   -8.7333
%!   -4.3333
%!   -7.9333
%! ], 1e-4);

%!test
%! X = [1 2 3 4];
%! y = 5;
%! theta = [0.1 0.2 0.3 0.4]';
%! [J g] = linearRegCostFunction(X, y, theta, 7)
%! assert(J, 3.0150, 1e-4);
%! assert(g, [
%!   -2.0000
%!   -2.6000
%!   -3.9000
%!   -5.2000
%! ], 1e-4);
