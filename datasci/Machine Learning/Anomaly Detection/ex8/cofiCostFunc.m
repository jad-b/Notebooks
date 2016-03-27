function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%
% =============================================================
% Raw predictions
preds = (X*Theta');
% Error between predictions and actual values
err = preds - Y .* R;
% Remove predictions without data to compare them to
% Cost function
J = sum(sum(err.^2)) / 2;
% Regularized cost
J = J + (lambda/2) * (sum(sum((Theta.^2))) + sum(sum(X.^2)));
%
X_grad = err * Theta .+ (X .* lambda);
Theta_grad = err' * X .+ (Theta .* lambda);

grad = [X_grad(:); Theta_grad(:)];

end

%!test
%! params = [ 1:14 ] / 10;
%! Y = magic(4);
%! Y = Y(:,1:3);
%! R = [1 0 1; 1 1 1; 0 0 1; 1 1 0] > 0.5;     % R is logical
%! num_users = 3;
%! num_movies = 4;
%! num_features = 2;
%! lambda = 0;
%! J = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda);
%! assert(J, 311.63);
