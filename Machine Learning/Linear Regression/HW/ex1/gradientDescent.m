function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = X * theta;
    err = h - y;
    d_theta = alpha * (1/m) * (X' * err);
    theta = theta - d_theta;

    % ============================================================
    % Save the cost J in every iteration
    j = computeCost(X, y, theta);
    % fprintf('theta = (%f, %f); j = %f\n', theta(1), theta(2), j);
    J_history(iter) = j;
end


end
