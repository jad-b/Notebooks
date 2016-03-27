function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
options = optimset('GradObj', 'on', 'MaxIter', 50);

for cls = 1:num_labels
    initial_theta = zeros(n+1, 1);
    theta = fmincg(@(t)(lrCostFunction(t, X, (y==cls), lambda)), ...
                          initial_theta, options);

    all_theta(cls, :) = theta(:)';
end
end


%!test
%! X = [magic(3) ; sin(1:3); cos(1:3)];
%! y = [1; 2; 2; 1; 3];
%! num_labels = 3;
%! lambda = 0.1;
%! [all_theta] = oneVsAll(X, y, num_labels, lambda);
%! exp_all_theta = [
%!   -0.559478   0.619220  -0.550361  -0.093502
%!   -5.472920  -0.471565   1.261046   0.634767
%!    0.068368  -0.375582  -1.652262  -1.410138
%! ];
%! %assert(all_theta, exp_all_theta, 10^-6);
