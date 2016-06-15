function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels
%are in the range 1..K, where K = size(all_theta, 1).
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples)

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the
%       max element, for more information see 'help max'. If your examples
%       are in rows, then, you can use max(A, [], 2) to obtain the max
%       for each row.
%

% all_theta is a num_labels x n matrix, where each row is the trained weights
% for the classifier. The class matches the classifier's row index, i.e. the
% classifier for the digit '3' lives at index 3 (the classifier for '10' lives
% at index 0).

% We can run the samples through every classifier through matrix
% multiplication.
% X: m x n
% all_theta: num_labels x n
% predictions: m x num_labels
predictions = X * all_theta';  %: row vectors of P(x_i|y==label)
assert(size(predictions), [m, num_labels]);

% Now we can choose the highest probability from each row to represent the
% digit. For a binary classifier, where the only outcomes are True|False, we'd
% use a threshold of >=0.5.

% Find the max from each row (moving *across* 2nd dimension)
% maxes: m x 1; a prediction per row in X
[maxes, p] = max(predictions, [], 2);
assert(size(p), [m, 1]);
% =========================================================================
end

%!test
%! all_theta = [1 -6 3; -2 4 -3]
%! X = [1 7; 4 5; 7 8; 1 4]
%! ans = predictOneVsAll(all_theta, X)
%! exp_ans = [
%!    1
%!    2
%!    2
%!    1
%! ];
%! assert(ans, exp_ans);
