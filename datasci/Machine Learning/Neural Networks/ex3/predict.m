function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
[m n] = size(X);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add bias column
a1 = [ones(m, 1) X];
assert(size(a1), ([m n+1]));

a2 = sigmoid(Theta1 * a1')';
% Add bias column
a2 = [ones(size(a2, 1), 1) a2];

a3 = sigmoid(Theta2 * a2')';
% Return max from output layer
[mx, p] = max(a3, [], 2);
p = p';
% =========================================================================
end

%!test
%! Theta1 = reshape(sin(0 : 0.5 : 5.9), 4, 3);
%! Theta2 = reshape(sin(0 : 0.3 : 5.9), 4, 5);
%! X = reshape(sin(1:16), 8, 2);
%! exp_p = predict(Theta1, Theta2, X);
%! p = [
%!   4
%!   1
%!   1
%!   4
%!   4
%!   4
%!   4
%!   2
%! ]';
%! assert(exp_p, p)
