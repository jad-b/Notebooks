function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices.
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% =========================================================================
% Feed-Forward
% =========================================================================
% Input layer
a1 = [ones(m, 1), X];
assert(size(a1), [m input_layer_size+1]); % (m x n+1)

% Hidden layer
z2 = a1 * Theta1';
a2 = [ones(size(z2, 1), 1), sigmoid(z2)];
assert(size(a2), [m hidden_layer_size+1]); % (m x h+1)

% Output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);
% Assert one prediction per training sample; m x 1
assert(size(a3), [m num_labels]); % (m x k)

% =========================================================================
% Cost Function
% =========================================================================
% Makes a m x k matrix of labels
y_matrix = eye(num_labels)(y, :);
% J = predicted - actual => 1 x 1 (scalar output)
%   = (error if y==1) - (error if y==0)
%   = y_pos - y_neg
y_pos = -y_matrix .* log(a3);
onez = ones(size(y_matrix));
y_neg = (onez - y_matrix) .* log((onez - a3));
err_matrix = y_pos - y_neg;
% Sum the error of all samples
sum_sample = sum(err_matrix(:));

J = (1/m) .* sum_sample;

% Add regularization
% Theta1(:, 2:end)(:) - Drop 1st column, unroll into vector
Theta1_sqsum = sum(Theta1(:, 2:end)(:) .^ 2);
Theta2_sqsum = sum(Theta2(:, 2:end)(:) .^ 2);
J_reg = (lambda/(2*m)) * (Theta1_sqsum + Theta2_sqsum);
J = J + J_reg;

% =========================================================================
% Backpropagation
% =========================================================================
% delta %
% First we calculate the deltas - the errors for our layers.
% This is easy for the output layer, as its output *is* our predictions
% delta_3 = predicted - actual
d3 = a3 - y_matrix;
assert(size(d3), [m num_labels]); % Dimensions: (m x k)

% The hidden layers' error is derived from the error of the next layer up. We
% work backwards from our known error (output layer) to compute the error
% for our hidden layers; we _back propagate_ error.
% Formula:
% delta_2 = delta_3 * Theta^2 * dg/dz(z2)
% Natural:
% delta_2 = Error of output layerWeights for l_hidden => l_output
%         * Weights used to calculate a3 from a2, ignoring bias
%         * derivative of sigmoid with pre-sigmoid values
% Dimensional (h is the number of hidden units, w/o bias):
% delta_2, sizes = (m x k) * (k x h) .* z2 => (m x h)
T2_tmp = Theta2(:, 2:end);
assert(size(T2_tmp), [num_labels hidden_layer_size]);
d2 = d3 * T2_tmp .* sigmoidGradient(z2);
assert(size(d2), [m hidden_layer_size]);

% Delta %
% With the error for each layers' units calculated in delta_l, we can derive the
% partial derivatives.
% Formula: d2 * a1
% Natural:
% Delta_1 = Error of hidden layer
%         * activation of input layer
% Dimensional:
% Delta_1 = (h x m) * (m x n+1) => (h x n)
D1 = d2' * a1;
assert(size(D1), [hidden_layer_size input_layer_size+1]);

% Dimensional: (k x m) * (m x h+1) => (k x h)
D2 = d3' * a2;
assert(size(D2), [num_labels hidden_layer_size+1]);

% Scale by the number of samples
Theta1_grad = D1./m;
Theta2_grad = D2./m;

% Regularize gradients
% Zero out bias columns
Theta1(:, 1) = 0;
Theta2(:, 1) = 0;
% Add regularization factor: lambda/m
reg = lambda/m;
Theta1_grad = Theta1_grad + (reg .* Theta1);
Theta2_grad = Theta2_grad + (reg .* Theta2);

% =========================================================================
% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end

% =========================================================================
% Testing
% =========================================================================
%!shared il, hl, nl, nn, X, y, lambda
%! il = 2;              % input layer
%! hl = 2;              % hidden layer
%! nl = 4;              % number of labels
%! nn = [ 1:18 ] / 10;  % nn_params
%! X = cos([1 2 ; 3 4 ; 5 6]);
%! y = [4; 2; 3];
%! lambda = 4;

%!test
%! [J grad] = nnCostFunction(nn, il, hl, nl, X, y, lambda);
%! assert(J, 19.474, 3.7e-4); % Regularized
%! assert(grad, [
%!  0.76614
%!  0.97990
%!  0.37246
%!  0.49749
%!  0.64174
%!  0.74614
%!  0.88342
%!  0.56876
%!  0.58467
%!  0.59814
%!  1.92598
%!  1.94462
%!  1.98965
%!  2.17855
%!  2.47834
%!  2.50225
%!  2.52644
%!  2.72233
%! ], 4e-6);
