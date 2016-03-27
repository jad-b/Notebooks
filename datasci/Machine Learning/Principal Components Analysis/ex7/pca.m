function [U, S] = pca(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = pca(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

% ====================== YOUR CODE HERE ======================
% Instructions: You should first compute the covariance matrix. Then, you
%               should use the "svd" function to compute the eigenvectors
%               and eigenvalues of the covariance matrix.
%
% Note: When computing the covariance matrix, remember to divide by m (the
%       number of examples).
%

Sigma = (X' * X) ./ m;

[U, S, V] = svd(Sigma);


% =========================================================================

end

%!test
%! format long
%! [U, S] = pca(sin([0 1; 2 3; 4 5]))
%! assert(U, [
%!   -0.654347329763442 -0.756194136469897;
%!   -0.756194136469897 0.654347329763442
%! ], 2.25e-16);
%! assert(S, [0.795511951488468 0; 0 0.220186670785774], 1.7e-16);

