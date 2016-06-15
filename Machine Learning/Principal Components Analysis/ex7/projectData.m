function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
% Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K
%               eigenvectors in U (first K columns).
%               For the i-th example X(i,:), the projection on to the k-th
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%
Z = X * U(:, 1:K);


% =============================================================

end

%!test
%! format long
%! ans = projectData(sin([0 1 2; 3 4 5; 6 7 8]), magic(3), 2);
%! assert(ans, [
%!   6.161602661726416 12.391031765470618;
%!   -4.977144520097401 -12.273210940448021;
%!   3.693068797058487 11.909741715005456
%! ]);
