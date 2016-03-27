function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

m = size(X, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Distance matrix
dist = zeros(m, K);

for c = 1:K % Compare each example to each centroid
    diffs = bsxfun(@minus, X, centroids(c, :));
    % Compute sum of square along the second axis (Columns)
    dist(:, c) = sum(diffs.^2, 2);
end
% fprintf('%d most similar to centroid %d\n', i, idx(i));
% =============================================================

% Return the indices of the closest centroids
[~, idx] = min(dist, [], 2);
end

%!test
%! ans = findClosestCentroids([0 1; 5 5; -1 8], [7 6; -2 2]);
%! assert(ans, [2 1 2]');

%!test
%! X = magic(8);
%! X = X(:, 2:4);
%! centroids = magic(4);
%! centroids = centroids(:,2:4);
%! ans = findClosestCentroids(X, centroids)
%! assert(ans, [1 4 4 2 4 3 3 4]');
