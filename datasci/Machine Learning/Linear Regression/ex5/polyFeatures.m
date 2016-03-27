function [X_poly] = polyFeatures(X, p)
%POLYFEATURES Maps X (1D vector) into the p-th power
%   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
%   maps each example into its polynomial features where
%   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
%

% Broadcast p to fit X (or vice-versa), and perform an element-wise operation.
X_poly = bsxfun(@power, X, (1:p));

% =========================================================================

end

%!test
%! assert(polyFeatures([1:3]',4), [
%!    1    1    1    1
%!    2    4    8   16
%!    3    9   27   81
%! ])
