function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%

% y==1 produces a boolean (mask) array of 0/1's (true/false), indicating
% to "Was the value at this index in vector 'y' equal to 1?"
% 'find' returns a vector of indicies where the value was non-zero.
% It's kind of like compression; skip the zeros, but you lose the non-zero
% value.
pos = find(y==1); neg = find(y==0);

plot(X(pos, 1), X(pos, 2), 'k+', 'linewidth', 2, 'markersize', 7);
plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'markersize', 7);
% =========================================================================
hold off;

end
