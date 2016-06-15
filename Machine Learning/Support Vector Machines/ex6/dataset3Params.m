function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
bestC = -1;
bestSigma = -1;
lowestCost = Inf();
steps = [.01, .03, .1, .3, 1, 3, 10, 30];
for c = steps
    for s = steps
        fprintf('New C: %f\nNew Sigma: %f\n', c, s);
        model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
        predictions = svmPredict(model, Xval);
        err = mean(double(predictions ~= yval));
        if err < lowestCost
            bestC = c;
            bestSigma = s;
            lowestCost = err;
            fprintf('New Best C: %f\nNew Best Sigma: %f\n', bestC, bestSigma);
    end
end

fprintf('Best C: %f\nBest Sigma: %f\n', bestC, bestSigma);
C = bestC;
sigma = bestSigma;
% =========================================================================

end
