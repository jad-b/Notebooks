function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% divide the distance between the lowest and highest probability by a 1000
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions
    % =============================================================
    cvPreds = (pval < epsilon);
    % True positives: Predicted matches actual
    tp = sum((cvPreds==1) & (yval==1));
    % False positives: Predicted, but isn't actual
    fp = sum((cvPreds==1) & (yval==0));
    % False negatives: Didn't predict, but was
    fn = sum((cvPreds==0) & (yval==1));
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);
    F1 = 2 * (prec * rec) / (prec + rec);

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end

%!test
%! [epsilon F1] = selectThreshold([1 0 0 1 1]', [0.1 0.2 0.3 0.4 0.5]')
%! assert(epsilon, 0.40040);
%! assert(F1, 0.57143, 1.5e-6);
