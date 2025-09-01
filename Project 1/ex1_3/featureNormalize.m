function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.


mu = mean(X);% mean of each column (feature)

sigma = std(X);% standart deviation of each column
X_norm = zeros(size(X));% normalize each column independently

columns_length = size(X, 2);

for j = 1 : columns_length % j is column index
     X_norm(:, j) = (X(:, j) - mu(1, j)) / sigma(1, j);
end
% ============================================================

end
