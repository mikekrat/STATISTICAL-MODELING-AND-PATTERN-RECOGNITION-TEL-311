function [U, S] = myPCA(X)
%PCA Run principal component analysis on the dataset X
%   [U, S, X] = myPCA(X) computes eigenvectors of the covariance matrix of X
%   Returns the eigenvectors U, the eigenvalues (on diagonal) in S
%

% Useful values
[m, n] = size(X);

% You need to return the following variables correctly (U, S).

% ====================== YOUR CODE GOES HERE ======================
% Apply PCA by computing the eigenvectors and eigenvalues of the covariance matrix. 
%

covariance_matrix = (1 / m) * (X' * X);
[U, S] = svd(covariance_matrix);


% =========================================================================

end