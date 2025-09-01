function out = transformation(X)
% MAPFEATURE Feature mapping function to polynomial features

out = zeros(size(X));

out(:,1) = X(:,1) - (X(:,1) .^ 2 + X(:,2) .^ 2) - 4;
out(:,2) = X(:,2) - (X(:,1) .^ 2 + X(:,2) .^ 2) - 4;

end