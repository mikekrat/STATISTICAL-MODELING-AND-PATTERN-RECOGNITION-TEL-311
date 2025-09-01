clear all;
close all;

X = [2 2; 2 -2; -2 -2; -2 2; 1 1; 1 -1; -1 -1; -1 1;];
x1 = X(:,2) .^ 2;

figure;
hold on;
plot(X(1: 4, 1), X(1: 4, 2), 'bo');
plot(X(5: 8, 1), X(5: 8, 2), 'ro');
title('Samples before transformation');
grid on;
axis([-3 3 -3 3]);
hold off;

X_trans = transformation(X);

figure;
hold on;
plot(X_trans(1: 4, 1), X_trans(1: 4, 2), 'bo');
plot(X_trans(5: 8, 1), X_trans(5: 8, 2), 'ro');
title('Samples after transformation');
grid on;
axis([-15 -4 -15 -4]);
hold off;