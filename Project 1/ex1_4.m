clear ; close all; clc

m1 = [3 3];
S1 = [1.2 -0.4; -0.4 1.2];

m2 = [6 6];
S2 = [1.2 0.4; 0.4 1.2];

x1 = -2 : 0.01 : 12;
x2 = -2 : 0.01 : 12;

x1_size = length(x1);
x2_size = length(x2);

[X1, X2] = meshgrid(x1, x2);

Y1 = mvnpdf([X1(:) X2(:)], m1, S1);
Y2 = mvnpdf([X1(:) X2(:)], m2, S2);

Y1_Reshape = reshape(Y1, x2_size, x1_size);
Y2_Reshape = reshape(Y2, x2_size, x1_size);

%% 1st case ( S1 != S2 )
figure(1)
hold on
contour(x1, x2, Y1_Reshape, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'r')
contour(x1, x2, Y2_Reshape, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'b')
grid on

P1 = [0.1 0.25 0.5 0.75 0.9];
P_Size = length(P1);
P2 = 1 - P1;

for i = 1 : P_Size
    syms x y
    y = (22.5 - 2 * log(P2(i) / P1(i))) / (1.25 * x);
    ezplot(y, [-2 12 -2 12]);
    legend('Class 1', 'Class 2', 'P1 = 0.1', 'P1 = 0.25', 'P1 = 0.5', 'P1 = 0.75', 'P1 = 0.9')
end
hold off

%% 2nd case ( S1 = S2 )

S1 = S2;

Y1 = mvnpdf([X1(:) X2(:)], m1, S1);
Y1_Reshape = reshape(Y1, x2_size, x1_size);

figure(2)
hold on
contour(x1, x2, Y1_Reshape, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'r')
contour(x1, x2, Y2_Reshape, [.0001 .001 .01 .05:.1:.95 .99 .999 .9999], 'b')
grid on

for i = 1 : P_Size
    syms x y
    y = -x + (33.75 - 2 * log(P2(i) / P1(i)) ) / 3.75;
    ezplot(y, [-2 12 -2 12]);
    legend('Class 1', 'Class 2', 'P1 = 0.1', 'P1 = 0.25', 'P1 = 0.5', 'P1 = 0.75', 'P1 = 0.9')
end
hold off





