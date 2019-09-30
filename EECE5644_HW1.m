%% Question 2-3
a1=0; b1=1; a2=1; b2=2; % Variables
x=-10:0.1:10; % x values
pL1=exp(-1*abs(x-a1)./b1)./(b1.*2); % First class
pL2=exp(-1*abs(x-a2)./b2)./(b2.*2); % Second class
lx=log(pL1)-log(pL2); % log-likelihood ratio
figure
plot(x,lx, 'k', 'LineWidth', 1.5);
grid on
ylabel('p(L=l)|x');
xlabel('x');
title('Log-Likelihood Ratio');
%% Question 4-2
mu = 1; % mean
sigma = sqrt(2); % standard deviation
x = -8:0.1:8; % x values
% Gaussian distribution
y1 = normpdf(x,0,1); % First class
y2 = normpdf(x,mu,sigma); % Second class
% Class posterior probabilities
z1 = y1./(y1+y2); 
z2 = y2./(y1+y2);
% Visualization
plot(x,y1, 'LineWidth', 1.5);
hold on
plot(x,y2, 'LineWidth', 1.5)
yL = get(gca,'YLim');
line([(-sqrt(2+log(4))-1) (-sqrt(2+log(4))-1)],yL,'Color','k', 'LineWidth', 1.5, 'LineStyle', '-.');
line([(sqrt(2+log(4))-1) (sqrt(2+log(4))-1)],yL,'Color','k', 'LineWidth', 1.5, 'LineStyle', '-.');
hold off
grid on
ylabel('p(x|L=l)');
xlabel('x');
title('Class Conditional PDFs');
legend('L = 1', 'L = 2', 'Decision Boundaries')
figure
plot(x,z1, 'LineWidth', 1.5);
hold on
plot(x,z2, 'LineWidth', 1.5)
hold off 
grid on
ylabel('p(L=l)|x');
xlabel('x');
title('Class Posterior Probabilities');
legend('L = 1', 'L = 2')
%% Question 5-3
N = 1000; % Number of samples
n = 3; % n-dimensions
mu = 5; % Mean 
sigma = 500; % Variance
z = randn(N,n); % Normal distribution generation
x = sqrt(sigma).*z + mu; % Linear transformation
