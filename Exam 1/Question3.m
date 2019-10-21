clear; close all; clc
sig = 2^2; % Sigma for v
w_tr = [1; -0.6; -0.24; 0.064]; % True labels
gamma = (10.^(-6:0.05:6)).^2; % Gamma values
exp_count = 1000; % Experiment count
error = zeros(length(gamma),length(exp_count));

for i = 1:length(gamma)
    for j = 1:exp_count
        x = rand(10,1).*2 - 1; % Uniform x
        v = normrnd(0,sig,10,1); % Generate noise
        x_vec = [x.^3, x.^2, x, ones(length(x),1)]; % Store all x as vector
        y = w_tr(1).*x_vec(:,1) + w_tr(2).*x_vec(:,2) + ...
            w_tr(3).*x_vec(:,3) + w_tr(4).*x_vec(:,4) + v; % Calculate y
        w_est = ((sig/gamma(i)*eye(4))+x_vec'*x_vec)\x_vec'*y; % Estimate true labels
        error(i, j) = (w_tr'*w_tr-w_est'*w_est)^2; % squared-error values
    end
end

% Store necessary error values
error_sorted = sort(error,2); error_min = error_sorted(:,1);...
    error_25 = error_sorted(:,exp_count/4); error_med = error_sorted(:,exp_count/2);...
    error_75 = error_sorted(:,3*exp_count/4); error_max = error_sorted(:,end);

loglog(sqrt(gamma),error_min)
hold on
loglog(sqrt(gamma),error_25)
loglog(sqrt(gamma),error_med)
loglog(sqrt(gamma),error_75)
loglog(sqrt(gamma),error_max)
hold off
grid on
ylabel('||w_{true} - w_{MAP}||_{2}^{2}')
xlabel('\gamma')
title('Squared Errors of MAP Estimator for Different Values of \gamma')
