clear; close all;
% generateDara_Exam1Question1.m slightly modified
m(:,1) = [-1;0]; Sigma(:,:,1) = 0.1*[10 -4;-4,5]; % mean and covariance of data pdf conditioned on label 1
m(:,2) = [1;0]; Sigma(:,:,2) = 0.1*[5 0;0,2]; % mean and covariance of data pdf conditioned on label 2
m(:,3) = [0;1]; Sigma(:,:,3) = 0.1*eye(2); % mean and covariance of data pdf conditioned on label 3
classPriors = [0.15,0.35,0.5]; thr = [0,cumsum(classPriors)];
N = 10000; u = rand(1,N); L = zeros(1,N); x = zeros(2,N);
figure(1),clf, colorList = 'rbg'; markerList = '+*o';
nInd = zeros(1,3);
for l = 1:3
    indices = find(thr(l)<=u & u<thr(l+1)); % if u happens to be precisely 1, that sample will get omitted - needs to be fixed
    L(1,indices) = l*ones(1,length(indices));
    x(:,indices) = mvnrnd(m(:,l),Sigma(:,:,l),length(indices))';
    nInd(l) = length(indices); % Data for each class
    figure(1), plot(x(1,indices),x(2,indices),markerList(l),'MarkerEdgeColor',colorList(l)); axis equal, hold on,
end
hold off
legend('Label 1','Label 2','Label 3'), 
title('Generated Data with True Labels'),
xlabel('x_1'), ylabel('x_2')

% Evaluate gaussian for decision boundary
dec1 = evalGaussian(x,m(:,2),Sigma(:,:,2))*classPriors(2) + evalGaussian(x,m(:,3),Sigma(:,:,3))*classPriors(3);
dec2 = evalGaussian(x,m(:,1),Sigma(:,:,1))*classPriors(1) + evalGaussian(x,m(:,3),Sigma(:,:,3))*classPriors(3);
dec3 = evalGaussian(x,m(:,1),Sigma(:,:,1))*classPriors(1) + evalGaussian(x,m(:,2),Sigma(:,:,2))*classPriors(2);
dec = zeros(1,N); 

% Classify
for i = 1:N
    if dec1(i) < dec2(i) && dec1(i) < dec3(i)
        dec(i) = 1;
    elseif dec2(i) < dec1(i) && dec2(i) < dec3(i)
        dec(i) = 2;
    elseif dec3(i) < dec1(i) && dec3(i) < dec2(i)
        dec(i) = 3;
    end
end

% Classified indices
L1_D1_ind = find(dec == 1 & L == 1);
L1_D2_ind = find(dec == 2 & L == 1);
L1_D3_ind = find(dec == 3 & L == 1);
L2_D1_ind = find(dec == 1 & L == 2);
L2_D2_ind = find(dec == 2 & L == 2);
L2_D3_ind = find(dec == 3 & L == 2);
L3_D1_ind = find(dec == 1 & L == 3);
L3_D2_ind = find(dec == 2 & L == 3);
L3_D3_ind = find(dec == 3 & L == 3);

figure(2), 
plot(x(1,L1_D1_ind),x(2,L1_D1_ind),markerList(1),'MarkerEdgeColor',colorList(1))
hold on
plot(x(1,L1_D2_ind),x(2,L1_D2_ind),markerList(1),'MarkerEdgeColor',colorList(2))
plot(x(1,L1_D3_ind),x(2,L1_D3_ind),markerList(1),'MarkerEdgeColor',colorList(3))
plot(x(1,L2_D1_ind),x(2,L2_D1_ind),markerList(2),'MarkerEdgeColor',colorList(1))
plot(x(1,L2_D2_ind),x(2,L2_D2_ind),markerList(2),'MarkerEdgeColor',colorList(2))
plot(x(1,L2_D3_ind),x(2,L2_D3_ind),markerList(2),'MarkerEdgeColor',colorList(3))
plot(x(1,L3_D1_ind),x(2,L3_D1_ind),markerList(3),'MarkerEdgeColor',colorList(1))
plot(x(1,L3_D2_ind),x(2,L3_D2_ind),markerList(3),'MarkerEdgeColor',colorList(2))
plot(x(1,L3_D3_ind),x(2,L3_D3_ind),markerList(3),'MarkerEdgeColor',colorList(3))
axis equal
legend('Label 1 - Decision 1 (Correct)','Label 1 - Decision 2 (Incorrect)','Label 1 - Decision 3 (Incorrect)' ...
    ,'Label 2 - Decision 1 (Incorrect)','Label 2 - Decision 2 (Correct)','Label 2 - Decision 3 (Incorrect)' ...
    ,'Label 3 - Decision 1 (Incorrect)','Label 3 - Decision 2 (Incorrect)','Label 3 - Decision 3 (Correct)'), 
title('Data Decision After Classification'),
xlabel('x_1'), ylabel('x_2')

% Confusion matrix
confusionMatrix = [ length(L1_D1_ind), length(L1_D2_ind), length(L1_D3_ind); ...
    length(L2_D1_ind), length(L2_D2_ind), length(L2_D3_ind); ...
    length(L3_D1_ind), length(L3_D2_ind), length(L3_D3_ind)];

% Total number of samples misclassified by classifier
misclass = N - trace(confusionMatrix);

% Probability of Error
probE = misclass/N;
