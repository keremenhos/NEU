nSamples = 999;
mu{1} = [-4;4]; mu{2} = [4,2];
sigma{1} = [10 -2;-2 3]; sigma{2} = [5 1;1 4];
prior = [0.3; 0.7];
xMin = -7; xMax = 7;
yMin = -7; yMax = 7;
nClass = numel(mu);

[data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior);
figure(1); 
titleString = 'Generated Data';
plotSamples(data, classIndex, nClass, titleString);
axis([xMin xMax yMin yMax]);

% LDA Classification and Visualization for Question 3
[ind01LDA,ind10LDA,ind00LDA,ind11LDA,pErrLDA,wLDA,bLDA] = classifyLDA(data, classIndex, mu, sigma, nSamples, prior);
figure(2); 
plotDecision(data,ind01LDA,ind10LDA,ind00LDA,ind11LDA);
title(sprintf('LDA Pe=%.4f',pErrLDA), 'FontSize', 18);
axis([xMin xMax yMin yMax]);

% Model
options = optimset('PlotFcns',@optimplotfval);
% aa = zeros(1,999);
f = @(w,z)(-1/nSamples).*(sum(log(1./(1+exp([w(1) w(2)]*z(:,classIndex==2)+w(3)))))+sum(log(1-(1./(1+exp([w(1) w(2)]*z(:,classIndex==1)+w(3)))))));
z = data';
fun = @(w)f(w,z);
x0 = [wLDA(1) wLDA(2) bLDA];
minValues = fminsearch(fun, x0, options);

y = 1./(1+exp([minValues(1) minValues(2)]*data'+minValues(3)));
labelMAP = zeros(1,nSamples);
for i = 1:nSamples
    if y(i) < 0.5
        labelMAP(i) = 0;
    else
        labelMAP(i) = 1;
    end
end
%Probability of true negative
ind00MAP = find(labelMAP==0 & classIndex'==1); 
%Probability of false positive
ind10MAP = find(labelMAP==1 & classIndex'==1); 
%Probability of false negative
ind01MAP = find(labelMAP==0 & classIndex'==2); 
%Probability of true positive
ind11MAP = find(labelMAP==1 & classIndex'==2);  
%Number of errors
errorMAP = length(ind10MAP)+length(ind01MAP);
%Probability of errors
pErrMAP = errorMAP/nSamples;

figure(3); 
plotDecision(data,ind01MAP,ind10MAP,ind00MAP,ind11MAP);
title(sprintf('Logic Model Pe=%.4f',pErrMAP), 'FontSize', 18);
axis([xMin xMax yMin yMax]);


function [data, classIndex] = generateGaussianSamples(mu, sigma, nSamples, prior)
% Function to simulate data from k Gaussian densities (1 for each class) in d dimensions.
%
% INPUTS:
% mu - cell with the class dependent d-dimensional mean vector
% sigma - k-by-1 cell with the class dependent d-by-d covariance matrix
% nSamples - scalar indicating number of samples to be generated
% prior - k-by-1 vector with class dependent mean
%
% OUTPUTS:
% data - nSamples-by-d array with the simulated data distributed along the rows
% classIndex - vector of length nSamples with the class index for each datapoint
if sum(prior) ~= 1
    error('Priors should add to one!');
end
% First, sample the respective class indexes. We can do this by generating uniformly
% distributed numbers from 0 to 1 and using thresholds based on the prior probabilities
classTempScalar = rand(nSamples, 1);
priorThresholds = cumsum([0; prior]);
nClass = numel(mu);
data = cell(nClass, 1);
classIndex = cell(nClass, 1);
for idxClass = 1:nClass
    nSamplesClass = nnz(classTempScalar>=priorThresholds(idxClass) & classTempScalar<priorThresholds(idxClass+1));
    % Generate samples according to class dependent parameters
    data{idxClass} = mvnrnd(mu{idxClass}, sigma{idxClass}, nSamplesClass);
    % Set class labels
    classIndex{idxClass} = ones(nSamplesClass,1) * idxClass;
end
data = cell2mat(data);
classIndex = cell2mat(classIndex);
end


function plotSamples(data, classIndex, nClass, titleString)
markerStrings = ['x','o']; colorString = ['r', 'b'];
for idxClass = 1:nClass
dataClass = data(classIndex==idxClass,:);
plot(dataClass(:,1), dataClass(:,2) , [colorString(idxClass)
markerStrings(idxClass)]);
hold on;
end
hold off;
title(titleString, 'FontSize', 18);
xlabel('Feature 1', 'FontSize', 16);
ylabel('Feature 2', 'FontSize', 16);
legend({'Class 1', 'Class 2'});
grid on; box on;
set(gca, 'FontSize', 14);
end


function [ind01LDA,ind10LDA,ind00LDA,ind11LDA,pEminLDA, wLDA,bLDA] = classifyLDA(data, classIndex,mu, sigma, nSamples, ~)
% Fisher LDA Classifer (using true model parameters)
Sb = (mu{1}'-mu{2}')*(mu{1}'-mu{2}')';
Sw = sigma{1} + sigma{2};
[V,D] = eig((Sw)\Sb); % LDA solution satisfies alpha Sw w = Sb w; ie w is a generalized eigenvector of (Sw,Sb)
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*data'; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(classIndex==2))-mean(yLDA(classIndex==1)))*wLDA; % ensures class1 falls on the + side of the axis
discriminantScoreLDA = sign(mean(yLDA(classIndex==2))- mean(yLDA(classIndex==1)))*yLDA; % flip yLDA accordingly
% Estimate the ROC curve for this LDA classifier
[ROCLDA,tauLDA] = estimateROC(discriminantScoreLDA,classIndex');
probErrorLDA = [ROCLDA(1,:)',1- ROCLDA(2,:)']*[sum(classIndex==1),sum(classIndex==2)]'/nSamples; % probability of error for LDA for different threshold values
pEminLDA = min(probErrorLDA); % minimum probability of error
ind = find(probErrorLDA == pEminLDA);
decisionLDA = (discriminantScoreLDA >= tauLDA(ind(1))); % use smallest min-error threshold
ind00LDA = find(decisionLDA==0 & classIndex'==1); % true negatives
ind10LDA = find(decisionLDA==1 & classIndex'==1); % false positives
ind01LDA = find(decisionLDA==0 & classIndex'==2); % false negatives
ind11LDA = find(decisionLDA==1 & classIndex'==2); % true positives
bLDA = yLDA(ind);
end

function [ROC,tau] = estimateROC(discriminantScoreLDA,label)
% Generate ROC curve samples
Nc = [length(find(label==1)),length(find(label==2))];
sortedScore = sort(discriminantScoreLDA,'ascend');
tau = [sortedScore(1)-1,(sortedScore(2:end)+sortedScore(1:end-1))/2,sortedScore(end)+1];
% thresholds at midpoints of consecutive scores in sorted list
for k = 1:length(tau)
    decision = (discriminantScoreLDA >= tau(k));
    ind10 = find(decision==1 & label==1); p10 = length(ind10)/Nc(1); % probability of false positive
    ind11 = find(decision==1 & label==2); p11 = length(ind11)/Nc(2); % probability of true positive
    ROC(:,k) = [p10;p11];
end
end

function plotDecision(data,ind01,ind10,ind00,ind11)
plot(data(ind01,1),data(ind01,2),'xm'); hold on; % false negatives
plot(data(ind10,1),data(ind10,2),'om'); hold on; % false positives
plot(data(ind00,1),data(ind00,2),'xg'); hold on;
plot(data(ind11,1),data(ind11,2),'og'); hold on;
xlabel('Feature 1', 'FontSize', 16);
ylabel('Feature 2', 'FontSize', 16);
grid on; box on;
set(gca, 'FontSize', 14);
legend({'Misclassified as C1','Misclassified as C2'});
end
