%% Question 2
clear
clc
close all
rng(222)

%% Generating the model
plotData = 1;
%Number of dimensions and number of samples
n = 2; Ntrain = 1000; Ntest = 10000; 
%GMM ratios, must add to 1.0
alpha = [0.33,0.34,0.33];
%Mean vectors
meanVectors = [-18 0 18;-8 0 8];
%Variance-covariance matrix
covEvalues = [3.2^2 0;0 0.6^2];
covEvectors(:,:,1) = [1 -1;1 1]/sqrt(2);
covEvectors(:,:,2) = [1 0;0 1];
covEvectors(:,:,3) = [1 -1;1 1]/sqrt(2);
t = rand(1,Ntrain);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtrain = zeros(n,Ntrain);
Xtrain(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtrain(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtrain(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);
t = rand(1,Ntest);
ind1 = find(0 <= t & t <= alpha(1));
ind2 = find(alpha(1) < t & t <= alpha(1)+alpha(2));
ind3 = find(alpha(1)+alpha(2) <= t & t <= 1);
Xtest = zeros(n,Ntrain);
Xtest(:,ind1) = covEvectors(:,:,1)*covEvalues^(1/2)*randn(n,length(ind1))+meanVectors(:,1);
Xtest(:,ind2) = covEvectors(:,:,2)*covEvalues^(1/2)*randn(n,length(ind2))+meanVectors(:,2);
Xtest(:,ind3) = covEvectors(:,:,3)*covEvalues^(1/2)*randn(n,length(ind3))+meanVectors(:,3);
if plotData == 1
    figure(1), subplot(1,2,1),
    plot(Xtrain(1,:),Xtrain(2,:),'.')
    title('Training Data'), axis equal,
    subplot(1,2,2),
    plot(Xtest(1,:),Xtest(2,:),'.')
    title('Testing Data'), axis equal,
end
percep = 10; 
fold = 10;
net = feedforwardnet(1);
net.divideMode = 'none';
net.divideFcn = 'dividetrain';
per = zeros(fold,1);
meanPer = zeros(percep,2);
dummy = ceil(linspace(0,Ntrain,fold+1));
for k = 1:fold
    indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
end
for i = 1:percep
    for j = 1:2
        for k = 1:fold
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            x1Validate = Xtrain(1,indValidate); 
            x2Validate = Xtrain(2,indValidate);
            if k == 1
                indTrain = [indPartitionLimits(k,2)+1:Ntrain];
            elseif k == fold
                indTrain = [1:indPartitionLimits(k,1)-1];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):Ntrain];
            end
            x1Train = Xtrain(1,indTrain); 
            x2Train = Xtrain(2,indTrain);
            Xtrain1 = [x1Train; x2Train];
            xValidate = [x1Validate; x2Validate];
            Ntrain1 = length(indTrain); Nvalidate = length(indValidate);
            if j==2
                net.layers{1}.transferFcn = 'softplus';
            else
                net.layers{1}.transferFcn = 'logsig';
            end
            net.layers{1}.size = i;
            net = configure(net,Xtrain1(1,:),Xtrain1(2,:));
            net.trainParam.showWindow = true;
            net = train(net,Xtrain1(1,:),Xtrain1(2,:));
            yy = net(xValidate(1,:));
            per(k) = perform(net,xValidate(2,:),yy);
        end
        meanPer(i,j) = mean(per);
        if i == 1 && j == 1
            bestNet = net;
            bestPer = meanPer(1,1);
        else
            if meanPer(i,j) < bestPer
                bestNet = net;
                bestPer = meanPer(i,j);
            end
        end       
    end
end
% Train again with best model
bestNet = configure(bestNet,Xtrain(1,:),Xtrain(2,:));
bestNet.trainParam.showWindow = true;
bestNet = train(bestNet,Xtrain(1,:),Xtrain(2,:));
yytest = bestNet(Xtest(1,:));
testPer = perform(bestNet,Xtest(2,:),yytest);

% Visualization
figure(2)
hold on
stem(meanPer(:,1))
stem(meanPer(:,2))
title("Average Performance of the MLP Estimator Across Folds")
xlabel("Number of Perceptrons") 
ylabel("MSE")
legend('Sigmoid','Softplus')
hold off

figure(3)
hold on
scatter(Xtest(1,:),Xtest(2,:));
scatter(Xtest(1,:),yytest);
hold off
title("Best MLP Model on Test Data")
xlabel("X1") 
ylabel("X2")
legend('True','Predicted');
