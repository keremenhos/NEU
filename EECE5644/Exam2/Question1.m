%% Question 1
clear
clc
close all
rng(222)

%% Data Generation
% Variables
N1 = 100; N2 = 1000; N3 = 10000; N4 = 15000;
N = [N1 N2 N3];
prior = [0.1,0.2,0.3,0.4];
m = [-4 7 -3 4;10 0 -2 11;3 6 -1 8];
Sigma(:,:,1) = [22 1 4;1 20 6;15 13 7];
Sigma(:,:,2) = [27 4 0;6 5 3;3 1 20];
Sigma(:,:,3) = [16 9 6;9 16 8;3 4 6];
Sigma(:,:,4) = [4 1 2;1 20 3;4 6 14];
% Gaussian Generation
[GMM1, lab1] = randGMM(N1,prior,m,Sigma);
[GMM2, lab2] = randGMM(N2,prior,m,Sigma);
[GMM3, lab3] = randGMM(N3,prior,m,Sigma);
[GMM4, lab4] = randGMM(N4,prior,m,Sigma);
Glab1 = zeros(4,N1);
Glab2 = zeros(4,N2);
Glab3 = zeros(4,N3);
Glab4 = zeros(4,N4);
% Label Generation
for i = 1:N1
    if lab1(i) == 1
        Glab1(1,i) = 1;
    elseif lab1(i) == 2
        Glab1(2,i) = 1;
    elseif lab1(i) == 3
        Glab1(3,i) = 1;
    else
        Glab1(4,i) = 1;
    end
end
for i = 1:N2
    if lab2(i) == 1
        Glab2(1,i) = 1;
    elseif lab2(i) == 2
        Glab2(2,i) = 1;
    elseif lab2(i) == 3
        Glab2(3,i) = 1;
    else
        Glab2(4,i) = 1;
    end
end
for i = 1:N3
    if lab3(i) == 1
        Glab3(1,i) = 1;
    elseif lab3(i) == 2
        Glab3(2,i) = 1;
    elseif lab3(i) == 3
        Glab3(3,i) = 1;
    else
        Glab3(4,i) = 1;
    end
end
for i = 1:N4
    if lab4(i) == 1
        Glab4(1,i) = 1;
    elseif lab4(i) == 2
        Glab4(2,i) = 1;
    elseif lab4(i) == 3
        Glab4(3,i) = 1;
    else
        Glab4(4,i) = 1;
    end
end

% Generated Data Visualization
figure(1)
scatter3(GMM2(1,lab2==1),GMM2(2,lab2==1),GMM2(3,lab2==1),'ob'); hold on
scatter3(GMM2(1,lab2==2),GMM2(2,lab2==2),GMM2(3,lab2==2),'or')
scatter3(GMM2(1,lab2==3),GMM2(2,lab2==3),GMM2(3,lab2==3),'ok')
scatter3(GMM2(1,lab2==4),GMM2(2,lab2==4),GMM2(3,lab2==4),'og')
title('Generated Data for 1000 Samples')
legend('Class 1', 'Class 2', 'Class 3', 'Class 4')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

%% MAP Classifier
% Loss Function
lambda = [0 1 1 1;1 0 1 1;1 1 0 1;1 1 1 0]; 
% Decision Calculation
g1 = lambda(1,1)*evalGaussian(GMM4,m(:,1),Sigma(:,:,1))*prior(1) + lambda(1,2)*evalGaussian(GMM4,m(:,2),Sigma(:,:,2))*prior(2) + lambda(1,3)*evalGaussian(GMM4,m(:,3),Sigma(:,:,3))*prior(3) + lambda(1,4)*evalGaussian(GMM4,m(:,4),Sigma(:,:,4))*prior(4);
g2 = lambda(2,1)*evalGaussian(GMM4,m(:,1),Sigma(:,:,1))*prior(1) + lambda(2,2)*evalGaussian(GMM4,m(:,2),Sigma(:,:,2))*prior(2) + lambda(2,3)*evalGaussian(GMM4,m(:,3),Sigma(:,:,3))*prior(3) + lambda(2,4)*evalGaussian(GMM4,m(:,4),Sigma(:,:,4))*prior(4);
g3 = lambda(3,1)*evalGaussian(GMM4,m(:,1),Sigma(:,:,1))*prior(1) + lambda(3,2)*evalGaussian(GMM4,m(:,2),Sigma(:,:,2))*prior(2) + lambda(3,3)*evalGaussian(GMM4,m(:,3),Sigma(:,:,3))*prior(3) + lambda(3,4)*evalGaussian(GMM4,m(:,4),Sigma(:,:,4))*prior(4);
g4 = lambda(4,1)*evalGaussian(GMM4,m(:,1),Sigma(:,:,1))*prior(1) + lambda(4,2)*evalGaussian(GMM4,m(:,2),Sigma(:,:,2))*prior(2) + lambda(4,3)*evalGaussian(GMM4,m(:,3),Sigma(:,:,3))*prior(3) + lambda(4,4)*evalGaussian(GMM4,m(:,4),Sigma(:,:,4))*prior(4);
dec = zeros(1,N4); 
% Decision Comparison
for i = 1:N4
    if g1(i)<g2(i) && g1(i)<g3(i) && g1(i)<g4(i)
        dec(i) = 1;
    elseif g2(i)<g1(i) && g2(i)<g3(i) && g2(i)<g4(i)
        dec(i) = 2;
    elseif g3(i)<g1(i) && g3(i)<g2(i) && g3(i)<g4(i)
        dec(i) = 3;
    elseif g4(i)<g1(i) && g4(i)<g2(i) && g4(i)<g3(i)
        dec(i) = 4;
    end
end

% MAP Classified Data
figure(2)
scatter3(GMM4(1,dec==1&lab4==1),GMM4(2,dec==1&lab4==1),GMM4(3,dec==1&lab4==1),'ob'); hold on
scatter3(GMM4(1,dec==2&lab4==1),GMM4(2,dec==2&lab4==1),GMM4(3,dec==2&lab4==1),'xr');
scatter3(GMM4(1,dec==3&lab4==1),GMM4(2,dec==3&lab4==1),GMM4(3,dec==3&lab4==1),'xk');
scatter3(GMM4(1,dec==4&lab4==1),GMM4(2,dec==4&lab4==1),GMM4(3,dec==4&lab4==1),'xg');
scatter3(GMM4(1,dec==1&lab4==2),GMM4(2,dec==1&lab4==2),GMM4(3,dec==1&lab4==2),'xb');
scatter3(GMM4(1,dec==2&lab4==2),GMM4(2,dec==2&lab4==2),GMM4(3,dec==2&lab4==2),'or');
scatter3(GMM4(1,dec==3&lab4==2),GMM4(2,dec==3&lab4==2),GMM4(3,dec==3&lab4==2),'xk');
scatter3(GMM4(1,dec==4&lab4==2),GMM4(2,dec==4&lab4==2),GMM4(3,dec==4&lab4==2),'xg');
scatter3(GMM4(1,dec==1&lab4==3),GMM4(2,dec==1&lab4==3),GMM4(3,dec==1&lab4==3),'xb');
scatter3(GMM4(1,dec==2&lab4==3),GMM4(2,dec==2&lab4==3),GMM4(3,dec==2&lab4==3),'xr');
scatter3(GMM4(1,dec==3&lab4==3),GMM4(2,dec==3&lab4==3),GMM4(3,dec==3&lab4==3),'ok');
scatter3(GMM4(1,dec==4&lab4==3),GMM4(2,dec==4&lab4==3),GMM4(3,dec==4&lab4==3),'xg');
scatter3(GMM4(1,dec==1&lab4==4),GMM4(2,dec==1&lab4==4),GMM4(3,dec==1&lab4==4),'xb');
scatter3(GMM4(1,dec==2&lab4==4),GMM4(2,dec==2&lab4==4),GMM4(3,dec==2&lab4==4),'xr');
scatter3(GMM4(1,dec==3&lab4==4),GMM4(2,dec==3&lab4==4),GMM4(3,dec==3&lab4==4),'xk');
scatter3(GMM4(1,dec==4&lab4==4),GMM4(2,dec==4&lab4==4),GMM4(3,dec==4&lab4==4),'og');
title('MAP-Classified Data for 15000 Samples')
legend('Decision 1 & Label 1', 'Decision 2 & Label 1', 'Decision 3 & Label 1', 'Decision 4 & Label 1', ...
    'Decision 1 & Label 2', 'Decision 2 & Label 2', 'Decision 3 & Label 2', 'Decision 4 & Label 2', ...
    'Decision 1 & Label 3', 'Decision 2 & Label 3', 'Decision 3 & Label 3', 'Decision 4 & Label 3', ...
    'Decision 1 & Label 4', 'Decision 2 & Label 4', 'Decision 3 & Label 4', 'Decision 4 & Label 4')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
% Error Calculation for MAP
in11 = 0; in12 = 0; in13 = 0; in14 = 0; in21 = 0; in22 = 0; in23 = 0;...
    in24 = 0; in31 = 0; in32 = 0; in33 = 0; in34 = 0; in41 = 0; in42 = 0; ...
    in43 = 0; in44 = 0;
for d = 1:N4
    if dec(d) == 1
        if lab4(d) == 1
            in11 = in11 + 1;
        elseif lab4(d) == 2
            in12 = in12 + 1;
        elseif lab4(d) == 3
            in13 = in13 + 1;
        else
            in14 = in14 + 1;
        end
    end
    if dec(d) == 2
        if lab4(d) == 1
            in21 = in21 + 1;
        elseif lab4(d) == 2
            in22 = in22 + 1;
        elseif lab4(d) == 3
            in23 = in23 + 1;
        else
            in24 = in24 + 1;
        end
    end
    if dec(d) == 3
        if lab4(d) == 1
            in31 = in31 + 1;
        elseif lab4(d) == 2
            in32 = in32 + 1;
        elseif lab4(d) == 3
            in33 = in33 + 1;
        else
            in34 = in34 + 1;
        end
    end
    if dec(d) == 4
        if lab4(d) == 1
            in41 = in41 + 1;
        elseif lab4(d) == 2
            in42 = in42 + 1;
        elseif lab4(d) == 3
            in43 = in43 + 1;
        else
            in44 = in44 + 1;
        end
    end
end
% Error table
perr1 = (in21 + in31 + in41)/N4;
perr2 = (in12 + in32 + in42)/N4;
perr3 = (in13 + in23 + in43)/N4;
perr4 = (in14 + in24 + in34)/N4;
perr = perr1 + perr2 + perr3 + perr4;
%% Neural Network
percep = 10; 
fold = 10;
net = patternnet(1);
net.divideMode = 'none';
net.divideFcn = 'dividetrain';
net.performFcn = 'crossentropy';
net.layers{1}.transferFcn = 'logsig';
net.layers{1}.transferFcn = 'softmax';      
bestPer = zeros(1,length(N));
per = zeros(fold,1);
meanPer = zeros(percep,length(N));
for j = 1:length(N)
    if j == 1
        GMM = GMM1;
        label = Glab1;
    elseif j == 2
        GMM = GMM2;
        label = Glab2;
    else
        GMM = GMM3;
        label = Glab3;
    end
    dummy = ceil(linspace(0,N(j),fold+1));
    for k = 1:fold
        indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)];
    end
    for i = 1:percep
            % K-Fold Cross-Validation
            for k = 1:fold
                indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
                % Using fold k as validation set
                x1Val = GMM(:,indValidate); 
                x2Val = label(:,indValidate);
                if k == 1
                    indTrain = [indPartitionLimits(k,2)+1:N(j)];
                elseif k == fold
                    indTrain = [1:indPartitionLimits(k,1)-1];
                else
                    indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,2):N(j)];
                end
                x1Train = GMM(:,indTrain); 
                x2Train = label(:,indTrain);
                Ntrain1 = length(indTrain); Nvalidate = length(indValidate);
                net.layers{1}.size = i;
                net = configure(net,x1Train,x2Train);
                net.trainParam.showWindow = true;
                net = train(net,x1Train,x2Train);
                yy = net(x1Val);
                per(k) = perform(net,x2Val,yy);
            end
            meanPer(i,j) = mean(per);
            if i == 1
                bestNet = net;
                bestPer = meanPer(1,j);
            else
                if meanPer(i,j) < bestPer
                    bestNet = net;
                    bestPer = meanPer(i,j);
                end
            end       
    end
    % Train again with best neural net
    if j == 1   
        bestNet1 = configure(bestNet,GMM1,label);
        bestNet1.trainParam.showWindow = true;
        bestNet1 = train(bestNet,GMM1,label);
        yy1 = bestNet1(GMM4);
        perftest1 = perform(bestNet1,Glab4,yy1);
    elseif j == 2
        bestNet2 = configure(bestNet,GMM2,label);
        bestNet2.trainParam.showWindow = true;
        bestNet2 = train(bestNet,GMM2,label);
        yy2 = bestNet2(GMM4);
        perftest2 = perform(bestNet2,Glab4,yy2);
    else
        bestNet3 = configure(bestNet,GMM3,label);
        bestNet3.trainParam.showWindow = true;
        bestNet3 = train(bestNet,GMM3,label);
        yy3 = bestNet3(GMM4);
        perftest3 = perform(bestNet3,Glab4,yy3);
    end
end

% Visualization
figure(3)
stem(meanPer(:,1))
title("Cross-Entropy Performance for N = 100")
xlabel("Perceptrons #") 
ylabel("Average Probability of Error")
grid on
figure(4)
stem(meanPer(:,2))
title("Cross-Entropy Performance for N = 1000")
xlabel("Perceptrons #") 
ylabel("Average Probability of Error")
grid on
figure(5)
stem(meanPer(:,3))
title("Cross-Entropy Performance for N = 10000")
xlabel("Perceptrons #") 
ylabel("Average Probability of Error")
grid on
% Error Calculation
[mData1, mLab1] = max(yy1);
[mData2, mLab2] = max(yy2);
[mData3, mLab3] = max(yy3);
val1Err = 0;
val1Err1 = 0;
val1Err2 = 0;
val1Err3 = 0;
val1Err4 = 0;
for d = 1:N4
    if mLab1(d) ~= lab4(d)
        if lab4(d) == 1
            val1Err1 = val1Err1 + 1;
        elseif lab4(d) == 2
            val1Err2 = val1Err2 + 1;
        elseif lab4(d) == 3
            val1Err3 = val1Err3 + 1;
        else
            val1Err4 = val1Err4 + 1;
        end
        val1Err = val1Err+1;
    end
end
val2Err = 0;
val2Err1 = 0;
val2Err2 = 0;
val2Err3 = 0;
val2Err4 = 0;
for d = 1:N4
    if mLab2(d) ~= lab4(d)
        if lab4(d) == 1
            val2Err1 = val2Err1 + 1;
        elseif lab4(d) == 2
            val2Err2 = val2Err2 + 1;
        elseif lab4(d) == 3
            val2Err3 = val2Err3 + 1;
        else
            val2Err4 = val2Err4 + 1;
        end
        val2Err = val2Err+1;
    end
end
val3Err = 0;
val3Err1 = 0;
val3Err2 = 0;
val3Err3 = 0;
val3Err4 = 0;
for d = 1:N4
    if mLab3(d) ~= lab4(d)
        if lab4(d) == 1
            val3Err1 = val3Err1 + 1;
        elseif lab4(d) == 2
            val3Err2 = val3Err2 + 1;
        elseif lab4(d) == 3
            val3Err3 = val3Err3 + 1;
        else
            val3Err4 = val3Err4 + 1;
        end
        val3Err = val3Err+1;
    end
end
figure(6)
scatter3(GMM4(1,mLab1==1&lab4==1),GMM4(2,mLab1==1&lab4==1),GMM4(3,mLab1==1&lab4==1),'ob'); hold on
scatter3(GMM4(1,mLab1==2&lab4==1),GMM4(2,mLab1==2&lab4==1),GMM4(3,mLab1==2&lab4==1),'xr');
scatter3(GMM4(1,mLab1==3&lab4==1),GMM4(2,mLab1==3&lab4==1),GMM4(3,mLab1==3&lab4==1),'xk');
scatter3(GMM4(1,mLab1==4&lab4==1),GMM4(2,mLab1==4&lab4==1),GMM4(3,mLab1==4&lab4==1),'xg');
scatter3(GMM4(1,mLab1==1&lab4==2),GMM4(2,mLab1==1&lab4==2),GMM4(3,mLab1==1&lab4==2),'xb');
scatter3(GMM4(1,mLab1==2&lab4==2),GMM4(2,mLab1==2&lab4==2),GMM4(3,mLab1==2&lab4==2),'or');
scatter3(GMM4(1,mLab1==3&lab4==2),GMM4(2,mLab1==3&lab4==2),GMM4(3,mLab1==3&lab4==2),'xk');
scatter3(GMM4(1,mLab1==4&lab4==2),GMM4(2,mLab1==4&lab4==2),GMM4(3,mLab1==4&lab4==2),'xg');
scatter3(GMM4(1,mLab1==1&lab4==3),GMM4(2,mLab1==1&lab4==3),GMM4(3,mLab1==1&lab4==3),'xb');
scatter3(GMM4(1,mLab1==2&lab4==3),GMM4(2,mLab1==2&lab4==3),GMM4(3,mLab1==2&lab4==3),'xr');
scatter3(GMM4(1,mLab1==3&lab4==3),GMM4(2,mLab1==3&lab4==3),GMM4(3,mLab1==3&lab4==3),'ok');
scatter3(GMM4(1,mLab1==4&lab4==3),GMM4(2,mLab1==4&lab4==3),GMM4(3,mLab1==4&lab4==3),'xg');
scatter3(GMM4(1,mLab1==1&lab4==4),GMM4(2,mLab1==1&lab4==4),GMM4(3,mLab1==1&lab4==4),'xb');
scatter3(GMM4(1,mLab1==2&lab4==4),GMM4(2,mLab1==2&lab4==4),GMM4(3,mLab1==2&lab4==4),'xr');
scatter3(GMM4(1,mLab1==3&lab4==4),GMM4(2,mLab1==3&lab4==4),GMM4(3,mLab1==3&lab4==4),'xk');
scatter3(GMM4(1,mLab1==4&lab4==4),GMM4(2,mLab1==4&lab4==4),GMM4(3,mLab1==4&lab4==4),'og');
title('MLP-Classified Data for N = 100')
legend('Decision 1 & Label 1', 'Decision 2 & Label 1', 'Decision 3 & Label 1', 'Decision 4 & Label 1', ...
    'Decision 1 & Label 2', 'Decision 2 & Label 2', 'Decision 3 & Label 2', 'Decision 4 & Label 2', ...
    'Decision 1 & Label 3', 'Decision 2 & Label 3', 'Decision 3 & Label 3', 'Decision 4 & Label 3', ...
    'Decision 1 & Label 4', 'Decision 2 & Label 4', 'Decision 3 & Label 4', 'Decision 4 & Label 4')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
figure(7)
scatter3(GMM4(1,mLab2==1&lab4==1),GMM4(2,mLab2==1&lab4==1),GMM4(3,mLab2==1&lab4==1),'ob'); hold on
scatter3(GMM4(1,mLab2==2&lab4==1),GMM4(2,mLab2==2&lab4==1),GMM4(3,mLab2==2&lab4==1),'xr');
scatter3(GMM4(1,mLab2==3&lab4==1),GMM4(2,mLab2==3&lab4==1),GMM4(3,mLab2==3&lab4==1),'xk');
scatter3(GMM4(1,mLab2==4&lab4==1),GMM4(2,mLab2==4&lab4==1),GMM4(3,mLab2==4&lab4==1),'xg');
scatter3(GMM4(1,mLab2==1&lab4==2),GMM4(2,mLab2==1&lab4==2),GMM4(3,mLab2==1&lab4==2),'xb');
scatter3(GMM4(1,mLab2==2&lab4==2),GMM4(2,mLab2==2&lab4==2),GMM4(3,mLab2==2&lab4==2),'or');
scatter3(GMM4(1,mLab2==3&lab4==2),GMM4(2,mLab2==3&lab4==2),GMM4(3,mLab2==3&lab4==2),'xk');
scatter3(GMM4(1,mLab2==4&lab4==2),GMM4(2,mLab2==4&lab4==2),GMM4(3,mLab2==4&lab4==2),'xg');
scatter3(GMM4(1,mLab2==1&lab4==3),GMM4(2,mLab2==1&lab4==3),GMM4(3,mLab2==1&lab4==3),'xb');
scatter3(GMM4(1,mLab2==2&lab4==3),GMM4(2,mLab2==2&lab4==3),GMM4(3,mLab2==2&lab4==3),'xr');
scatter3(GMM4(1,mLab2==3&lab4==3),GMM4(2,mLab2==3&lab4==3),GMM4(3,mLab2==3&lab4==3),'ok');
scatter3(GMM4(1,mLab2==4&lab4==3),GMM4(2,mLab2==4&lab4==3),GMM4(3,mLab2==4&lab4==3),'xg');
scatter3(GMM4(1,mLab2==1&lab4==4),GMM4(2,mLab2==1&lab4==4),GMM4(3,mLab2==1&lab4==4),'xb');
scatter3(GMM4(1,mLab2==2&lab4==4),GMM4(2,mLab2==2&lab4==4),GMM4(3,mLab2==2&lab4==4),'xr');
scatter3(GMM4(1,mLab2==3&lab4==4),GMM4(2,mLab2==3&lab4==4),GMM4(3,mLab2==3&lab4==4),'xk');
scatter3(GMM4(1,mLab2==4&lab4==4),GMM4(2,mLab2==4&lab4==4),GMM4(3,mLab2==4&lab4==4),'og');
title('MLP-Classified Data for N = 1000')
legend('Decision 1 & Label 1', 'Decision 2 & Label 1', 'Decision 3 & Label 1', 'Decision 4 & Label 1', ...
    'Decision 1 & Label 2', 'Decision 2 & Label 2', 'Decision 3 & Label 2', 'Decision 4 & Label 2', ...
    'Decision 1 & Label 3', 'Decision 2 & Label 3', 'Decision 3 & Label 3', 'Decision 4 & Label 3', ...
    'Decision 1 & Label 4', 'Decision 2 & Label 4', 'Decision 3 & Label 4', 'Decision 4 & Label 4')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')
figure(8)
scatter3(GMM4(1,mLab3==1&lab4==1),GMM4(2,mLab3==1&lab4==1),GMM4(3,mLab3==1&lab4==1),'ob'); hold on
scatter3(GMM4(1,mLab3==2&lab4==1),GMM4(2,mLab3==2&lab4==1),GMM4(3,mLab3==2&lab4==1),'xr');
scatter3(GMM4(1,mLab3==3&lab4==1),GMM4(2,mLab3==3&lab4==1),GMM4(3,mLab3==3&lab4==1),'xk');
scatter3(GMM4(1,mLab3==4&lab4==1),GMM4(2,mLab3==4&lab4==1),GMM4(3,mLab3==4&lab4==1),'xg');
scatter3(GMM4(1,mLab3==1&lab4==2),GMM4(2,mLab3==1&lab4==2),GMM4(3,mLab3==1&lab4==2),'xb');
scatter3(GMM4(1,mLab3==2&lab4==2),GMM4(2,mLab3==2&lab4==2),GMM4(3,mLab3==2&lab4==2),'or');
scatter3(GMM4(1,mLab3==3&lab4==2),GMM4(2,mLab3==3&lab4==2),GMM4(3,mLab3==3&lab4==2),'xk');
scatter3(GMM4(1,mLab3==4&lab4==2),GMM4(2,mLab3==4&lab4==2),GMM4(3,mLab3==4&lab4==2),'xg');
scatter3(GMM4(1,mLab3==1&lab4==3),GMM4(2,mLab3==1&lab4==3),GMM4(3,mLab3==1&lab4==3),'xb');
scatter3(GMM4(1,mLab3==2&lab4==3),GMM4(2,mLab3==2&lab4==3),GMM4(3,mLab3==2&lab4==3),'xr');
scatter3(GMM4(1,mLab3==3&lab4==3),GMM4(2,mLab3==3&lab4==3),GMM4(3,mLab3==3&lab4==3),'ok');
scatter3(GMM4(1,mLab3==4&lab4==3),GMM4(2,mLab3==4&lab4==3),GMM4(3,mLab3==4&lab4==3),'xg');
scatter3(GMM4(1,mLab3==1&lab4==4),GMM4(2,mLab3==1&lab4==4),GMM4(3,mLab3==1&lab4==4),'xb');
scatter3(GMM4(1,mLab3==2&lab4==4),GMM4(2,mLab3==2&lab4==4),GMM4(3,mLab3==2&lab4==4),'xr');
scatter3(GMM4(1,mLab3==3&lab4==4),GMM4(2,mLab3==3&lab4==4),GMM4(3,mLab3==3&lab4==4),'xk');
scatter3(GMM4(1,mLab3==4&lab4==4),GMM4(2,mLab3==4&lab4==4),GMM4(3,mLab3==4&lab4==4),'og');
title('MLP-Classified Data for N = 10000')
legend('Decision 1 & Label 1', 'Decision 2 & Label 1', 'Decision 3 & Label 1', 'Decision 4 & Label 1', ...
    'Decision 1 & Label 2', 'Decision 2 & Label 2', 'Decision 3 & Label 2', 'Decision 4 & Label 2', ...
    'Decision 1 & Label 3', 'Decision 2 & Label 3', 'Decision 3 & Label 3', 'Decision 4 & Label 3', ...
    'Decision 1 & Label 4', 'Decision 2 & Label 4', 'Decision 3 & Label 4', 'Decision 4 & Label 4')
xlabel('x_1')
ylabel('x_2')
zlabel('x_3')

%% Functions
function [x, labels] = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
    labels(ind) = m;
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
