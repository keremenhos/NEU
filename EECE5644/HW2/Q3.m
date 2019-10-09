%% Q3-1
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = eye(2);
cov2 = eye(2);
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
s1 = max(min([LDA1 LDA2]));
s2 = min(max([LDA1 LDA2]));
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');

%% Q3-2
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = [3,1; 1 0.8];
cov2 = [3,1; 1 0.8];
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
s1 = max(min([LDA1 LDA2]));
s2 = min(max([LDA1 LDA2]));
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');
%% Q3-3
mu1 = [0 0]';
mu2 = [2 2]';
cov1 = [2,0.5; 0.5 1];
cov2 = [2, -1.9; -1.9 5];
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
s1 = max(min([LDA1 LDA2]));
s2 = min(max([LDA1 LDA2]));
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,?1.9;?1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,-1.9;-1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');
%% Q3-4
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = eye(2);
cov2 = eye(2);
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
m(1) = min(LDA1);
m(2) = min(LDA2);
s1 = max(m);
m(1) = max(LDA1);
m(2) = max(LDA2);
s2 = min(m);
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');

%% Q3-5
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = [3,1; 1 0.8];
cov2 = [3,1; 1 0.8];
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
m(1) = min(LDA1);
m(2) = min(LDA2);
s1 = max(m);
m(1) = max(LDA1);
m(2) = max(LDA2);
s2 = min(m);
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');
%% Q3-6
mu1 = [0 0]';
mu2 = [2 2]';
cov1 = [2,0.5; 0.5 1];
cov2 = [2, -1.9; -1.9 5];
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);

mu11 = mean(P1)';
mu22 = mean(P2)';
cov11 = cov(P1);
cov22 = cov(P2);
sB = (mu11-mu22)*(mu11-mu22)';
sW = (cov11 + cov22);
[W, D] = eig(sB,sW);
[~, sortInd]=sort(diag(D),'descend');
W=W(:,max(sortInd));
% W2 = sW\(mu11-mu22);
LDA1 = P1*W;
LDA2 = P2*W;
del=0.002;
e = min([LDA1;LDA2]):del:max([LDA1;LDA2]);
N1 = histcounts(LDA1,e);
N2 = histcounts(LDA2,e);
m(1) = min(LDA1);
m(2) = min(LDA2);
s1 = max(m);
m(1) = max(LDA1);
m(2) = max(LDA2);
s2 = min(m);
error = zeros(1,find((e > s2 - del/2),1)-find((e > s1 - del/2),1));
ci = 1;
for i = find((e > s1 - del/2),1):find((e > s2 - del/2),1)
    if mean(LDA1) < mean(LDA2)
        error(ci) = sum(N1(i:end))+sum(N2(1:i));
        ci = ci + 1;
    else
        error(ci) = sum(N2(i:end))+sum(N1(1:i));
        ci = ci + 1;
    end
end
minErrInd = find(error == min(error));
decInd = find((e > s1 - del/2),1)+minErrInd(randi(length(minErrInd)))-2;
if mean(LDA1)<mean(LDA2)
    err1 = sum(N1(decInd:end));
    err2 = sum(N2(1:decInd));
    pErr1 = 100*err1/400;
    pErr2 = 100*err2/400;
else
    err2 = sum(N2(decInd:end));
    err1 = sum(N1(1:decInd));
    pErr2 = 100*err2/400;
    pErr1 = 100*err1/400;
end
figure
plot(LDA1,0,'bo')
hold on
plot(LDA2,0,'r*')
line([e(decInd) e(decInd)],[-1 1],'LineWidth',1,'Color','k')
ylim([-1 1])
LH(1) = plot(nan, nan, 'bo');
LH(2) = plot(nan, nan, 'r*');
LH(3) = plot(nan, nan, 'k-');
legend(LH,'\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,-1.9;-1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Projection');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(err1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(err2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(pErr1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(pErr2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
figure
stem(e(1:end-1),N1,'bo')
hold on
stem(e(1:end-1),N2,'r*')
line([e(decInd) e(decInd)],[0.1 max([N1 N2])+1],'LineWidth',1,'Color','k')
ylim([0.1 max([N1 N2])+1])
grid on
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,-1.9;-1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Fisher LDA Scores');
