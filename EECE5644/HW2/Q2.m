%% Q2-1
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = eye(2);
cov2 = eye(2);
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:1:8;
x3 = zeros(1,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(i) = double(sol);
end

xx1=[x3(1) x3(end) x3(end)];
xx2=[x4(1) x4(end) x4(1)];
numE1 = sum(inpolygon(P1(:,1),P1(:,2),xx1,xx2)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),xx1,xx2)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'ro')
plot(x3(1,:),x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
%% Q2-2
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = [3,1; 1 0.8];
cov2 = [3,1; 1 0.8];
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:1:8;
x3 = zeros(1,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(i) = double(sol);
end

xx1=[x3(1) x3(end) x3(end)];
xx2=[x4(1) x4(end) x4(1)];
numE1 = sum(inpolygon(P1(:,1),P1(:,2),xx1,xx2)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),xx1,xx2)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'ro')
plot(x3(1,:),x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
%% Q2-3
mu1 = [0 0]';
mu2 = [2 2]';
cov1 = [2,0.5; 0.5 1];
cov2 = [2, -1.9; -1.9 5];
w1 = 0.5;
w2 = 0.5;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:0.1:8;
x3 = zeros(2,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(:,i) = double(sol);
end

numE1 = sum(inpolygon(P1(:,1),P1(:,2),x3(1,:),x4)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),x3(1,:),x4)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'ro')
plot(x3(1,:),x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,-1.9;-1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=P(\omega_{2})=0.5';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
%% Q2-4
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = eye(2);
cov2 = eye(2);
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:1:8;
x3 = zeros(1,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(i) = double(sol);
end

xx1=[x3(1) x3(end) x3(end)];
xx2=[x4(1) x4(end) x4(1)];
numE1 = sum(inpolygon(P1(:,1),P1(:,2),xx1,xx2)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),xx1,xx2)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'ro')
plot(x3,x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = I', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = I','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
%% Q2-5
mu1 = [0 0]';
mu2 = [3 3]';
cov1 = [3,1; 1 0.8];
cov2 = [3,1; 1 0.8];
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:1:8;
x3 = zeros(1,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(i) = double(sol);
end

xx1=[x3(1) x3(end) x3(end)];
xx2=[x4(1) x4(end) x4(1)];
numE1 = sum(inpolygon(P1(:,1),P1(:,2),xx1,xx2)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),xx1,xx2)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'r*')
plot(x3,x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [3,1;1,0.8]', '\fontname{Times New Roman} \omega_{2}, \mu = [3,3]^{\itT}, \Sigma = [3,1;1,0.8]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
%% Q2-6
mu1 = [0 0]';
mu2 = [2 2]';
cov1 = [2,0.5; 0.5 1];
cov2 = [2, -1.9; -1.9 5];
w1 = 0.05;
w2 = 0.95;
P1 = mvnrnd(mu1,cov1,400*w1);
P2 = mvnrnd(mu2,cov2,400*w2);
x4 = -8:0.1:8;
x3 = zeros(2,length(x4));
invcov1 = inv(cov1);
invcov2 = inv(cov2);
a = invcov1-invcov2;
b = 2.*(cov2\mu2-cov1\mu1)';
c = mu1'/cov1*mu1-mu2'/cov2*mu2-2.*log(w1/w2);
syms x1
for i = 1:length(x4)
    x2=x4(i);
    sol = vpasolve(a(1)*(x1.^2)+(a(2)+a(3)).*(x1.*x2)+a(4).*(x2.^2)+b(1).*x1+b(2).*x2+c == 0, x1);
    x3(:,i) = double(sol);
end

numE1 = sum(inpolygon(P1(:,1),P1(:,2),x3(1,:),x4)==0);
numE2 = sum(inpolygon(P2(:,1),P2(:,2),x3(1,:),x4)==1);
probE1 = 100*(numE1/400);
probE2 = 100*(numE2/400);

figure
scatter(P1(:,1),P1(:,2),24,'bo')
hold on
scatter(P2(:,1),P2(:,2),24,'ro')
plot(x3(1,:),x4,'k')
legend('\fontname{Times New Roman} \omega_{1}, \mu = [0,0]^{\itT}, \Sigma = [2,0.5;0.5,1]', '\fontname{Times New Roman} \omega_{2}, \mu = [2,2]^{\itT}, \Sigma = [2,-1.9;-1.9,5]','\fontname{Times New Roman} Decision Boundary');
xlabel('\fontname{Times New Roman} x_{1}');
ylabel('\fontname{Times New Roman} x_{2}');
title('Densities of 2 Classes with 400 Samples of Bivariate Gaussian Distribution');
str = '\fontname{Times New Roman} P(\omega_{1})=0.05 P(\omega_{2})=0.95';
str1 = ['\fontname{Times New Roman} # of Errors for \omega_{1} = ' num2str(numE1)];
str2 = ['\fontname{Times New Roman} # of Errors for \omega_{2} = ' num2str(numE2)];
str3 = ['\fontname{Times New Roman} P_{E}(\omega_{1}) = ' num2str(probE1) '%'];
str4 = ['\fontname{Times New Roman} P_{E}(\omega_{2}) = ' num2str(probE2) '%'];
str = {str str1, str2, str3, str4};
dim = [.2 .5 .3 .3];
annotation('textbox',dim,'String',str,'FitBoxToText','on')
