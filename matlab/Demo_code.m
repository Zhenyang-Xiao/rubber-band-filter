clearvars;
%% clear path
restoredefaultpath;
rng(0);
dt = .01;
t = [0:dt:10]';
V = 2*t + randn(size(t))*0.2 + sin(2*pi*1.5*t+3*pi/8);
fc = 2;
[Vf,info] = Rubber_Band_Filter(t,V,fc,'tol',1e-7,'niter',80);
% [Vf1,info1] = Rubber_Band_Filter_GCV(tdn(:,1),tdn(:,2),fc,'tol',1e-11,'niter',50);
%% make plots
figure;
plot(t,V); hold on
plot(t,Vf);
% plot(tdn(:,1),Vf1);
legend('Initial','Filtered')