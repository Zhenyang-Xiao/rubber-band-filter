clearvars;
%% clear path
restoredefaultpath;
td=load('example data\TDS_data.mat');
tdn=td.tdn;
fc=4.5;
[Vf,info] = Rubber_Band_Filter(tdn(:,1),tdn(:,2),fc,'exitmode','thresh','tol',1e-11);
[Vf1,info1] = Rubber_Band_Filter_GCV(tdn(:,1),tdn(:,2),fc,'exitmode','thresh','tol',1e-11,'niter',50);
%% make plots
figure;
plot(tdn(:,1),tdn(:,2)); hold on
plot(tdn(:,1),Vf);
plot(tdn(:,1),Vf1);
legend('Initial','Filtered','GCV')