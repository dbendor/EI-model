%% Puretone  06/06/2018

% real data, median spike time comparisons. 
SN = load('med_spk_PT_SN.mat');
SP = load('med_spk_PT_SP.mat');
SN.median_spike = SN.median_spike-0.2;
SP.median_spike = SP.median_spike-0.2;
SN.median_spike = SN.median_spike(find(SN.median_spike<0.2));
SP.median_spike = SP.median_spike(find(SP.median_spike<0.2));
[p,h,stats] = ranksum(SN.median_spike,SP.median_spike)
% [p,h,stats] = ranksum(SN.median_spike,SP.median_spike)
data_mean = mean(SN.median_spike);
data_error = std(SN.median_spike);
data_mean(2) = mean(SP.median_spike);
data_error(2) = std(SP.median_spike);
figure
bar(data_mean)
hold on 
e = errorbar(data_mean,data_error,'.','CapSize',18,'LineWidth',2);
e.Color = 'black';
e.CapSize = 18;

edges = [0:0.025:0.2];
figure
histogram(SN.median_spike,edges)
hold on
histogram(SP.median_spike,edges)


% model data
load('puretoneModel.mat');
mdlSN.spiketime = UnitInfo.Info(3).Output.spiketime{1}(find(UnitInfo.Info(3).Output.spiketime{1}>0));
mdlSN.spiketime = mdlSN.spiketime(find(mdlSN.spiketime<0.2));

median(mdlSN.spiketime)

mdlSP.spiketime = UnitInfo.Info(2).Output.spiketime{1}(find(UnitInfo.Info(1).Output.spiketime{1}>0));
mdlSP.spiketime = mdlSP.spiketime(find(mdlSP.spiketime<0.2));

median(mdlSP.spiketime)

[p,h,stats] = ranksum(mdlSN.spiketime,mdlSP.spiketime)

% model with SFA

%% Median spike time for different adaptation MOdel




tau_PE = unique(UnitInfo.List(:,3)).'; %x
tau_PI = unique(UnitInfo.List(:,4)).';%y

f_DE = unique(UnitInfo.List(:,1)).' ; %a
f_DI = unique(UnitInfo.List(:,2)).' ; %b 


Z_positive1 = zeros(length(tau_PE),length(tau_PI));



for x = 1:length(f_DE)
    for y = 1:length(f_DI)
        collect = [];
        idx = find(UnitInfo.List(:,1) == f_DE(x) & UnitInfo.List(:,2) == f_DI(y)).';
        for z = idx 
            spks = [];
            for t = 1:length(UnitInfo.Info(z).Output.spiketime{1, 1})
                if UnitInfo.Info(z).Output.spiketime{1, 1}(t) >0 && UnitInfo.Info(z).Output.spiketime{1, 1}(t)<0.2
            spks = [spks UnitInfo.Info(z).Output.spiketime{1, 1}(t)];
                end
            end
            collect1 = [collect median(spks)];
                Z_positive1(x,y) = mean(collect1);
        end
%         toc
    end
end

figure
colormap(flipud(hot))
imagesc(Z_positive1)
caxis([0 0.14])



% 
% 
% 
% 
% Z_positive1 = zeros(length(tau_PE),length(tau_PI));
% Z_negative1 = zeros(length(tau_PE),length(tau_PI));
% Z = zeros(length(f_DE),length(f_DI));
% Z_onset = zeros(length(f_DE),length(f_DI));
% 
% 
% for x = 1:length(tau_PE)
%     for y = 1:length(tau_PI)
% %         tic
%         idx = find(UnitInfo.List(:,3) == tau_PE(x) & UnitInfo.List(:,4) == tau_PI(y)).';
%         for z = idx 
%             if UnitInfo.Info(z).Positive ==1 %1 is positive
%                 Z_positive1(x,y) = Z_positive1(x,y) - UnitInfo.Info(z).Rho;
%             elseif UnitInfo.Info(z).Positive == -1
%                 Z_negative1(x,y) = Z_negative1(x,y) - UnitInfo.Info(z).Rho;
%             end
%         end
% %         toc
%     end
% end
% 
% 
% 
% for a = 1:length(f_DE)
%     for b = 1:length(f_DI)
%         z = find(UnitInfo.List(:,1) == f_DE(a) & UnitInfo.List(:,2) == f_DI(b) &...
%             UnitInfo.List(:,3) == tau_PE(7) & UnitInfo.List(:,4) == tau_PI(6));
%         Z(a,b) = -UnitInfo.Info(z).Rho;
%         Z_onset(a,b) = UnitInfo.Info(z).Output.spikes_per_click{1}.mean(1);
%     end
% end
% 
% 
% Z_positive1 = Z_positive1/36; 
% Z_negative1 = Z_negative1/36;
% 
% A_DE = 1-f_DE;
% A_DI = 1-f_DI;
% 
% figure
% imagesc(tau_PI,tau_PE,Z_positive1,[-1, 1])
% % surf(tau_PI,tau_PE,Z_positive1)
ylabel('\tau_pE')
xlabel('\tau_pI')
zlabel('Rho')
title('Sync+')
colormap(flipud(jet))
set(gca, 'FontSize', 16)
colorbar
set(colorbar, 'ylim', [0 1])

figure
imagesc(tau_PI,tau_PE,Z_negative1,[-1, 1])
% surf(tau_PI,tau_PE,Z_negative1)
ylabel('\tau_pE')
xlabel('\tau_pI')
zlabel('Rho')
title('Sync-')
colormap(flipud(jet))
set(gca, 'FontSize', 16)
colorbar
set(colorbar, 'ylim', [-1 0])

% figure

%% amplitude for MI

load('modeldataampSP.mat')
X1 = [];
for p = 1:10
    X1 = [X1;UnitInfo.Info(8).Output.mean_discharge_rate.mean + (p-1)*10 + randn*3];
end




load('modeldataampSN.mat')
X2 = [];
for p = 1:10
    X2 = [X2;UnitInfo.Info(8).Output.mean_discharge_rate.mean + (p-1)*10 + randn*3];
end



figure
subplot(2,1,1)
plot(mean(X1,1)/max(mean(X1,1)))
axis([-inf inf 0 1.2])
subplot(2,1,2)
plot(mean(X1,2)/max(mean(X1,2)))
axis([-inf inf 0 1.2])
title('SP')
figure
subplot(2,1,1)
plot(mean(X2,1)/max(mean(X2,1)))
axis([-inf inf 0 1.2])

subplot(2,1,2)
plot(mean(X2,2)/max(mean(X2,2)))
axis([-inf inf 0 1.2])
title('SN')

X3 = X1 + X2;
figure
subplot(2,1,1)
plot(mean(X3,1)/max(mean(X3,1)))
axis([-inf inf 0 1.2])

subplot(2,1,2)
plot(mean(X3,2)/max(mean(X3,2)))
axis([-inf inf 0 1.2])
title('SP+SN')

X4 = X1 - X2;

figure
subplot(2,1,1)
plot(mean(X4,1)/max(mean(X4,1)))
axis([-inf inf 0 1.2])

subplot(2,1,2)
plot(mean(X4,2)/max(mean(X4,1)))
axis([-inf inf 0 1.2])
title('SP-SN')


figure
subplot(2,2,1)
imagesc(X1)
subplot(2,2,2)
imagesc(X2)
subplot(2,2,3)
imagesc(X3)
subplot(2,2,4)
imagesc(X4)
colormap(flipud(hot))
% caxis([0 220])
