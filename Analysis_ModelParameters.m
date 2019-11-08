%% Figure 5 on the previous model, and model parameters 
%%% revision 18/06/2018 Need to recreate data file for model responses across parameters.  
% load('model_datasync2.mat')
tau_PE = unique(UnitInfo.List(:,3)).'; %x
tau_PI = unique(UnitInfo.List(:,4)).';%y
% E =  unique(UnitInfo.List(:,5)).';
% I =  unique(UnitInfo.List(:,6)).';
f_DE = 0.5:0.1:1. ; %a
f_DI = 0.5:0.1:1. ; %b 

Z_positive1 = zeros(length(tau_PE),length(tau_PI));
Z_negative1 = zeros(length(tau_PE),length(tau_PI));
Z = zeros(length(f_DE),length(f_DI));
Z_onset = zeros(length(f_DE),length(f_DI));


for x = 1:length(tau_PE)
    for y = 1:length(tau_PI)
%         tic
        idx = find(UnitInfo.List(:,3) == tau_PE(x) & UnitInfo.List(:,4) == tau_PI(y)).';
        for z = idx 
            if UnitInfo.Info(z).Positive ==1 %1 is positive
                Z_positive1(x,y) = Z_positive1(x,y) - UnitInfo.Info(z).Rho;
            elseif UnitInfo.Info(z).Positive == -1
                Z_negative1(x,y) = Z_negative1(x,y) - UnitInfo.Info(z).Rho;
            end
        end
%         toc
    end
end



for a = 1:length(f_DE)
    for b = 1:length(f_DI)
        z = find(UnitInfo.List(:,1) == f_DE(a) & UnitInfo.List(:,2) == f_DI(b) &...
            UnitInfo.List(:,3) == tau_PE(7) & UnitInfo.List(:,4) == tau_PI(6));
        Z(a,b) = -UnitInfo.Info(z).Rho;
        Z_onset(a,b) = UnitInfo.Info(z).Output.spikes_per_click{1}.mean(1);
    end
end


Z_positive1 = Z_positive1/36; 
Z_negative1 = Z_negative1/36;

A_DE = 1-f_DE;
A_DI = 1-f_DI;

figure
imagesc(tau_PI,tau_PE,Z_positive1,[-1, 1])
% surf(tau_PI,tau_PE,Z_positive1)
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
% imagesc(A_DI,A_DE,Z)
% % surf(f_DI,f_DE,Z)
% ylabel('A_DE')
% xlabel('A_DI')
% zlabel('Rho')
% colormap(flipud(jet))
% title('with optimal \tau_pE and \tau_pI')
%     set(gca, 'FontSize', 16)
% 
% figure
% imagesc(A_DI,A_DE,Z_onset)
% % surf(f_DI,f_DE,Z_onset)
% ylabel('A_DE')
% xlabel('A_DI')
% zlabel('rate (spikes/s')
% title('onset response')
% % colormap hot 
% % size(Z_positive)
% set(gca, 'FontSize', 16)
% test =1;
% 
% 
% % %%% from here, EI 
% idx = find([UnitInfo.Info.Positive] == 1 & [UnitInfo.Info.Significant_rate] == 1);
% idx2 = find([UnitInfo.Info.Positive] == 0 & [UnitInfo.Info.Significant_rate] == 1);
% idx3 = find([UnitInfo.Info.Positive] == -1 & [UnitInfo.Info.Significant_rate] == 1);
% for i = 1:length(idx)
%     x1(i) = UnitInfo.List(idx(i),5);
%     y1(i) = UnitInfo.List(idx(i),6)/x1(i);
%     z1(i) = UnitInfo.Info(idx(i)).Rho;
% end
% 
% 
% 
% for i = 1:length(idx2)
%     x2(i) = UnitInfo.List(idx2(i),5);
%     y2(i) = UnitInfo.List(idx2(i),6)/x2(i);
%     z2(i) = UnitInfo.Info(idx2(i)).Rho;
% end
% 
% for i = 1:length(idx3)
%     x3(i) = UnitInfo.List(idx3(i),5);
%     y3(i) = UnitInfo.List(idx3(i),6)/x3(i);
%     z3(i) = UnitInfo.Info(idx3(i)).Rho;
% end
%  
% figure 
% cmapp = [[0.1 0.7 0.1]; [0.9 0.6 0.1]; [0 0 0]   ];%    [0.26 0.5 0.9]   ]; %; [0.9 0.3 0.26]; ]; 
% 
% scatter(x1,y1,'DisplayName','Sync+','LineWidth',1.5,'MarkerEdgeColor',cmapp(1,:))
% hold on 
% % pause
% scatter(x2,y2,'x','DisplayName','Non Significant','LineWidth',1.5,'MarkerEdgeColor',cmapp(3,:))
% % pause
% scatter(x3,y3,'DisplayName','Sync-','LineWidth',1.5,'MarkerEdgeColor',cmapp(2,:))
% axis([0, 7, 0, 7])
% set(gca, 'FontSize', 16)
% grid on
% legend show
% test=1 ;
% scatter3(x1,y1,z1)
% hold on 
% scatter3(x2,y2,z2)

