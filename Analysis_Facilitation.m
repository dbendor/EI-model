%% New model data including facilitation.
%Facilitation. recreate the data file 

load('D:\John_Wanglab\Bendorlab backup\dataaaaa2.mat')

% UnitInfo.List : {f_dE, f_fI, tau_pE, tau_pI, E, I}
% E and I stays constant.

X1 = zeros(length(unique(UnitInfo.List(:,1))),length(unique(UnitInfo.List(:,2)))); % Matrix of Rho depending on adaptation values
% X3 = X1;
for n = 1:length(UnitInfo.List)
    if UnitInfo.Info(n).Pval < 0.05
        
        X1(floor(UnitInfo.List(n,1)*10)+1,floor(UnitInfo.List(n,2)*10)+1) = X1(floor(UnitInfo.List(n,1)*10)+1,floor(UnitInfo.List(n,2)*10)+1) + UnitInfo.Info(n).Rho;
    end
%     X3(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) = X3(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) + UnitInfo.Info(n).Output.mean_discharge_rate.mean(11);
end



% X1 = zeros(length(unique(UnitInfo.List(:,1))),length(unique(UnitInfo.List(:,2)))); % Matrix of Rho depending on adaptation values
% X3 = X1;
% for n = 1:length(UnitInfo.List)
%     if UnitInfo.Info(n).Pval < 0.05
%         
%         X1(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) = X1(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) + UnitInfo.Info(n).Rho;
%     end
% %     X3(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) = X3(floor(UnitInfo.List(n,1)*10),floor(UnitInfo.List(n,2)*10)) + UnitInfo.Info(n).Output.mean_discharge_rate.mean(11);
% end

% for new stuff 29/08/2018
xticks([1:5])
xticklabels({'0','0.1','0.2','0.3','0.4'});


yticks([1:5])
yticklabels({'0','0.1','0.2','0.3','0.4'});

% end new stuff
X1 = X1/(8*8);
X3 = X3/64;

figure
imagesc(-X1)
colormap(flipud(jet))
colorbar
xticks([1:9])
xticklabels({'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'})
xlabel('f_DI')
yticks([1:3])
yticklabels({'0.1','0.2','0.3'})
ylabel('f_DE')
figure
imagesc(X3)
xticks([1:9])
xticklabels({'0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9'})
xlabel('f_DI')
yticks([1:3])
yticklabels({'0.1','0.2','0.3'})
ylabel('f_DE')

X2 = {};
figure
i = 1;
f_dE = 0.3;
tau_pE = unique(UnitInfo.List(:,3));
tau_pI = unique(UnitInfo.List(:,4));
for f_dI = 0.1:0.1:0.9
    subplot(2,5,i)
    X2{i} = zeros(8,8);
    
    x = find(UnitInfo.List(:,1)==f_dE & UnitInfo.List(:,2)==f_dI);
    for n = 1:length(x)
        if UnitInfo.Info(x(n)).Pval < 0.05
        X2{i}(find(tau_pE == UnitInfo.List(x(n),3)),tau_pI == UnitInfo.List(x(n),4)) =  X2{i}(find(tau_pE == UnitInfo.List(x(n),3)),tau_pI == UnitInfo.List(x(n),4))+ UnitInfo.Info(x(n)).Rho;
        end
    end
    imagesc(X2{i})
    colorbar
    caxis([-1 1])
    i = i+1;
end

 figure
subplot(1,3,2)
f_dE = 0.2;
f_dI = 0.9;
X2{1} = zeros(8,8);

x = find(round(UnitInfo.List(:,1)*10)==f_dE*10 & round(UnitInfo.List(:,2)*10)==f_dI*10);
for n = 1:length(x)
    if UnitInfo.Info(x(n)).Pval < 0.05
        
        X2{1}(find(tau_pE == UnitInfo.List(x(n),3)),tau_pI == UnitInfo.List(x(n),4)) =  X2{1}(find(tau_pE == UnitInfo.List(x(n),3)),tau_pI == UnitInfo.List(x(n),4))+ UnitInfo.Info(x(n)).Rho;
    end
end
imagesc(X2{1})
colorbar
caxis([-1 1])
