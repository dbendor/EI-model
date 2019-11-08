%% Testing difference between first click and last click for 8, 24, and 48Hz 
% 19/05/2019
load('SyncN_new.mat')
% load('SyncP_new.mat')
Stim_freq = 4:4:48;

diffN = [];
for f = 2:12
    diffN(:,f) = 1- (output.spikes_per_click{f}.brut1(:,end)./ ...
        (output.spikes_per_click{f}.brut1(:,1))); 

end
load('SyncP_new.mat')

diffP = [];
for f = 2:12
    diffP(:,f) = 1- (output.spikes_per_click{f}.brut1(:,end)./ ...
        (output.spikes_per_click{f}.brut1(:,1))); 

end

for f = 2:12
       [p(f), h(f), stats{f}]  = ranksum(diffN(:,f),diffP(:,f));
       median_diff(f) = median(diffN(:,f))-median(diffP(:,f));
    Zvalue(f) = stats{f}.zval; 

end

% plot(Stim_freq,median_diff)
figure
plot(Stim_freq,Zvalue)

%% Supp figure 2

% load('SyncN_new.mat')
load('SyncP_new.mat')
Stim_freq = 4:4:48;

diff = [];
for f = 2:12
    diff(:,f) = 1- (output.spikes_per_click{f}.brut1(:,end)./ ...
        (output.spikes_per_click{f}.brut1(:,1))); 
   [p(f), h(f), stats{f}]  = signrank(diff(:,f));
    Zvalue(f) = stats{f}.zval; 
end
%     figure
% y = [mean(output.spikes_per_click{2}.brut1(:,1)) mean(output.spikes_per_click{2}.brut1(:,end)); ...
%     mean(output.spikes_per_click{6}.brut1(:,1)) mean(output.spikes_per_click{6}.brut1(:,end)); ...
%     mean(output.spikes_per_click{11}.brut1(:,1)) mean(output.spikes_per_click{11}.brut1(:,end))];

y = [1 mean(output.spikes_per_click{2}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1)); ...
    1 mean(output.spikes_per_click{6}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1)); ...
    1 mean(output.spikes_per_click{11}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1))];


z = [std(output.spikes_per_click{2}.brut1(:,1))/mean(output.spikes_per_click{2}.brut1(:,1)) std(output.spikes_per_click{2}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1)); ...
    std(output.spikes_per_click{6}.brut1(:,1))/mean(output.spikes_per_click{2}.brut1(:,1)) std(output.spikes_per_click{6}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1)); ...
    std(output.spikes_per_click{11}.brut1(:,1))/mean(output.spikes_per_click{2}.brut1(:,1)) std(output.spikes_per_click{11}.brut1(:,end))/mean(output.spikes_per_click{2}.brut1(:,1))];


nn = size(output.spikes_per_click{1, 2}.brut1,1);
SE = z/nn;
ts = tinv([0.025  0.975],sqrt(nn));  % T-Score
error = ts(2)*SE;

figure
bar(y)
hold on
errorbar(y,error,'LineWidth',2);
axis([0.5 3.5 0 1.2])
xticklabels({'8Hz' '24Hz' '48Hz'})

figure
plot(Stim_freq, Zvalue,'LineWidth',2)
axis([0 50 0 5])
xlabel('Hz')
ylabel('Z-score')
% mean_diff = mean(diff,1);
% error_diff = std(diff,[],1);
% errorbar(Stim_freq,mean_diff,error_diff)


%% Fit spike per click data with exponential curves

load('SyncN_new.mat')
% load('SyncP_new.mat')
fit_raw = {};
fit_time = {};
fit_curve = {};

Stim_freq = 4:4:48;


fitOptions1 = fitoptions('poly1');
fitOptions2 = fitoptions('exp1');
fitOptions1.Normal = 'on';
fitOptions2.Normal = 'on';
fitOptions2.StartPoint = [1,-15];
fit_analys.fit = [];
fit_analys.goodness = [];

for f = 2:12
    fit_raw{f} = output.spikes_per_click{1,f}.brut1 - mean(output.spikes_per_click{1,f}.brut1,2);
    fit_raw{f} = reshape(fit_raw{f},[],1);
    fit_time{f} = output.spikes_per_click{1,f}.xaxis*1e-3;
    fit_time{f} = reshape(fit_time{f},[],1);
    [fit_curve1{f}, gof1{f}] = fit(fit_time{f},fit_raw{f},'poly1',fitOptions1);
    fit_analys.fit(f,1) = fit_curve1{f}.p1;
    fit_analys.fit(f,2) = fit_curve1{f}.p2;
    ci1 = confint(fit_curve1{f});
    fit_analys.fit(f,3) = abs(ci1(1) - fit_analys.fit(f,1));
    fit_analys.fit(f,4) = abs(ci1(3) - fit_analys.fit(f,2));
    
    [fit_curve2{f}, gof2{f}] = fit(fit_time{f},fit_raw{f},'exp1',fitOptions2);
    fit_analys.fit(f,5) = fit_curve2{f}.a;
    fit_analys.fit(f,6) = fit_curve2{f}.b;
    
    ci2 = confint(fit_curve2{f});
    fit_analys.fit(f,7) = abs(ci2(1) - fit_analys.fit(f,5));
    fit_analys.fit(f,8) = abs(ci2(3) - fit_analys.fit(f,6));
    
    fit_analys.goodness(f,1) = gof1{f}.rsquare;
    fit_analys.goodness(f,2) = gof2{f}.rsquare;
    
end


figure

for f =2:12
    subplot(3,4,f)
    scatter(fit_time{f},fit_raw{f},10,[0.4,0.4,0.4],'filled')
    hold on
    p = plot(fit_curve1{f},'r');%,'LineWidth',2)
    p.LineWidth = 2;
    p2 = plot(fit_curve2{f},'b');%,'LineWidth',2)
    p2.LineWidth = 2;
    legend('Data','linear fit','exponential fit')
    axis([0.45 1.05 -60 200])
    hold off
end


figure
subplot(2,2,1)
hold on
errorbar(Stim_freq(2:end),fit_analys.fit(2:end,1),fit_analys.fit(2:end,3))
xlabel('Stimulus repetition rate (Hz)')
ylabel('')
title('p1')

subplot(2,2,2)
hold on
errorbar(Stim_freq(2:end),fit_analys.fit(2:end,2),fit_analys.fit(2:end,4))
xlabel('Stimulus repetition rate (Hz)')
ylabel('')
title('p2')

subplot(2,2,3)
hold on
errorbar(Stim_freq(2:end),fit_analys.fit(2:end,5),fit_analys.fit(2:end,7))
xlabel('Stimulus repetition rate (Hz)')
ylabel('')
title('a')

subplot(2,2,4)
hold on

errorbar(Stim_freq(2:end),fit_analys.fit(2:end,6),fit_analys.fit(2:end,8))
xlabel('Stimulus repetition rate (Hz)')
ylabel('')
title('b')

figure
subplot(2,1,1)
hold on
plot(Stim_freq(2:end),fit_analys.goodness(2:end,1),'-o')
title('R-squared linear fit')
xlabel('Stimulus repetition rate (Hz)')
ylabel('R-squared')
axis([5 50 0 0.6])

subplot(2,1,2)
hold on
plot(Stim_freq(2:end),fit_analys.goodness(2:end,2),'-o')
title('R-squared exponential fit')
xlabel('Stimulus repetition rate (Hz)')
ylabel('R-squared')
axis([5 50 0 0.6])




%%
% %48Hz adaptation
% xaxis = [];
% brut = [];
% for i = 1:16
%     xaxis = [xaxis output.spikes_per_click{1, 11}.xaxis];
%     brut = [brut output.spikes_per_click{1, 11}.brut2(i,:)];%/output.spikes_per_click{1, 11}.brut(i,1)];
% end
% 
%   
% 
% figure
% j = 1;
% for j = 1:25
%     disp(j)
%         hold on
%         plot(output.spikes_per_click{1, 11}.xaxis(j,:),output.spikes_per_click{1, 11}.brut1(j,:),'linewidth',2.0)
%         pause
% %     if brut(j) ==0;
% %     brut(j) =1;
% %     end
% end


% logbrut = log(brut);



% Analyse adaptation between acoustic events 
bin = [];
f = 10;
tets = 0;
for j = size(output.spikes_per_click{1, f}.brut1,1) :-1:1
    if output.spikes_per_click{1,f}.brut1(j,1)< 8 || output.spikes_per_click{1,f}.brut1(j,2)<1
        output.spikes_per_click{1,f}.brut1(j,:) = [];
        tets = tets +1;
    end
end

    




bin(1,:) = output.spikes_per_click{1, f}.brut1(:,1);
bin(2,:) = -output.spikes_per_click{1, f}.brut1(:,1)+output.spikes_per_click{1, f}.brut1(:,2);
bin(3,:) = -output.spikes_per_click{1, f}.brut1(:,1)+output.spikes_per_click{1, f}.brut1(:,3);

bin(4,:) = output.spikes_per_click{1, f}.brut1(:,2);
bin(5,:) = -output.spikes_per_click{1, f}.brut1(:,2)+output.spikes_per_click{1, f}.brut1(:,3);

bin(6,:) = -output.spikes_per_click{1, f}.brut1(:,2)+output.spikes_per_click{1, f}.brut1(:,5);
bin(7,:) = output.spikes_per_click{1, f}.brut1(:,5);
bin(8,:) = -output.spikes_per_click{1, f}.brut1(:,5)+output.spikes_per_click{1, f}.brut1(:,8);

for j = 1:size(output.spikes_per_click{1, f}.brut1,1)

    bin(2,j) = bin(2,j)/bin(1,j);
    bin(3,j) = bin(3,j)/bin(1,j);
    bin(5,j) = bin(5,j)/bin(4,j);
    bin(6,j) = bin(6,j)/bin(4,j);
    bin(8,j) = bin(8,j)/bin(7,j);
end



C1 = cov(bin(1,:),bin(2,:));
[U,S,V] = svd(C1);
V = V*sqrt(S)*3;
v1 = V(:,1);
v2 = V(:,2);
coM(1) = mean(bin(1,:));
coM(2) = mean(bin(2,:));

C2 = cov(bin(1,:),bin(3,:));
[U2, S2, V2] = svd(C2);
V2 = V2*sqrt(S2)*3;
v3 = V2(:,1);
v4 = V2(:,1);
coM(3) = mean(bin(3,:));

% C3 = cov(bin(4,:),bin(5,:));
% C4 = corr(bin(4,:).',bin(6,:).');
% C5 = corr(bin(7,:).',bin(8,:).');

% SEM = std(bin(2,:))/sqrt(length(bin(2,:)));
% ts = tinv([0.025  0.975],length(bin(2,:)));      % T-Score 95%




figure
subplot(2,2,[1 3])
scatter(bin(1,:),bin(2,:),'filled')
hold on
scatter(bin(4,:),bin(6,:),'filled')
scatter(bin(7,:),bin(8,:),'filled')

xlimit = get(gca,'xlim');


line([0 250], [0 0]);
line([0 250], [1 1]);
line([0 250], [-1 -1]);
axis([0,250,-5,5])

legend1 = sprintf('1st to 2nd pulse. corr = %.2f',corr(bin(1,:).',bin(2,:).'));
legend2 = sprintf('2nd to 5th pulse. corr = %.2f',corr(bin(4,:).',bin(6,:).'));
legend3 = sprintf('5th to 8th pulse. corr = %.2f',corr(bin(7,:).',bin(8,:).'));
legend({legend1, legend2, legend3});
ylabel('factor of increase in activity')
xlabel('firing rate above spont rate')
title('Evolution of firing rate in response to intial activity')
% scatter(bin(1,:),bin(7,:),'filled')
% quiver(coM(1),coM(2),v1(1),v1(2), 'b-')
% quiver(coM(1),coM(2),v2(1),v2(2), 'b-')

% scatter(bin(1,:),bin(3,:),'filled')
% % quiver(coM(1),coM(3),v3(1),v3(2), 'r-')
% % quiver(coM(1),coM(3),v4(1),v4(2), 'r-')
% 
% % hold on 



subplot(2,2,2)
edges = -5:0.20:5;
% [h1,edges] = histcounts(bin(2,:), edges);
histogram(bin(2,:), edges);
hold on 
% [h2,edges] = histcounts(bin(3,:), edges);
histogram(bin(3,:), edges);
legend1 = sprintf('1st to 2nd pulse');%. corr = %.2f',corr(bin1,bin2));
legend2 = sprintf('1st to 3rd pulse');%. corr = %.2f',corr(bin1,bin3));
% legend3 = sprintf('mu = %.3f', mean(h3));
legend({legend1, legend2}) %, legend3});

title('plasticity near onset')

subplot(2,2,4)
edges = -5:0.20:5;
% [h3,edges]= histcounts(bin(6,:), edges);
histogram(bin(6,:), edges);
hold on
% [h4,edges] = histcounts(bin(8,:), edges);
histogram(bin(8,:), edges);
legend1 = sprintf('2nd to 5th pulse');%. corr = %.2f',corr(bin1,bin2));
legend2 = sprintf('5th to 8thpulse');%. corr = %.2f',corr(bin1,bin3));
% legend3 = sprintf('mu = %.3f', mean(h3));
legend({legend1, legend2}) %, legend3});
title('plasticity near middle of stimuli')
xlabel('factor of increase in activity')


% stats

[p1,~,stats1] = signrank(bin(2,:))
[p2,~,stats2] = signrank(bin(3,:))

%% Adaptation in different model neurons.
clear all
% load('Md_SN_june.mat')
% load('Md_SN_SRAweak_june.mat')
load('Md_SN_SRAstrong_june.mat')
% load('Md_SN_facil_june')

f= 10;
bin(1,:) = UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,1);

bin(2,:) = -UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,1)+UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,2);
bin(3,:) = -UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,1)+UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,3);

bin(4,:) = UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,2);
bin(5,:) = -UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,2)+UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,3);

bin(6,:) = -UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,2)+UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,5);
bin(7,:) = UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,5);
bin(8,:) = -UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,5)+UnitInfo.Info.Output.spikes_per_click{1, f}.brut(:,8);


%For facilitation only, when we have 0 spikes at onset)
bin2 = round(bin);
%end
            


for j = 1:size(UnitInfo.Info.Output.spikes_per_click{1, f}.brut,1)

    bin(2,j) = bin(2,j)/bin(1,j);
    bin(3,j) = bin(3,j)/bin(1,j);
    bin(5,j) = bin(5,j)/bin(4,j);
    bin(6,j) = bin(6,j)/bin(4,j);
    bin(8,j) = bin(8,j)/bin(7,j);
end

% %for faciltation
% for i = 1:50
%     if bin2(1,i) ==0
%         if bin2(2,i)==48
%             bin(2,i) = 1;
%         elseif bin2(2,i)==-48
%             bin(2,i) = -1;
%         elseif bin2(2,i) == 95
%             bin(2,i) = 2;
%         elseif bin2(2,i) ==-95
%             bin(2,i) = -2;
%         end
%         
%         if bin2(3,i)==48
%             bin(3,i) = 1;
%         elseif bin2(3,i)==-48
%             bin(3,i) = -1;
%         elseif bin2(3,i) == 95
%             bin(3,i) = 2;
%         elseif bin2(3,i) ==-95
%             bin(3,i) = -2;
%         end
%     end
% end
% %end


figure
subplot(2,2,[1 3])
scatter(bin(1,:),bin(2,:),'filled')
hold on
scatter(bin(4,:),bin(6,:),'filled')
scatter(bin(7,:),bin(8,:),'filled')

xlimit = get(gca,'xlim');


line([0 250], [0 0]);
line([0 250], [1 1]);
line([0 250], [-1 -1]);
axis([0,250,-5,5])

legend1 = sprintf('1st to 2nd pulse. corr = %.2f',corr(bin(1,:).',bin(2,:).'));
legend2 = sprintf('2nd to 5th pulse. corr = %.2f',corr(bin(4,:).',bin(6,:).'));
legend3 = sprintf('5th to 8th pulse. corr = %.2f',corr(bin(7,:).',bin(8,:).'));
legend({legend1, legend2, legend3});
ylabel('factor of increase in activity')
xlabel('firing rate above spont rate')
title('Evolution of firing rate in response to intial activity')
% scatter(bin(1,:),bin(7,:),'filled')
% quiver(coM(1),coM(2),v1(1),v1(2), 'b-')
% quiver(coM(1),coM(2),v2(1),v2(2), 'b-')

% scatter(bin(1,:),bin(3,:),'filled')
% % quiver(coM(1),coM(3),v3(1),v3(2), 'r-')
% % quiver(coM(1),coM(3),v4(1),v4(2), 'r-')
% 
% % hold on 



subplot(2,2,2)
edges = -5:0.20:5;
% [h1,edges] = histcounts(bin(2,:), edges);
histogram(bin(2,:), edges);
hold on 
% [h2,edges] = histcounts(bin(3,:), edges);
histogram(bin(3,:), edges);
legend1 = sprintf('1st to 2nd pulse');%. corr = %.2f',corr(bin1,bin2));
legend2 = sprintf('1st to 3rd pulse');%. corr = %.2f',corr(bin1,bin3));
% legend3 = sprintf('mu = %.3f', mean(h3));
legend({legend1, legend2}) %, legend3});

title('plasticity near onset')

subplot(2,2,4)
edges = -5:0.20:5;
% [h3,edges]= histcounts(bin(6,:), edges);
histogram(bin(6,:), edges);
hold on
% [h4,edges] = histcounts(bin(8,:), edges);
histogram(bin(8,:), edges);
legend1 = sprintf('2nd to 5th pulse');%. corr = %.2f',corr(bin1,bin2));
legend2 = sprintf('5th to 8thpulse');%. corr = %.2f',corr(bin1,bin3));
% legend3 = sprintf('mu = %.3f', mean(h3));
legend({legend1, legend2}) %, legend3});
title('plasticity near middle of stimuli')
xlabel('factor of increase in activity')



Statss = zeros(3,4);
X1 = bin(2,:);
X1(isnan(X1)) = [];
Y1 = bin(3,:);
Y1(isnan(Y1)) = [];
Statss(1,1) = mean(X1); 
Statss(2,1) = median(X1);
Statss(3,1) = std(X1);
Statss(1,2) = mean(Y1);
Statss(2,2) = median(Y1);
Statss(3,2) = std(Y1);

[p,h,stats] = signrank(X1, Y1,'tail','right')


X = bin(6,:);
X(isnan(X)) = [];
X(isinf(X)) = [];
Y = bin(8,:);
Y(isnan(Y)) = [];
Y(isinf(Y)) = [];
Statss(1,3) = mean(X);
Statss(2,3) = median(X);
Statss(3,3) = std(X);
Statss(1,4) = mean(Y);
Statss(2,4) = median(Y);
Statss(3,4) = std(Y);
% Statss
% [h,p] = ttest(X)
% [h,p] = ttest(Y)

%% bar plot for model adaptation analysis

figure
subplot(2,1,1)
y = [-0.63 -0.95 -0.17 -0.05; -0.85 -0.95 -.25 -.13; -.95 -.98 -.14 -.2; -.475 -.751 -.12 -.20];
bar(-y)
xticklabels({'Normal','weak SFA','strong SFA' 'Facilitation' });

subplot(2,1,2)
y2 = [.42 -.043; -0.33 -.73; -.2 -.31; -.20 -.06];
bar(-y2)
xticklabels({'Normal','weak SFA','strong SFA' 'Facilitation' 'Real neurons'});


%% 01/25/2018 Spike latency calculation, see Bendor 2008
% This calculates Onset latency for each trial for each neuron.
% clear all
% load('SyncN_new.mat');
load('SyncP_new.mat');


nn = size(output.mean_neuron_spont{1},1);
time = size(output.mean_neuron_spont{1},2);
for f = 1:12
    spont{f} = zeros(nn,time/2);
    rate{f} = zeros(nn,time/2);
end

for f = 1:12
    for n = 1:nn
        t = 1;
        while t < time/2 + 1
            spont{f}(n,t) = (output.mean_neuron_spont{f}(n,2*t)...
                +(output.mean_neuron_spont{f}(n,2*t-1)))/2 ;
            rate{f}(n,t) = (output.mean_neuron_rates{f}(n,2*t)...
                +(output.mean_neuron_rates{f}(n,2*t-1)))/2 ;
            t = t+1;
        end
        spontstd{f}(n) = std(spont{f}(n,:),1);
    end
end


responsetime = zeros(nn,12);
for n = 1:nn
    for f = 1:12
        t = 1;
        while t < time/2-2
            if rate{f}(n,t) > spontstd{f}(n)*2 ...
                    && rate{f}(n,t+1) > spontstd{f}(n)*2 ...
                    && rate{f}(n,t+2) > spontstd{f}(n)*2
                responsetime(n,f) = t;
                t = time;
            else
                t = t+1;
            end
        end

    end
end


responsetime = responsetime*1e-3;
% responsetime(12,:) = []; %for SyncN neurons, neuron 12 doesnt react to
% stim

edges = 0:5*1e-3:max(responsetime);
figure
for n = 1:nn
    [h{n},edges] = histcounts(responsetime(n,:),edges);
    subplot(15,2,n)
    histogram(responsetime(n,:),edges)
    stdNeuron(n) = std(responsetime(n,:),1);
end
figure

edges = 0:2*1e-3:max(responsetime);
histogram(responsetime,edges)

ICI_list= [250 125 83.3333 62.5 50 41.6667 35.7143 31.25 27.7778 25 22.7273 20.8333]*1e-3;

for n = 1:nn
    for f = 1:12
        firsthalf{n,f} = output.ind_spike_time{n,f}(find(responsetime(n,f) <= output.ind_spike_time{n,f} & output.ind_spike_time{n,f} <= 0.275));
        if ~isempty(firsthalf{n,f})
        firsthalf{n,f} = 2*pi*mod(firsthalf{n,f},ICI_list(f))/ICI_list(f);
        end
        secondhalf{n,f} = output.ind_spike_time{n,f}(find(0.275 < output.ind_spike_time{n,f} & output.ind_spike_time{n,f} < 0.550));
        if ~isempty(secondhalf{n,f})
        secondhalf{n,f} = 2*pi*mod(secondhalf{n,f},ICI_list(f))/ICI_list(f);
        end
    end
end

% perihistogram
% edges = 0:0.001:0.25;
% edges = 0: pi/32: 2*pi;
% for n = 1:nn
%     figure('position',[800 100 800 900])
%     for f = 1:12
%         subplot(12,1,f)
%         
%         histogram(firsthalf{n,f},edges) %Blue
% %         pause
%         hold on
%         histogram(secondhalf{n,f},edges) %Orange
%         axis([0,2*pi,0,20])
%         if f <12
%             set(gca,'XTick',[]);
%         else
%             xticks([0 0.5*pi pi 1.5*pi 2*pi])
%             xticklabels({'0' '0.5\pi' '\pi' '1.5\pi' '2\pi'})
%         end
%         
%     end
% %     pause
% end
% 


figure 
boxplot(responsetime.')


% 29/03/2018    subtracting spike latency to spike time to pool all neurons
% Spikes per click regarding latency.
spiketime = {};
for f = 1:12
    spiketime{f} = [];
    for n = 1:nn
        spiketime{f} = [spiketime{f} output.ind_spike_time{n,f}-responsetime(n,f)];
    end
end



% attempt to compare first half/ second half.
for f = 1:12
    firsthalf{f} = spiketime{f}(find(responsetime(n,f) <= spiketime{f} & spiketime{f} <= 0.275));
    if ~isempty(firsthalf{f})
        firsthalf{f} = 2*pi*mod(firsthalf{f},ICI_list(f))/ICI_list(f);
    end
    secondhalf{f} = spiketime{f}(find(0.275 < spiketime{f} & spiketime{f} < 0.550));
    if ~isempty(secondhalf{f})
        secondhalf{f} = 2*pi*mod(secondhalf{f},ICI_list(f))/ICI_list(f);
    end
end

edges = 0: pi/32: 2*pi;
% edges = 0:0.002:0.25;

 figure('position',[800 100 800 900])
 for f = 1:12
     subplot(12,1,f)
     
     histogram(firsthalf{f},edges) %Blue
     %         pause
     hold on
     histogram(secondhalf{f},edges) %Orange
     axis([0,2*pi,0,inf])
     if f <12
         set(gca,'XTick',[]);
%      else
         xticks([0 0.5*pi pi 1.5*pi 2*pi])
         xticklabels({'0' '0.5\pi' '\pi' '1.5\pi' '2\pi'})
     end
     
 end

 
 %% spikes per click in model neurons. 

load('SyncPfMdl.mat')
ICI_list = [125 83.3333 62.5 50 41.6667 35.7143 31.25 27.7778 25 22.7273 20.8333];
Hz_list = [];
for i = 1:length(ICI_list)
    Hz_list = [Hz_list round(1000/ICI_list(i))];
end

figure
cmap = colormap(jet(length(ICI_list)+1));

for n =1:2:length(ICI_list)
    SEM = UnitInfo.Info.Output.spikes_per_click{n}.std/sqrt(20*600);
    ts = tinv([0.025  0.975],20*(600)-1);   
    error = ts(2)*SEM;
    errorbar(UnitInfo.Info.Output.spikes_per_click{n}.xaxis,UnitInfo.Info.Output.spikes_per_click{n}.mean,...
        error, 'linewidth', 1.7,'color',cmap(n+1,:),'DisplayName', ...
        [num2str(Hz_list(n)) 'Hz'])
    hold on
    %         norm_mean = [norm_mean UnitInfo.Info(p).Output.mean_discharge_rate.mean(n)/Hz_list(n)];
end
axis([500,1000,0,90]);
legend('show')


%%



