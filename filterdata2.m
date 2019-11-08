function filterdata2()

clear all

% will be needed when taking all data
%load('infoset.mat');
%animals = unique(cellfun(@char,{output.animal},'unif',0));

%% file info

Sync = 1; %or 0 for Nsync
Positive = -1; % or 1 for Negative
StimType = 12; % or 20
putfigure = 1;


ICI_list1 = [2 2.5 3 5 7.5 10 12.5 15 20 25 30 35 40 45 50 55 60 65 70 75]; %ms, ICI
ICI_list2 = [250 125 83.3333 62.5 50 41.6667 35.7143 31.25 27.7778 25 22.7273 20.8333];

if StimType == 12
    ICI_list = ICI_list2;
else % UnitInfo.Info(2,n) == 20
    ICI_list = ICI_list1;
end


% createFileInfo('m2p') %animal namecode
animal_list = {'m36n','m2p','m41o','m32q'};

load('Analysis.mat')
% Analysis.SyncPosi = {};
% Analysis.SyncNega = {};
% Analysis.NsyncPosi = {};
% Analysis.NsyncNega = {};
output.isi_comp = [];
for f = length(ICI_list)
    output.rates_stim{f} = [];
    output.rates_pre{f} = [];
    output.rates_post{f} = [];
    output.isi_total{f} = [];
    output.mean_neuron_rates{f} = [];
    output.mean_neuron_spont{f} = [];
    output.spike_time_IPI{f} = [];
    output.isi_onset_tot{f,1} = [];
    output.isi_onset_tot{f,2} = [];
    output.spikecount{f} = [];
    output.spikecount_neuron{f} = [];
end

output.ind_spike_time = {};
output.names = [];
nn = 1;
trial_count = 0;
freq_list=round(1000./ICI_list);


for ind =2:4% 2:4
    animal = animal_list{ind};
    load([animal '_List3']);
    
    %% spikes, VS and raster
    %     directory = ['U:\Neural and Behavioural Data' '\marmoset\' animal];
%                 directory = ['C:\Users\John\Documents\Marmoset\' animal];
    directory = ['C:\Users\John.Lee\OneDrive\Bendorlab\Marmoset\' animal];
    indxList = find([UnitInfo.Info.Stimuli_Nb] ==StimType &...
        [UnitInfo.Info.Positive] == Positive &...
        [UnitInfo.Info.Sync] == Sync &  ...
        [UnitInfo.Info.Pre_stim_Duration] == 500 &...
        [UnitInfo.Info.Post_stim_Duration] == 500 & [UnitInfo.Info.Significant_rate] ==1 );
    %     & [UnitInfo.Info.Stimuli_Nb] ==StimType &[UnitInfo.Info.Positive] == Positive &
    
    for n =  indxList
        disp(UnitInfo.List{1,n})
        output.names(nn,1) = ind;
        spiketable = load([directory filesep UnitInfo.List{1,n}]);
        output.names(nn,2) = str2num(UnitInfo.List{1,n}(length(animal)+1:end-4));
        output.names(nn,3) = UnitInfo.Info(n).Channel_Nb;
        
        PREstim_duration = UnitInfo.Info(n).Pre_stim_Duration/1000; %seconds
        stim_duration = 0.5;
        POSTstim_duration = UnitInfo.Info(n).Post_stim_Duration/1000;
        TrialLength = UnitInfo.Info(n).Pre_stim_Duration + UnitInfo.Info(n).Post_stim_Duration +500;
        
        raster.stim=[];  raster.rep=[];  raster.spikes=[];
        emptycount = 0;
        for f = 1:length(ICI_list)
            output.spike_time_IPI{f} = [];  % for ind neurons. for pop analysis, remove this line
            output.ind_spike_time{nn,f} = [];
            indstim = find(spiketable(:,1)==f);
            nbrep = spiketable(find(spiketable(:,1)==f),2);
            nreps = max(nbrep.');
            spikes_pooled = [];
            rate_total = [];
            isi_total = [];
            if ~isempty(nbrep)
                for r = unique(nbrep.')
                    output.isi_rep{nn}{f,r} = [];
                    output.isi_spikes{nn}{f,r} = [];
                    spikes1 = []; %channel 1
                    spikes2 = []; %channel 2
                    for i = indstim.'
                        if spiketable(i,2) == r && spiketable(i,3)== 1 && spiketable(i,4) > 0 % channel 1
                            spikes1 = [spikes1 (spiketable(i,4)*1e-6 - PREstim_duration)]; % spikes rounded up to 0.1ms cad 100microseconds. converted to seconds
                                                                                           % spike time minus Prestim duration. thus this is spike time after stim onset 
                        end
                        if spiketable(i,2) == r && spiketable(i,3)== 2 && spiketable(i,4) > 0 % channel 2
                            spikes2 = [spikes2 (spiketable(i,4)*1e-6 - PREstim_duration)];
                        end
                    end
                    if length(spikes1) > length(spikes2)
                        spikes =  spikes1;
                    else
                        spikes = spikes2;
                    end
                    if isempty(spikes(find(spikes>PREstim_duration & spikes< PREstim_duration + 0.5)))
                        emptycount = emptycount+1;
                    end
                    
                    spikes_pooled = [spikes_pooled spikes];
                    
                    raster.stim=[raster.stim f*ones(size(spikes))];
                    raster.rep=[raster.rep r*ones(size(spikes))];
                    raster.spikes=[raster.spikes spikes];
                    
                    %spikecount
                    output.spikecount{f} = [output.spikecount{f}; length(spikes)];
                    output.spikecount_neuron{f} = [output.spikecount_neuron{f}; nn];
                    
                    % rate
                    rate = zeros(1,TrialLength);
                    spikes4rate = spikes(find(spikes<PREstim_duration+0.5)) + PREstim_duration;
                    
                    
                    for st = spikes4rate
                        if ceil(st*1e3) <= length(rate)
                            rate(1,ceil(st*1e3)) = rate(1,ceil(st*1e3))+1;
                        end
                        
                    end
                    rate_total = [rate_total ; rate*1000];
                    
                    %ISI
                    spikes3 = spikes(find(spikes>PREstim_duration & spikes<PREstim_duration+0.5));
                    spikes4 = [0 spikes3(1:end-1)];
                    isi = spikes3-spikes4;
                    isi = isi(2:end);
                    output.isi_total{f} = [output.isi_total{f} isi];
                    output.isi_rep{nn}{f,r} = isi;
                    output.isi_spikes{nn}{f,r} = spikes3;
                    
                    %ISI end
                end
                
                %Vector Strength
                spikes_pooled_for_vector_strength=spikes_pooled(find(spikes_pooled>0.05 & spikes_pooled<=(stim_duration+0.05)));
                freq2 = 1000./ICI_list(f);
                if ~isempty(spikes_pooled_for_vector_strength)
                    total_spikes=length(spikes_pooled_for_vector_strength);
                    x=0;
                    y=0;
                    if total_spikes>0
                        x=sum(cos(2*pi*(spikes_pooled_for_vector_strength*freq2)));
                        y=sum(sin(2*pi*(spikes_pooled_for_vector_strength*freq2)));
                    end
                    if total_spikes==0
                        vector=0;
                    else
                        vector=sqrt(x^2+y^2)/total_spikes;
                    end
                    rayleigh=2*total_spikes*vector^2;
                else
                    vector=0;
                    rayleigh(f)=0;
                end
                
                if rayleigh<13.8
                    vector=0;
                end
                output.VS(nn,f) = vector;
                
                %%% spike time distribution
                %if studying actual spike time distrib
                output.spike_time_IPI{f} =  [output.spike_time_IPI{f} mod(spikes_pooled_for_vector_strength,ICI_list(f)*1e-3)];
                
                %                  mod(spikes_pooled_for_vector_strength,ICI_list(f)*1e-3)];
                %replace if studying VS %
                %                 output.spike_time_IPI{f} =  [output.spike_time_IPI{f} 2*pi*[mod(spikes_pooled_for_vector_strength,ICI_list(f)*1e-3)]*freq2];
                %
                
                
                %%% neuron/neuron spiketime
                output.ind_spike_time{nn,f} = [output.ind_spike_time{nn,f} spikes_pooled_for_vector_strength]; 
                
                
                
                %VS comp between first and second half.
                for i = 1:2
                    spikes_for_VS = spikes_pooled(find(spikes_pooled>0.05+0.25*(i-1) & spikes_pooled<=0.05+0.25*(i)));
                    if ~isempty(spikes_for_VS)
                        total_spikes = length(spikes_for_VS);
                        x = 0;
                        y = 0;
                        if total_spikes>0
                            x=sum(cos(2*pi*(spikes_for_VS*freq2)));
                            y=sum(sin(2*pi*(spikes_for_VS*freq2)));
                            vector=sqrt(x^2+y^2)/total_spikes;
                        else
                            vector = 0;
                        end
                        rayleigh=2*total_spikes*vector^2;
                    else
                        vector=0;
                        rayleigh=0;
                    end
                    output.VStime{f}(nn,i) = vector;
                    output.Raleightime{f}(nn,i) = rayleigh;
                end
                %                 %VS through time
                %                 window = 0.2; %200ms time window.
                %                 timebin = 0.05 : 0.01: 0.05+stim_duration-window;
                %                 for i = 1:length(timebin)
                %                     spikes_for_VS = spikes_pooled(find(spikes_pooled>timebin(i) & spikes_pooled<=timebin(i)+window));
                %                     if ~isempty(spikes_for_VS)
                %                         total_spikes = length(spikes_for_VS);
                %                         x = 0;
                %                         y = 0;
                %                         if total_spikes>0
                %                             x=sum(cos(2*pi*(spikes_for_VS*freq2)));
                %                             y=sum(sin(2*pi*(spikes_for_VS*freq2)));
                %                             vector=sqrt(x^2+y^2)/total_spikes;
                %                         else
                %                             vector = 0;
                %                         end
                %                         rayleigh=2*total_spikes*vector^2;
                %                     else
                %                         vector=0;
                %                         rayleigh=0;
                %                     end
                %                     output.VStime{f}(nn,i) = vector;
                %                     output.Raleightime{f}(nn,i) = rayleigh;
                %                 end
                %
                %                 timebin = 0.05:ICI_list(f)*1e-3:0.05+stim_duration
                %                 for i = 1:length(timebin)-1
                %                     spikes_for_VS = spikes_pooled(find(spikes_pooled>timebin(i) & spikes_pooled<=timebin(i+1)));
                %                     if ~isempty(spikes_for_VS)
                %                         total_spikes = length(spikes_for_VS);
                %                         x = 0;
                %                         y = 0;
                %                         if total_spikes>0
                %                             x=sum(cos(2*pi*(spikes_for_VS*freq2)));
                %                             y=sum(sin(2*pi*(spikes_for_VS*freq2)));
                %                             vector=sqrt(x^2+y^2)/total_spikes;
                %                         else
                %                             vector = 0;
                %                         end
                %                         rayleigh=2*total_spikes*vector^2;
                %                     else
                %                         vector=0;
                %                         rayleigh=0;
                %                     end
                %                     output.VStime{f}(nn,i) = vector;
                %                     output.Raleightime{f}(nn,i) = rayleigh;
                %                 end
                %
                
                
                %  average rate
                
                PRE = PREstim_duration*1000;
                POST = POSTstim_duration*1000;
                STIM = stim_duration*1000;
                total_time = PRE+POST+STIM;
                %         rate_av = mean(rate_total,1);
                %         spont_rate = mean2(rate_total(find(indvec == neuronNB),PRE-100:PRE));
                %         discharge_rate{neuronNB}.mean = mean2(rate_total(find(indvec == neuronNB),PRE+1:PRE+STIM+100))-spont_rate;
                %         discharge_rate{neuronNB}.std = std2(rate_total(find(indvec == neuronNB),PRE+1:PRE+STIM+100))/sqrt(nb_rep*(STIM+100));
                %         output.DRmean{neuronNB} = [output.DRmean{neuronNB} discharge_rate{neuronNB}.mean];
                %         output.DRstd{neuronNB} = [output.DRstd{neuronNB} discharge_rate{neuronNB}.std];
                
            end
            %             spikes_pooled_for_VS=spikes_pooled(find(spikes_pooled>0.05 & spikes_pooled<=(stim_duration+0.05)));
            
            %Interspike interval
            
            %             var_isi = [var_isi std(isi)];
            output.rates_pre{f} = [output.rates_pre{f}; ...
                rate_total(:,1:PRE)];
            output.rates_stim{f} = [output.rates_stim{f}; ...
                rate_total(:,PRE+1:PRE+STIM+100)];
            output.mean_neuron_rates{f} = [output.mean_neuron_rates{f}; ...
                mean(rate_total(:,PRE+1:PRE+STIM+100),1)];
            output.mean_neuron_spont{f} = [output.mean_neuron_spont{f}; ...
                mean(rate_total(:,1:PRE),1)];
            output.rates_post{f} = [output.rates_post{f}; ...
                rate_total(:,PRE+STIM+101:end)];
            
            
        end
        
        trial_count = trial_count + r; %counting the number of trials, cumulative.
        
        %%% check for inconsistencies in repetition number
        for f = 2:12
            if size(output.rates_pre{f},1)> size(output.rates_pre{f-1},1)
                trial_count = trial_count-1;
                output.rates_pre{f}(end,:) = [];
                output.rates_stim{f}(end,:) = [];
                output.rates_post{f}(end,:) = [];
                output.spikecount{f}(end) = [];
                output.spikecount_neuron{f}(end) = [];
                %                 output.mean_neuron_rates{f}(end,:) = [];
            elseif size(output.rates_pre{f},1)< size(output.rates_pre{f-1},1)
                trial_count = trial_count-1;
                for ff = 1:f-1
                    output.rates_pre{ff}(end,:) = [];
                    output.rates_stim{ff}(end,:) = [];
                    output.rates_post{ff}(end,:) = [];
                    output.spikecount{ff}(end) = [];
                    output.spikecount_neuron{ff}(end) = [];
                end
            end
        end
        %%% calculate the time of peak in the histogram (latency of
        %%% response for each neuron)
        med = [];
        for f = 1:12
            %             if (mean2(output.rates_stim{f}((nn-1)*10+1:nn*10,:))-mean2(output.rates_pre{f}((nn-1)*10+1:nn*10,:)))>0
            if (mean2(output.rates_stim{f}(trial_count-r+1:trial_count,:))-mean2(output.rates_pre{f}(trial_count-r+1:trial_count,:)))>0
                med = [med median(output.spike_time_IPI{f})];
            else
                med = [med -1];
            end
        end
        %         med = median(output.spike_time_IPI{2});
        mednorm = med(find(med ~= -1));
        medtime = round(median(med(1:5))*1e3);
        %         disp(median(med))
        
        
        
        %%% Calculate 
        
        %% calculate spikes per click
        if Positive ~=0
            if Sync ==1
                for f = 1:length(ICI_list)
                    p = floor(500/ICI_list(f));
                    
                    for q = 1:p
                        clicktime = round((q-1)*ICI_list(f))+medtime;
                        output.spikes_per_click{f}.brut1(nn,q) = mean(output.mean_neuron_rates{f}(nn,clicktime-15:clicktime+15))-mean2(output.rates_pre{f}(trial_count-r+1:trial_count,:));
                        %                 output.spikes_per_click{f}.brut2(nn,q) = mean(output.mean_neuron_rates{f}(nn,clicktime-15:clicktime+15));
                        %                 output.spikes_per_click{f}.brut2(nn,q) = ...
                        %                     (output.spikes_per_click{f}.brut1(nn,q)-mean2(output.rates_pre{f}((nn-1)*10+1:nn*10,:)))...
                        %                     /(mean2(output.rates_stim{f}((nn-1)*10+1:nn*10,:))-mean2(output.rates_pre{f}((nn-1)*10+1:nn*10,:)));
                        output.spikes_per_click{f}.xaxis(nn,q) = clicktime+500;
                    end
                    output.spikes_per_click{f}.spont(nn) = mean2(output.rates_pre{f}(trial_count-r+1:trial_count,:));
                    % Individual neuron analysis:
                    output.indspont{nn,f} = mean(rate_total(:,1:PRE),1);
                    output.indrate{nn,f} = mean(rate_total(:,PRE+1:PRE+STIM+100),1);
                end
            end
        end
        %            if nn == 23
        %% plot histogram and fit
        %         edges = 0: pi/64: 2*pi;
        % %                 edges = 0:0.001:0.125;
        %         figure('position',[800 100 800 900])                    %         %
        %         for i = 1:12
        %             subplot(12,1,i)
        %             histogram(output.spike_time_IPI{i},edges)
        %             [h1,edges] = histcounts(output.spike_time_IPI{3}, edges);
        %             axis([0,2*pi,0,20])
        %             %             hold on
        %             %             pd = fitdist(output.spike_time_IPI{2}.','Gamma');
        %             %             y = pdf(pd,edges);
        %             %             y = y/sum(y)*length(output.spike_time_IPI{2});
        %             %             % figure
        %             %             % xx = 0:0.002:0.125;
        %             %             plot(edges,y,'linewidth',2,'color','r')
        %             if i <12
        %                 set(gca,'XTick',[]);
        %             else
        %                 xticks([0 0.5*pi pi 1.5*pi 2*pi])
        %                 xticklabels({'0' '0.5\pi' '\pi' '1.5\pi' '2\pi'})
        %             end
        %         end
        %         test = 1;
        %         pause
        
        
        %% end histogram
        
        % histogram of isi
        
        
        % Onset response isi analysis.
        
        for f = 1:12
            output.isi_onset{nn}{f,1} = [];
            output.isi_onset{nn}{f,2} = [];
            if med(f) ~=-1
                for r = 1:size(output.isi_spikes{nn},2)
                    
                    indd1 = find(output.isi_spikes{nn}{f,r}>0.5-0.02+med(f) & output.isi_spikes{nn}{f,r}<0.5 + med(f));
                    indd2 = find(output.isi_spikes{nn}{f,r}>0.5+med(f) & output.isi_spikes{nn}{f,r}<0.5+0.02 + med(f));
                    indd3 = find(output.isi_spikes{nn}{f,r}>0.5-0.02+medtime*1e-3 & output.isi_spikes{nn}{f,r}<0.5+0.02 + medtime*1e-3);
                    
                    
                    
                    if length(indd1) > 1
                        indd1 = indd1-1;
                        indd1(1) = [];
                        output.isi_onset{nn}{f,1} = [output.isi_onset{nn}{f,1} ...
                            output.isi_rep{nn}{f,r}(indd1)];
                    end
                    if length(indd2) > 1
                        indd2 = indd2-1;
                        indd2(1) = [];
                        output.isi_onset{nn}{f,2} = [output.isi_onset{nn}{f,2} ...
                            output.isi_rep{nn}{f,r}(indd2)];
                    end
                    
                    if length(indd3)> 2 && f<10
                        indd3 = indd3-1;
                        indd3(1) = [];
                        if length(indd3)<3
                            output.isi_comp = [output.isi_comp;[output.isi_rep{nn}{f,r}(indd3(1:2)) NaN]];
                        else
                            output.isi_comp = [output.isi_comp;output.isi_rep{nn}{f,r}(indd3(1:3))];
                        end
                    end
                    
                end
                
                output.isi_onset_tot{f,1} = [output.isi_onset_tot{f,1} output.isi_onset{nn}{f,1}];
                output.isi_onset_tot{f,2} = [output.isi_onset_tot{f,2} output.isi_onset{nn}{f,2}];
            end
        end
        
        
        if median(output.spike_time_IPI{2})<0.07
            nn = nn+1;
        else %%% correcting output.mean_neuron data for ind neuron analysis
            for f = 1:12
                output.mean_neuron_rates{f}(nn,:) = [];
                output.mean_neuron_spont{f}(nn,:) = [];
            end
        end
    
    
    
    
        
    end
end
% pause
disp(nn)
Hz_list = [];
for i = 1:length(ICI_list)
    Hz_list = [Hz_list round(1000/ICI_list(i))];
end

% Rasterplot
% x = freq_list;
% xlabel('time (s)')
% % ylabel('IPI (ms)')
% ylabel('Repetition rate (Hz)')
% % area([0 500 stimulus_duration 0],[length(x)*nreps+1 length(x)*nreps+1 0 0],'LineStyle','none','FaceColor',[.85 .85 1]);
% hold on
% plot(raster.spikes,nreps*(raster.stim-1)+raster.rep,'k.','MarkerSize',9);
% axis([0 stimulus_duration+POSTstimulus_duration 0 length(x)*nreps+1])

% Rasterplot end

%     test =1;
%     all_rate_total = [];
output.Fanofactor = [];
% vector = zeros(1,length(ICI_list));
% edges = 0: pi/32: 2*pi;
% % edges = 0:0.001:0.125;

% % % figure for VS across time

% cmap = colormap(jet(length(ICI_list)));




%
% figure
%
%
% VS_x = 0.1:0.01:0.4;
% test = [];
% test2 = [];
% for i = 1:31
%     test = [test; output.VStime{9}(:,i)];
%     test2 = [test2;ones(nn-1,1)*i];
% end
%
%
% for f = 2:2:12
%     SEM = std(output.VStime{f},1)/sqrt(nn);
%     ts = tinv([0.025  0.975],sqrt(nn));  % T-Score
%     %     CI = mean2(output.rates_stim{f}) + ts(2)*SEM;                      % Confidence Intervals
%     error = ts(2)*SEM;
%
%     errorbar(VS_x,mean(output.VStime{f},1),error,'color',cmap(f,:),'LineWidth',1.7,'DisplayName', ...
%         [num2str(ceil(1000/ICI_list(f))) 'Hz'])
%     hold on
% end
%










%scatter plot, isi(1,2) isi(1,3)
%
figure
scatter(output.isi_comp(:,1),output.isi_comp(:,2),25,'r','filled')
line([0 0.04],[0 0.04])
figure
scatter(output.isi_comp(:,1),output.isi_comp(:,3),25,'r','filled')
line([0 0.04],[0 0.04])

for i = 1:length(output.isi_comp)
    output.isi_proj1(i,1) = output.isi_comp(i,1);
    if ~isnan(output.isi_comp(i,3))
        output.isi_proj2(i,1) = (output.isi_comp(i,1)+output.isi_comp(i,3))/sqrt(2);
    end
end

R = [cos(pi/4) -sin(pi/4); sin(pi/4) cos(pi/4)];
output.isi_proj1 = R*output.isi_comp(:,1:2).';
output.isi_proj2 = R*output.isi_comp(:,1:2:3).';

figure
edges = -0.025:0.001:0.025;
subplot(2,1,1)
histogram(output.isi_proj1(1,:),edges)
subplot(2,1,2)
histogram(output.isi_proj2(1,:),edges)

% histogram of ISI

figure
for f = 1:12
    subplot(12,1,f)
    if f <12
        set(gca,'XTick',[]);
    end
    edges = 0:0.001:0.05;
    histogram(output.isi_onset_tot{f,1},edges)
    hold on
    histogram(output.isi_onset_tot{f,2},edges)
    ylabel({Hz_list(f)})
end


figure

for i = 1:12
    subplot(12,1,i)
    
    histogram(output.spike_time_IPI{i},edges,'Normalization','pdf')
    hold on
    if i <12
        set(gca,'XTick',[]);
    else
        %         xticks([0 0.5*pi pi 1.5*pi 2*pi])
        %         xticklabels({'0' '0.5\pi' '\pi' '1.5\pi' '2\pi'})
    end
    
    ylabel({Hz_list(i)})
    % pause
end




% [l,lci] = poissfit(output.spike_time_IPI{2});
%
%
% x = 0:1:125;
% l = l*1000;
% y = poisspdf(x,l);
% y = y*length(output.spike_time_IPI{2});

%
% pd = fitdist(output.spike_time_IPI{2}.','Lognormal');
% y = pdf(pd,edges);
% % figure
% % xx = 0:0.002:0.125;
% plot(edges,y,'linewidth',2,'color','r')

% disp(x);

%

output.meanVS = mean(output.VS,1);
tsVS = (tinv([0.025  0.975],nn));
output.errorVS = std(output.VS,1)/sqrt(nn)*tsVS(2);


output.meanDR = [];
output.errorDR = [];
output.totalDR = [];
output.totalDR_yaxis = [];
output.var_isi = [];
% output.meanspt = [];
for f = 1:length(ICI_list)
    output.mean_rate_stim{f} = mean(output.rates_stim{f},1);
    output.std_rate_stim{f} = std(output.rates_stim{f},0,1);
    output.mean_rate_pre{f} = mean(output.rates_pre{f},1);
    output.std_rate_pre{f} = std(output.rates_pre{f},0,1);
    output.mean_rate_post{f} = mean(output.rates_post{f},1);
    SpikeCount = sum(output.rates_stim{f},2)/1000.;
    output.Fanofactor = [output.Fanofactor std(SpikeCount)^2/mean(SpikeCount)];
    output.var_isi = [output.var_isi std(output.isi_total{f})];
    
    if Sync ==1 && Positive ~=0
        %         spikes per click
        output.spikes_per_click{f}.mean = mean(output.spikes_per_click{f}.brut1,1);
        output.spikes_per_click{f}.sem = std(output.spikes_per_click{f}.brut1,1)/sqrt(size(output.spikes_per_click{f}.brut1,1)*size(output.spikes_per_click{f}.brut1,2));
        output.spikes_per_click{f}.xaxismean = mean(output.spikes_per_click{f}.xaxis,1);
        ts = tinv([0.025  0.975],size(output.spikes_per_click{f}.brut1,1)*size(output.spikes_per_click{f}.brut1,2)-1);
        output.spikes_per_click{f}.error = ts(2)*output.spikes_per_click{f}.sem;
        
    end
    
    
    
    %Gaussian smoothing
    xs = 1:1500;
    h = 10;
    for i = 1:1500
        ys(i) = gaussian_kern_reg(xs(i),xs,[output.mean_rate_pre{f} output.mean_rate_stim{f} output.mean_rate_post{f}],h);
    end
    smooth_rate{f} = ys;
    output.meanDR = [ output.meanDR mean2(output.rates_stim{f})-mean2(output.rates_pre{f})];
    %     output.meanspt = [output.meanspt mean2(output.rates_pre{f})];
    SEM = std2(output.rates_stim{f})/sqrt(size(output.rates_stim{1},1)*size(output.rates_stim{1},2));
    ts = tinv([0.025  0.975],size(output.rates_stim{1},1)*size(output.rates_stim{1},2)-1);      % T-Score
    %     CI = mean2(output.rates_stim{f}) + ts(2)*SEM;                      % Confidence Intervals
    output.errorDR = [output.errorDR  ts(2)*SEM];
    %     for g = 1:46
    %         output.totalDR = [output.totalDR; mean2(output.rates_stim{f}((g-1)*10+1:g*10,:))];
    %     end
    %     output.totalDR_yaxis = [output.totalDR_yaxis; ones(46,1)*f];
end



% output.onset = mean(output.mean_rate_stim{1,1}(1,20:30),2);


[RHO,PVAL] = corr(ICI_list(2:end).',output.meanDR(2:end).','Type','Spearman')

if Positive == -1
    save('SyncP_new.mat', 'output')
elseif Positive ==1
    save('SyncN_new.mat', 'output')
end

if putfigure == 1
    
    cmapp = [[0.26 0.5 0.9]; [0.9 0.3 0.26]; [0 0 0];  [0.1 0.7 0.1]; [0.9 0.6 0.1] ]; %Green Yellow Black Blue Red : Nsync+ Nsync- Model Sync+ Sync-
    %
    if Positive ~=0
        color = cmapp(3*Sync + 1.5 + Positive/2,:);
    else
        color = cmapp(3,:);
    end
    cmap = colormap(jet(length(ICI_list)));
    
    
    % ff = 12;
    % norm48Hz = output.spikes_per_click{ff}.mean/output.spikes_per_click{ff}.mean(1);
    % normSEM = output.spikes_per_click{ff}.sem/sqrt(output.spikes_per_click{ff}.mean(1));
    % plot(output.spikes_per_click{ff}.xaxis,norm48Hz,'color',cmapp(4,:),'LineWidth',1.7)
    % errorbar(output.spikes_per_click{ff}.xaxis,norm48Hz,normSEM,'color',cmapp(5,:),'LineWidth',1.7)
    % axis([500,1100,0.2,1.2])
    
    % set(gca, 'FontSize', 16)
    hold on
    if Sync ==1 && Positive ~=0
        figure
        for f = 2:2: length(ICI_list)
            errorbar(output.spikes_per_click{f}.xaxismean,output.spikes_per_click{f}.mean,output.spikes_per_click{f}.error, 'color',cmap(f,:),'LineWidth',1.7,'DisplayName', ...
                [num2str(ceil(1000/ICI_list(f))) 'Hz'])
            hold on
        end
        legend('show')
        set(gca, 'FontSize', 16)
        test = 1;
        xlabel('Time(ms)')
        ylabel('Average number of spikes per click (Spikes/sec)')
    end
    
    titre = ['RHO= ' num2str(RHO) ', p value = ' num2str(PVAL)];
    figure
    subplot(2,3,2)
%     shadedErrorBar(Hz_list,output.meanVS,output.errorVS,{'Color',color})
    errorbar(Hz_list,output.meanVS,output.errorVS,'color',color,'LineWidth',1.7)
    xlabel('Repetition rate (Hz)')
    ylabel('Firing rate (Spikes/sec)')
    title('Vector Strength')
    axis([0,50,-0,1])
    set(gca, 'FontSize', 16)
    subplot(2,3,3)
    
    % shadedErrorBar(Hz_list,output.meanDR,output.errorDR,{'Color',color})
    errorbar(Hz_list,output.meanDR,output.errorDR,'color',color,'LineWidth',1.7)
    ylabel('Firing rate (Spikes/sec)')
    xlabel('Repetition rate (Hz)')
    title('Discharge Rate')
    axis([0,50,0,40])
    set(gca, 'FontSize', 16)
    
    cmap = colormap(jet(length(ICI_list)));
    subplot(2,3,[1 4])
    
    for f = 2:2: length(ICI_list)
        plot(smooth_rate{f},'color',cmap(f,:),'LineWidth',1.7,'DisplayName', ...
            [num2str(ceil(1000/ICI_list(f))) 'Hz'] );
        %     lgd{f} = [num2str(ceil(1000/ICI_list(f))) 'Hz'];
        hold on
    end
    legend('show')
    axis([300,1200,0,80])
    title(titre)
    set(gca, 'FontSize', 16)
    %
    %
    %
    subplot(2,3,5)
    plot(Hz_list,output.Fanofactor,'Color',color)
    title('FanoFactor')
    xlabel('Repetition rate (Hz)')
    subplot(2,3,6)
    plot(Hz_list,output.var_isi,'Color',color)
    title('Variance of ISI')
    xlabel('Repetition rate (Hz)')
end

