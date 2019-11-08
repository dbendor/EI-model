function model_main()

clear all

% Rewrite of the fuction Conductance_LIF to clean the code and include
% facilitation as well as depression

%% Parameters
global ICI_list PureTone tau_pE tau_pI kernel_time_constant

ICI_list = [125 83.3333 62.5 50 41.6667 35.7143 31.25 27.7778 25 22.7273 20.8333]; %Stimulus list

PureTone = 0; % For modeling puretone responses, set to 1, otherwise, set to 0
if PureTone ==1
    ICI_list = 2;
end

%LIF model parameters
IE_delay = 5; %ms
E_strength = 4.5; % nS
I_strength = 8.5; %  nS
kernel_time_constant = 0.005;
noise = 4;

% adaptation model time constants
tau_pE = 0.15;
tau_pI = 0.10;
adaptation.E = 1; %adaptation for {E,I} 0 is facilitation, 1 is depression.
adaptation.I = 1;

% code parameters
figureon = 1; % to plot figures, set to 1. Otherwise, set to 0
export_model = 1; % to save model output, set to 1. Otherwise, set to 0
n = 0;
%% Run model
%for Sync+ with depression f_E = 0.1, f_I = 0.4
%for Sync- with depression f_E = 0.4, f_I = 0.1




for f_E = 0.4
    for f_I = 0.1
%         for amp = 0.6:0.2:2.0
%             E_strength = E_strength*amp;
%             I_strength = I_strength*amp; %nS

%             for tau_pI = 0.06:0.04:0.20
%                 for  tau_pE = 0.06:0.04:0.20
%                     for IE_ratio = 0.5:0.1:2
%                         I_strength = E_strength*IE_ratio;
                        for jitter = 1 %0:10
                            for trial_nb = 1
                                output = {};
                                output.raster.stim = [];
                                output.raster.rep = [];
                                output.raster.spikes = [];
                                output.spiketime = {};
                                for f = 1:length(ICI_list)
                                    out = run_model(IE_delay,E_strength,I_strength,f,f_E,f_I,adaptation,PureTone, noise, jitter);
                                    output.raster.stim = [output.raster.stim out.raster.stim];
                                    output.raster.rep = [output.raster.rep out.raster.rep];
                                    output.raster.spikes = [output.raster.spikes out.raster.spikes];
                                    output.spiketime{f} = out.raster.spikes;
                                    output.VS(f) = out.VS;
                                    %             output.VS_pop(f,:) = out.vector;
                                    output.mean_discharge_rate.mean(f) = out.discharge_rate.mean;
                                    output.mean_discharge_rate.error(f)  = out.discharge_rate.error;
                                    output.rate{f} = out.rate;
                                    output.rate_brut{f} = out.rate_brut;
                                    output.rate_total{f} = out.rate_total;
                                    output.rates_stim{f} = out.rates_stim;
                                    %             output.adaptation.E{f} = out.E_strength;
                                    %             output.adaptation.I{f} = out.I_strength;
                                    %             output.adaptation.E_I{f} = out.I_strength-out.E_strength;
                                    %             output.adaptation.E_I{f} = output.adaptation.E_I{f}(1:end-1);
                                    output.spikes_per_click{f} = out.spikes_per_click;
                                    output.Fanofactor(f) = out.Fanofactor;
                                    output.var_ISI(f) = out.var_ISI;
                                    %                     for t = 1:length(output.adaptation.E_I{f})
                                    %                         output.product{f}(t) = output.adaptation.E_I{f}(t)*output.time_period{f}.mean(t);
                                    %                     end
                                end
                                n= n+1;
                                disp(n)
                                UnitInfo.Info(n).Output = output;
                                UnitInfo.List(n,1) = f_E;
                                UnitInfo.List(n,2) = f_I;
                                UnitInfo.List(n,3) = tau_pE;
                                UnitInfo.List(n,4) = tau_pI;
                                UnitInfo.List(n,5) = E_strength;
                                UnitInfo.List(n,6) = I_strength;
                                
                                disp(UnitInfo.List)
                                
                                UnitInfo.Info(n).Output = output;
                                
                                new_all_mean_rate_stim = output.mean_discharge_rate.mean; %(1:end-1);
                                [RHO,PVAL] = corr(ICI_list.',new_all_mean_rate_stim.','Type','Spearman')
                                
                                UnitInfo.Info(n).Rho = RHO;
                                UnitInfo.Info(n).Pval = PVAL;
                                %     subplot(5,5,n)
                                %     plot(UnitInfo.Info(n).Output.rate{1}, 'linewidth', 1.7,'DisplayName',[num2str(n)])
                                %     xlabel([num2str(UnitInfo.List(n,1)) num2str(UnitInfo.List(n,2))]);
                                %         hold on
                                %         norm_mean = [norm_mean UnitInfo.Info(p).Output.mean_discharge_rate.mean(n)/Hz_list(n)];
                                %     pause
                                %     hold on
                            end
                        end
%                     end
%                 end
%             end
%         end
    end
end

if export_model == 1
    if f_E >0.2
        save('modelSN.mat','UnitInfo')
    else
        save('modelSP.mat','UnitInfo')
    end
end

% save('modeldataampSP.mat','UnitInfo')
% save('Md_SRAonly_Sep.mat','UnitInfo')
% save('PT_Md_spktime.mat','UnitInfo')

%% plot figures
if figureon == 1
    Hz_list = [];
    for i = 1:length(ICI_list)
        Hz_list = [Hz_list round(1000/ICI_list(i))];
    end
    
    
    % figure
    cmapp = [[0.1 0.7 0.1]; [0.9 0.6 0.1]; [0 0 0]   ];%    [0.26 0.5 0.9]   ]; %; [0.9 0.3 0.26]; ];
    
    cmap = colormap(jet(length(ICI_list)+1));
    
    if  PureTone == 1
        
        for n = 1
            plot(UnitInfo.Info(n).Output.rate{1}, 'linewidth', 1.7,'DisplayName',[num2str(n)])
            %         hold on
            %         norm_mean = [norm_mean UnitInfo.Info(p).Output.mean_discharge_rate.mean(n)/Hz_list(n)];
            %         pause
            hold on
        end
    else
        p = n;
        norm_mean = [];
        subplot(2,2,[1 3])
        
        hold off
        for n =1:2:length(UnitInfo.Info(p).Output.rate)
            plot(UnitInfo.Info(p).Output.rate{n}, 'linewidth', 1.7,'color',cmap(n+1,:),'DisplayName', ...
                [num2str(Hz_list(n)) 'Hz'])
            hold on
            %         norm_mean = [norm_mean UnitInfo.Info(p).Output.mean_discharge_rate.mean(n)/Hz_list(n)];
        end
        hold off
        axis([300,1200,0,120]);
        legend('show')
        title(['// f_DE = ' num2str(UnitInfo.List(p,1))...
            '// f_DI = ' num2str(UnitInfo.List(p,2)) ...
            '// tau_pE = ' num2str(UnitInfo.List(p,3)) ...
            '// tau_pI = ' num2str(UnitInfo.List(p,4))])
        set(gca, 'FontSize', 16)
        hold on
        subplot(2,2,2)
%             shadedErrorBar(Hz_list,UnitInfo.Info(p).Output.mean_discharge_rate.mean,UnitInfo.Info(p).Output.mean_discharge_rate.error,{'--','Color',cmapp(2,:)})
        errorbar(Hz_list,UnitInfo.Info(p).Output.mean_discharge_rate.mean,UnitInfo.Info(p).Output.mean_discharge_rate.error)
        
        
        hold on
        axis([0,50,-5,50]);
        subplot(2,2,4)
        
        plot(Hz_list,UnitInfo.Info(p).Output.VS,'linewidth',2.0);
        set(gca, 'FontSize', 16)
        axis([0,50,0.5,1]);
        hold on
        %                 pause
        pause(0.1)
    end
end

function out = run_model(IE_delay,E_strength,I_strength,f,f_E,f_I,adaptation,PureTone,noise_magnitude,jitt)

global ICI_list kernel_time_constant tau_pE tau_pI
ICI = ICI_list(f);
kernel_time_constant=.005;  %time constant of 5 ms
step=.0001; %.1 ms duration  (temporal increment for running simulation)
stimulus_duration=0.5;  %half second
PREstimulus_duration=0.5;  %half second
POSTstimulus_duration=0.5;  %half second (0.1 second is included in stimulus)

if PureTone == 1
    stimulus_duration=0.2;
end
latency_time=0.01;
latency=length(0:step:latency_time); %10 ms latency (auditory nerve to auditory cortex)
% stepfn = zeros(1,15000);
% stepfn(1,1:5000) = 0.01;

spikes_pooled=[];
freq = 1000./ICI;

%Alpha kernel
t=0:step:(kernel_time_constant*10);
kernel=t.*exp(-t/kernel_time_constant);
kernel=1e-9*kernel/max(kernel); %amplitude of 1 nS


input=zeros(size(0:step:(POSTstimulus_duration+stimulus_duration)));
stimulus_input_length=length(0:step:(stimulus_duration));
ipi=round(1/(freq*step)); %ipi=interpulse interval
freq2=1/(step*ipi);


%% Modeling Conductance and adaptation.

nb_rep = 30;
E_str(1) = E_strength;
I_str(1) = I_strength;
E_strength_mean = [];
I_strength_mean = [];
rate_total = [];
Ge_total = [];
Gi_total = [];
Net_excit_total = [];
raster.stim=[];  raster.rep=[];  raster.spikes=[];
PRE = PREstimulus_duration*1000;
POST = POSTstimulus_duration*1000;
STIM = stimulus_duration*1000;
total_time = PRE+POST+STIM;

% noise_magnitude=4*1e-8; %default noise level in conductance
noise_magnitude=noise_magnitude*1e-8;
%adaptation parameters

if adaptation.E ==1
    f_DE = 1-f_E;
    P_0E = 1;
else
    P_0E = 0.5;
end
if adaptation.I ==1
    f_DI = 1-f_I;
    P_0I = 1;
else
    P_0I = 0.5;
end


for r = 1:nb_rep
    disp({f,r})
    E_input=input;
    I_input=input;
    %     delay=round(abs(IE_delay)/(1000*step));
    for j=1:15  %15 jitter excitatory and inhibitory inputs
        delay = round(abs((IE_delay-1)+randn(1)*sqrt(0.25))/(1000*step)); % 3/12/2018   adding noise to the delay as shown in Wehr 2003. centered at 4ms.
        p = 0;
        P_relE(1) = P_0E;
        P_relI(1) = P_0I;
        for i=1:ipi:(stimulus_input_length)
            p = p+1; %Click number (starts with 1)
            %             jitter=round(randn(1)/(1000*step)); %1 ms jitter
            jitter = floor(gamrnd(2.54,0.007)*1e3); %jitter with gamma distribution with parameters extracted from stim latency data (real neurons)
            %             jitter = jitter*1;
            while (i+jitter)<1 || (i+jitter)>(length(input)-length(kernel)) || jitter >ipi
                %                 while jitter >ipi
                                jitter = floor(gamrnd(2.54,0.007)*1e3); %original
%                 jitter= 17+ jitt*round(randn(1)/(1000*step));
                %                 end
            end
            t0 = i+jitter;
            if i == 1
                P_relE(1:t0) = P_0E;
                P_relI(1:t0) = P_0I;
            end
            for t = t0:t0+2*ipi-1
                if adaptation.E ==1 % depression
                    P_relE(t+1) = P_0E + (f_DE*P_relE(t0)-P_0E)*exp(-(t+1-t0)*step/tau_pE);
                else % Facilitation
                    P_relE(t+1) = P_0E + (P_relE(t0) + f_E*(1-P_relE(t0))-P_0E)*exp(-(t+1-t0)*step/tau_pE);
                end
                if adaptation.I==1
                    P_relI(t+1) = P_0I + (f_DI*P_relI(t0)-P_0I)*exp(-(t+1-t0)*step/tau_pI);
                else
                    P_relI(t+1) = P_0I + (P_relI(t0) + f_I*(1-P_relI(t0))-P_0I)*exp(-(t+1-t0)*step/tau_pI);
                end
                
            end
            
            E_str(p) = P_relE(t0)*E_strength;
            I_str(p) = P_relI(t0)*I_strength;
            E_input((latency+t0):(latency+t0+length(kernel)-1))=E_input((latency+t0):(latency+t0+length(kernel)-1))+ kernel*E_str(p); % exponential  +tau_m(1)/(tau_d-tau_r)*kernel*E_str(p);
            %             jitter=round(randn(1)/(1000*step)); %1 ms jitter
            %
            %             if (t0)<1 || (t0)>(length(input)-length(kernel))
            %                 jitter=0;
            %             end
            I_input((latency+t0+delay):(latency+t0+delay+length(kernel)-1))=I_input((latency+t0+delay):(latency+t0+delay+length(kernel)-1))+ kernel*I_str(p); %exponential +tau_m(2)/(tau_d-tau_r)*kernel*I_str(p);
            
            %             plot(P_relE)
            %             hold on
        end
        E_strength_mean = [E_strength_mean ; E_str];
        I_strength_mean = [I_strength_mean ; I_str];
        
        
        % for pure-tone responses, convolve alpha kernel with a rect
        % distribution
        
        
        
    end
    
    %   %
    delay=round(abs(IE_delay)/(1000*step));  %delay in steps
    
    if IE_delay>=0
        Ge=E_input;
        Gi=I_input;
        
        %         Gi=[zeros(1,delay) I_input(1:(length(I_input)-delay))];
    elseif IE_delay<0
        Gi=I_input;
        Ge=[zeros(1,delay) E_input(1:(length(E_input)-delay))];
    end
    
    
    
    %add pre  stim time of 500 ms
    Ge=[zeros(size(0:step:PREstimulus_duration)) Ge];
    Gi=[zeros(size(0:step:PREstimulus_duration)) Gi];
    Ge_total = [Ge_total; Ge];
    Gi_total = [Gi_total; Gi];
    Net_excit_total = [Net_excit_total; Ge-Gi];
    
    if PureTone ==1
        Ge = Ge*0.5;
        Gi = Gi*0.5;
        %         rect = [ones(1,length(0:step:stimulus_duration)) zeros(1,length(0:step:POSTstimulus_duration))];
        %         pureKernel = conv(rect,kernel);
        %         E_input = pureKernel.*[P_relE ones(1,(length(pureKernel-length(P_relE))];
        %         E_input = E_input*E_strength*0.1; %0.1 is a factor to have an onset response similar to
    end
    
    %avoid negative conductances
    Ge=Ge+noise_magnitude*randn(1,length(Ge));
    Gi=Gi+noise_magnitude*randn(1,length(Gi));
    Ge(find(Ge<0))=0;
    Gi(find(Gi<0))=0;
    
    %     test = 1;
    [spikes,V]=run_LIFmodel(Ge,Gi);
    
    %rate
    rate = zeros(1,total_time);
    for st = spikes
        if round(st/(step*10))>0 && round(st/(step*10)) <= length(rate)
            rate(1,round(st/(step*10))) = rate(1,round(st/(step*10)))+1;
        end
    end
    rate_total = [rate_total ; rate*1000];
    
    %ISI
    spikes3 = spikes(find(spikes<PREstimulus_duration+0.5));
    spikes4 = [0 spikes3(1:end-1)];
    isi = spikes3-spikes4;
    isi = isi(2:end);
    %pool spikes
    spikes=spikes-PREstimulus_duration;
    spikes_pooled=[spikes_pooled spikes];
    
    %
    raster.stim=[raster.stim f*ones(size(spikes))];
    raster.rep=[raster.rep r*ones(size(spikes))];
    raster.spikes=[raster.spikes spikes];
    
    %     figure
    %     plot(Ge)
    %     hold on
    %     plot(Gi)
end
out.raster = raster;


spikes_pooled_for_vector_strength=spikes_pooled(find(spikes_pooled>0.05 & spikes_pooled<=(stimulus_duration+0.05)));

%for calculating vector strength, subtract the first 50 ms and
%include 50 ms post stimulus
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


%  average rate

PRE = PREstimulus_duration*1000;
POST = POSTstimulus_duration*1000;
STIM = stimulus_duration*1000;
total_time = PRE+POST+STIM;
rate_av = mean(rate_total,1);
spont_rate = mean2(rate_total(:,1:PRE));
discharge_rate.mean = mean2(rate_total(:,PRE+1:PRE+STIM+100))-spont_rate;
SEM = std2(rate_total(:,PRE+1:PRE+STIM+100))/sqrt(nb_rep*(STIM+100));
ts = tinv([0.025  0.975],nb_rep*(STIM+100)-1);      % T-Score 95%
discharge_rate.error = ts(2)*SEM;

%

% CI = mean(x) + ts*SEM;                      % Confidence Intervals



xs = 1:total_time;
h = 12; % 10; %kernal bandwidth. determines the shape of the function
for i = 1:total_time
    ys(i)=gaussian_kern_reg(xs(i),xs,rate_av,h);
end

% spikes per click
spikes_per_click = {};
p = floor(STIM/ICI);
for q = 1:p
    clicktime = round((q-1)*ICI)+15+PRE; %plus input latency + kernel peak
    spikes_per_click.brut(:,q) = mean(rate_total(:,clicktime - 10 : clicktime + 10),2);
    spikes_per_click.mean(q) = mean2(rate_total(:,clicktime - 10 : clicktime + 10));
    spikes_per_click.std(q) = std2(rate_total(:,clicktime - 10 : clicktime + 10))/sqrt(20*nb_rep);
    spikes_per_click.xaxis(q) = clicktime;
end


% Fanofactor
SpikeCount = sum(rate_total,2)/1000.;
out.Fanofactor = std(SpikeCount)^2/mean(SpikeCount);


%Var ISI
out.var_ISI = std(isi);

out.rate_brut = rate_av;
out.rate_total = rate_total;
out.rates_stim = rate_total(:,PRE+1:PRE+STIM+100);
out.rate = ys;
out.VS = vector;
out.spikes_per_click = spikes_per_click ;
out.discharge_rate = discharge_rate;

% evolution of E_strength and I_strength

out.E_strength = mean(E_strength_mean,1);
out.I_strength = mean(I_strength_mean,1);
for ind = 1:length(out.I_strength)
    out.IE_ratio(ind) = out.I_strength(ind)/out.E_strength(ind);
end












function [spikes,V]=run_LIFmodel(Ge,Gi)
spikes=[]; V=[]; t=1; i=1;
step=.0001; %.1 ms duration  (temporal increment for running simulation)
C=0.25*1e-9; %0.25 nF, 10 ms time constant
Grest=25*1e-9; %25 nS
Erest=-0.065; %-65 mV
Ee=0; % 0 mV
Ei=-0.085;  %-85 mV
Ek = -0.075;



% noise_magnitude=4*1e-8; %default noise level in conductance

% %avoid negative conductances
% Ge=Ge+noise_magnitude*randn(1,length(Ge));
% Gi=Gi+noise_magnitude*randn(1,length(Gi));
% Ge(find(Ge<0))=0;
% Gi(find(Gi<0))=0;
sigma = 0.01    ;

%spike rate adaptation
Gsra = zeros(1,length(Ge));
% tau_sra = 0.1; %100ms
% delta_sra = 1e-9;
delta_sra = 0*1e-9;

% f_sra = 0.998;
f_sra = 0.995;



V(1)=Erest; %Initializing voltage
% running without spike-rate adaptation. to include this adaptation, change
% voltage equation.
while(t<length(Ge))
    %     V(t+1)=(-step*( Ge(t)*(V(t)-Ee) + Gi(t)*(V(t)-Ei) + Grest*(V(t)-Erest))/C)+V(t) + sigma*randn*sqrt(step); %+Gsra(t)*(V(t)-Ek))/C)
    V(t+1)=(-step*( Ge(t)*(V(t)-Ee) + Gi(t)*(V(t)-Ei) + Grest*(V(t)-Erest)+Gsra(t)*(V(t)-Ek) )/C)+V(t) + sigma*randn*sqrt(step); % % if SRA
    Gsra(t+1) = Gsra(t)*f_sra; % + sigma*randn*sqrt(step);
    if V(t+1)>(Erest+0.020) %20 mV above Erest %artificial threshold
        V(t+1)=0.050; %spike to 50 mV
        spikes(i)=step*(t+1);
        Gsra(t+1) = Gsra(t) +delta_sra;
        i=i+1;
        t=t+1;
        V(t+1)=Erest;
        Gsra(t+1) = Gsra(t)*f_sra;
        
    end
    t =t+1;
end
% figure
% plot(Gsra)
% %
% %
% test = 1;



