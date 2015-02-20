function out=LIFmodel_basic(IE_delay, E_strength, IE_ratio)
%E_strength: strength of excitation
%I_strength:strength of inhibition
%IE_ratio:ratio of Excitation strength to Inhibition strength
%IE_delay:delay in milliseconds between excitation and inhibition


I_strength=IE_ratio*E_strength;
kernel_time_constant=.005;  %time constant of 5 ms
jitter_magnitude=1;
step=.0001; %.1 ms duration  (temporal increment for running simulation)
stimulus_duration=0.5;  %half second
PREstimulus_duration=0.5;  %half second
POSTstimulus_duration=0.5;  %half second (0.1 second is included in stimulus)
latency_time=0.01;
latency=length(0:step:latency_time); %10 ms latency (auditory nerve to auditory cortex)

%%%%%%%%%acoustic pulse train stimulus%%%%%%%%%%%%%%%%%
nreps=10;
ICI_list=[3 5 7.5 10 12.5 15 20 25 30 35 40 45 50 55 60 65 70 75];

freq_list=1000./ICI_list;
spike_distribution=NaN(length(ICI_list),nreps);
time_distribution=NaN(length(ICI_list),nreps);
spont_distribution=NaN(1,length(ICI_list)*nreps);
raster.stim=[];  raster.rep=[];  raster.spikes=[];
for f=1:length(freq_list)
    spikes_pooled=[];
    freq=freq_list(f);
    t=0:step:(kernel_time_constant*10);
    kernel=t.*exp(-t/kernel_time_constant);
    kernel=1e-9*kernel/max(kernel); %amplitude of 1 nS
    input=zeros(size(0:step:(POSTstimulus_duration+stimulus_duration)));
    stimulus_input_length=length(0:step:(stimulus_duration));
    ipi=round(1/(freq*step)); %ipi=interpulse interval
    freq2=1/(step*ipi);

    for r=1:nreps
        E_input=input;
        I_input=input;
        for j=1:10  %10 jitter excitatory and inhibitory inputs
            % for i=1:ipi:(stimulus_input_length-(length(kernel)/2))
            for i=1:ipi:(stimulus_input_length-250)
                jitter=round(randn(1)/(1000*step)); %1 ms jitter
                if (jitter+i)<1 | (jitter+i)>(length(input)-length(kernel))
                    jitter=0;
                end
                E_input((latency+i+jitter):(latency+i+jitter+length(kernel)-1))=E_input((latency+i+jitter):(latency+i+jitter+length(kernel)-1))+kernel;


                jitter=round(randn(1)/(1000*step)); %1 ms jitter

                if (jitter+i)<1 | (jitter+i)>(length(input)-length(kernel))
                    jitter=0;
                end
                I_input((latency+i+jitter):(latency+i+jitter+length(kernel)-1))=I_input((latency+i+jitter):(latency+i+jitter+length(kernel)-1))+kernel;
            end
        end
        
        delay=round(abs(IE_delay)/(1000*step));  %delay in steps
        if IE_delay>=0
            Ge=E_input*E_strength;
            Gi=[zeros(1,delay) I_input(1:(length(I_input)-delay))]*I_strength;
        elseif IE_delay<0
            Gi=I_input*I_strength;
            Ge=[zeros(1,delay) E_input(1:(length(E_input)-delay))]*E_strength;
        end

        %add pre  stim time of 500 ms
        Ge=[zeros(size(0:step:PREstimulus_duration)) Ge];
        Gi=[zeros(size(0:step:PREstimulus_duration)) Gi];
        [spikes,V]=run_LIFmodel(Ge,Gi);

        spikes=spikes-PREstimulus_duration;
        spikes_pooled=[spikes_pooled spikes];

        spike_distribution(f,r)=length(spikes(find(spikes>0 & spikes<=(stimulus_duration+0.1))))/(stimulus_duration+0.1);
        spont_distribution(r+nreps*(f-1))=length(spikes(find(spikes>-PREstimulus_duration & spikes<0)))/PREstimulus_duration;

        raster.stim=[raster.stim f*ones(size(spikes))];
        raster.rep=[raster.rep r*ones(size(spikes))];
        raster.spikes=[raster.spikes spikes];
    end

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
            vector(f)=0;
        else
            vector(f)=sqrt(x^2+y^2)/total_spikes;
        end
        rayleigh(f)=2*total_spikes*vector(f)^2;
    else
        vector(f)=0;
        rayleigh(f)=0;
    end

    if rayleigh(f)<13.8
        vector(f)=0;
    end
    %spike rate calculated over stimulus duration plus 100 ms post stimulus
    spike_rate(f)=length(find(spikes_pooled>0 & spikes_pooled<=(stimulus_duration+0.1)))/(nreps*(stimulus_duration+0.1));
end
mean_spont=mean(spont_distribution);
std_spont=std(spont_distribution);
spike_rate=spike_rate-mean_spont;
spike_rate_stderr=std(spike_distribution,0,2)'/sqrt(nreps);

out.spike_rate_stderr=spike_rate_stderr;
out.mean_spont=mean_spont;
out.std_spont=std_spont;
out.vector=vector;
out.spike_rate=spike_rate;
out.rayleigh=rayleigh;

fastICI=spike_rate(1);
slowICI=max(spike_rate(10:end));
if slowICI<1
    slowICI=1;
end



out.rayleigh_at_100ms=rayleigh(end);
temp=min(find(rayleigh>13.8));
if ~isempty(temp)
    out.sync_boundary=ICI_list(temp);
else
    out.sync_boundary=-1;
end
for i=1:length(ICI_list)
    temp(i)=ranksum(spike_distribution(i,:),spike_distribution(end,:));
end
if ~isempty(find(temp<0.05))
    out.rate_boundary=ICI_list(max(find(temp<0.05)));
else
    out.rate_boundary=NaN;
end

neuron_summary_plot(ICI_list,raster, std_spont,spike_rate,spike_rate_stderr,vector,nreps,...
    PREstimulus_duration, stimulus_duration, POSTstimulus_duration)



function [spikes,V]=run_LIFmodel(Ge,Gi)
spikes=[]; V=[]; t=1; i=1;
step=.0001; %.1 ms duration  (temporal increment for running simulation)
C=0.25*1e-9; %0.25 nF, 10 ms time constant
Grest=25*1e-9; %25 nS
Erest=-0.065; %-65 mV
Ee=0; % 0 mV
Ei=-0.085;  %-85 mV
V(1)=Erest;
noise_magnitude=4e-8; %default noise level in conductance

%avoid negative conductances
Ge=Ge+noise_magnitude*randn(1,length(Ge));
Gi=Gi+noise_magnitude*randn(1,length(Gi));
Ge(find(Ge<0))=0;
Gi(find(Gi<0))=0;

while(t<length(Ge))
    V(t+1)=(-step*(Ge(t)*(V(t)-Ee)+Gi(t)*(V(t)-Ei)+Grest*(V(t)-Erest))/C)+V(t);
    if V(t+1)>(Erest+0.020) %20 mV above Erest
        V(t+1)=0.050; %spike to 50 mV
        spikes(i)=step*(t+1);
        i=i+1;
        t=t+1;
        V(t+1)=Erest;
    end
    t=t+1;
end

function neuron_summary_plot(x,raster, std_spont,spike_rate,nstd,vector_strength,nreps,...
    PREstimulus_duration, stimulus_duration, POSTstimulus_duration)

xtext = [];
xtext_position = [];
for s = 1:2:length(x)
    xtext_position = [xtext_position;s];
    xtext = char(xtext,num2str(x(s)));
end
xtext(1,:) = [];

ytext = [];
ytext_position = [];
for s = 1:length(x)
    ytext_position = [ytext_position;s*nreps-nreps/2];
    ytext = char(ytext,num2str(x(s)));
end
ytext(1,:) = [];
figure
h=subplot(2,2,[1 3]);
hold on
xlabel('time (s)')
ylabel('IPI (ms)')
area([0 stimulus_duration stimulus_duration 0],[length(x)*nreps+1 length(x)*nreps+1 0 0],'LineStyle','none','FaceColor',[.85 .85 1]);
hold on
plot(raster.spikes,nreps*(raster.stim-1)+raster.rep,'k.','MarkerSize',9);
axis([-PREstimulus_duration stimulus_duration+POSTstimulus_duration 0 length(x)*nreps+1])
set(gca,'yTickMode','manual');
set(gca,'yTick',ytext_position);
set(gca,'yTickLabel',ytext);
set(h,'Box','off',...
    'LineWidth',1,'FontName','Arial','FontSize',24)
h=subplot(2,2,2);
hold on

if ~isempty(nstd)
    errorbars=NaN(1,length(spike_rate)*3);
    errorbars(1:3:length(spike_rate)*3)=spike_rate+nstd;
    errorbars(2:3:length(spike_rate)*3)=spike_rate-nstd;
    xx=NaN(1,length(spike_rate)*3);
    xx(1:3:length(spike_rate)*3)=1:length(x);
    xx(2:3:length(spike_rate)*3)=1:length(x);
    plot(xx,errorbars,'b')
end

plot(1:length(x),spike_rate,'b')
plot([1 length(x)],[2*std_spont 2*std_spont],'k--')
xlabel('IPI (ms)')
ylabel('discharge rate (spk/s)')
set(gca,'xTickMode','manual');
set(gca,'xTick',xtext_position);
set(gca,'xTickLabel',xtext);
set(h,'Box','off',...
    'LineWidth',1,'FontName','Arial','FontSize',24)
h=subplot(2,2,4);
hold on
plot(1:length(x),vector_strength,'b')

xlabel('IPI (ms)')
ylabel('vector strength')
set(gca,'xTickMode','manual');
set(gca,'xTick',xtext_position);
set(gca,'xTickLabel',xtext);
set(h,'Box','off',...
    'LineWidth',1,'FontName','Arial','FontSize',24)


