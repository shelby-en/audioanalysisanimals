clear; close all; clc;

names = dir("*.wav");
n = length(names);

for i = 1:n
    if (i ~= 23)
        [audio, sr] = audioread(names(i).name);
        audio = audio(:,1);
        figure(i);
        filt = bandpass(audio, [0.05, 0.4]);
        spectrogram(audio, 100, 10);
        % audiowrite(names(i).name(1:end-4) + "_filt.wav", filt, sr);

    end
    
    % spectrogram()
end

% [y, Fs] = audioread(dataPath + )