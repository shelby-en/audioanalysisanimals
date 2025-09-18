clear; close all; clc;

names = dir("*.wav");
n = length(names);

for i = 1:n
    if (i ~= 23)
        [audio, sr] = audioread(names(i).name);
        figure(i);
        spectrogram(audio, 100, 10);
    end
    
    % spectrogram()
end

% [y, Fs] = audioread(dataPath + )