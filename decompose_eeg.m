% given a 64-channel EEG signal 
% returns the signal decomposed into five frequency bands
%
% inputs:
% signal - 64-channel EEG signal (row = channel)
% fs - sample frequency (Hz)
%
% outputs:
% delta - delta frequency band (<4 Hz)
% theta - theta frequency band (4-8 Hz)
% alpha - alpha frequency band (8-12 Hz)
% beta - beta frequency band (12-30Hz)
% gamma - gamma frequency band (>30 Hz)

function [delta, theta, alpha, beta, gamma] = decompose_eeg(signal, fs)
    delta = lowpass(signal, 4, fs);
    theta = bandpass(signal, [4,8], fs);
    alpha = bandpass(signal, [8,12], fs);
    beta = bandpass(signal, [12,30], fs);
    gamma = highpass(signal, 30, fs); 
end
