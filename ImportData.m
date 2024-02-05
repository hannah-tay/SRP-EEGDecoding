%% classify finger/reach and grasp movements
% 2024-01-15_10-25-00 / 2024-01-15_11-01-39
% 2024-01-15_13-02-44 / 2024-01-15_13-30-02
% 2024-01-17_11-20-15 / 2024-01-17_11-53-53
%
% 3x5x5 finger movement recordings (30s each): fingers open/close every 2s
% 3x5x5 reach and grasp recordings 

clc; clear; close all;

fs = 30000; % sampling rate: 30kS/s
fsnew = 500;
scale = 0.195; % multiply to get to microvolts 
f_cutoff = 15; % Hz
channels = [51,21,61,33,32,63,48,15,34,2,64,49,37]; % channels with gel applied
num_samples = 15; % number of recordings for each category across all datasets
num_channels = 64; 
num_participants = 3;

%% load and filter data (run to save finger_movements.mat)
% 3 participants = 3 folders
paths = {
    '../../Open Ephys/2024-01-15_10-25-00/',...
    '../../Open Ephys/2024-01-15_13-02-44/',...
    '../../Open Ephys/2024-01-17_11-20-15/'
    };

% dataset 
data = struct('thumb',[],'index',[],'middle',[],'ring',[],'pinky',[]);
for i = 1:num_participants
    if i == 1 % i messed up the data collection at first
        recordings = {
            'thumb', 'thumb', 'thumb', 'thumb', 'thumb',...
            'index', 'index', 'index', 'index', 'index',...
            'middle', 'ring', 'pinky', 'middle', 'ring',...
            'pinky', 'middle', 'ring', 'pinky', 'middle',...
            'ring', 'pinky', 'middle', 'ring', 'pinky'
            };
    else
        recordings = {
            'thumb', 'index', 'middle', 'ring', 'pinky',...
            'thumb', 'index', 'middle', 'ring', 'pinky',...
            'thumb', 'index', 'middle', 'ring', 'pinky',...
            'thumb', 'index', 'middle', 'ring', 'pinky',...
            'thumb', 'index', 'middle', 'ring', 'pinky'
            }; % should be this
    end

    for j = 1:25
        % read data
        data_path = [paths{i},'Record Node 121/experiment1/' ...
            'recording', num2str(j), '/continuous/Acquisition_Board-128.Rhythm Data/continuous.dat'];
        file = fopen(data_path,'rb');
        temp = fread(file,[num_channels, Inf],'int16'); 

        % resample data
        temp = resample_array(temp, fsnew, fs);
        temp = temp(:,1:fsnew*sample_length);

        % filter data
        for k = 1:num_channels
            temp(k,:) = lowpass(temp(k,:),f_cutoff,fsnew);
        end

        % save data
        if isempty(data.(recordings{j}))
            data.(recordings{j}) = temp;
        else
            data.(recordings{j}) = cat(3, data.(recordings{j}), temp);
        end
        fclose(file);
        clc;
    end
end

% save data struct
save('finger_movements.mat', 'data');

%% load and filter data (run to save reach_and_grasp.mat)
% 3 participants = 3 folders
paths = {
    '../../Open Ephys/2024-01-15_11-01-39/',...
    '../../Open Ephys/2024-01-15_13-30-02/',...
    '../../Open Ephys/2024-01-17_11-53-53/'
    };

% dataset 
data = struct('bowl',[],'glass',[],'mug',[],'bottle',[],'key',[]);
for i = 1:num_participants
    recordings = {
        'bowl', 'glass', 'mug', 'bottle', 'key',...
        'bowl', 'glass', 'mug', 'bottle', 'key',...
        'bowl', 'glass', 'mug', 'bottle', 'key',...
        'bowl', 'glass', 'mug', 'bottle', 'key',...
        'bowl', 'glass', 'mug', 'bottle', 'key'
        }; 

    for j = 1:25
        % read data
        data_path = [paths{i},'Record Node 121/experiment1/' ...
            'recording', num2str(j), '/continuous/Acquisition_Board-128.Rhythm Data/continuous.dat'];
        file = fopen(data_path,'rb');
        temp = fread(file,[num_channels, Inf],'int16'); 

        % resample data
        temp = resample_array(temp, fsnew, fs);
        temp = temp(:,1:fsnew*sample_length_2);

        % filter data
        for k = 1:num_channels
            temp(k,:) = lowpass(temp(k,:),f_cutoff,fsnew);
        end

        % save data
        if isempty(data.(recordings{j}))
            data.(recordings{j}) = temp;
        else
            data.(recordings{j}) = cat(3, data.(recordings{j}), temp);
        end
        fclose(file);
        clc;
    end
end

% save data struct
save('reach_and_grasp.mat', 'data');