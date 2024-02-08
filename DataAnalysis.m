%% classify finger/reach and grasp movements
%
% 3x5x5 finger movement recordings (30s each): fingers open/close every 2s
% 3x5x5 reach and grasp recordings 
%
% gel applied to 6 electrodes 
% - FC1(51), FC3(21), FC5(61), C1(33), C3(32), C5(63), CP1(48), CP3(15), 
% CP5(34), CZ(2), FC2(64), C2(49), CP2(37)

clc; clear; close all;

fs = 30000; % sampling rate: 30kS/s
fsnew = 500;
scale = 0.195; % multiply to get to microvolts 
f_cutoff = 15; % Hz
channels = [51,21,61,33,32,63,48,15,34,2,64,49,37]; % channels with gel applied
num_samples = 15; % number of recordings for each category across all datasets
num_channels = 64; 
num_participants = 3;

%% load data from .mat (run to load finger_movements.mat)
load('finger_movements.mat')
current_data = 'Finger Movements';
sample_length = 29; % desired sample length (seconds)
classes = {'thumb', 'index', 'middle', 'ring', 'pinky'};

% fprintf('Data loaded from finger_movements.mat\n')

%% load data from .mat (run to load reach_and_grasp.mat)
load('reach_and_grasp.mat')
sample_length = 4; 
current_data = 'Reach and Grasp';
classes = {'bowl', 'glass', 'mug', 'bottle', 'key'};

% fprintf('Data loaded from reach_and_grasp.mat\n')

%% visualise data structure
fprintf('Data:\n')
fprintf('--------------------------------------------------\n')
fprintf('Class           Channels Time           Recordings\n')
fprintf('--------------------------------------------------\n')

for i = 1:5
    fprintf('%-6s',classes{i})
    disp(size(data.(classes{i})))
end

%% eeg data plots
% plot EEG signal (channels with gel applied) for each class 
% note array structure is 64(channels) x (time) x 15(samples)

for i=1:length(classes)
    figure;
    data_plot = data.(classes{i})(:,:,1); % take just one sample

    for j=1:length(channels)
        subplot(4,4,j);
        plot(0:1/fsnew:(length(data_plot(channels(j),:))-1)/fsnew, data_plot(channels(j),:));
        title_str = return_electrode(channels(j));
        title(title_str);
        xlabel('Time(s)');
    end
    sgtitle(classes{i});
end

%% EEG data plot (one channel)
data_plot = data.(classes{1})(:,1:15*fsnew,1); % take just one sample
figure;
% stackedplot(1:length(data_plot(:,1)), data_plot);
plot(0:1/fsnew:(length(data_plot(channels(1),:))-1)/fsnew, 0.195*data_plot(channels(1),:), 'color', '#346CAA');
xlabel('Time (s)');
ylabel('Electrode Potential (uV)');

%% visualise averages across all 64 channels for each sample
% data 64 x time x 15 --> 1 x time x 15
data_averaged = struct;
for i=classes
    data_averaged.(string(i)) = mean(data.(string(i)),1);
end
% visualise channel averages
% plot just one sample for each class
figure;
for i=1:length(classes)
    subplot(2,3,i);
    plot(0:1/fsnew:(length(data_averaged.(classes{i})(:,:,1))-1)/fsnew, data_averaged.(classes{i})(:,:,1));
    title(classes{i});
end
sgtitle('Data Averaged Across EEG Channels');

%% classification per participant - whole sample
% MATLAB classify() - linear discriminant analysis (LDA)
% of the 25 data points (5 recordings x 5 fingers), train using 24 and test
% on 1 sample, then permute

labels = classes;
labels = repelem(labels, num_samples/num_participants); 

% populate array for classification
data_classify = zeros(length(classes)*num_samples/num_participants, num_channels*sample_length*fsnew);
k = 1;
for i=1:length(classes)
    for j=11:15 % first participant 1:5; second 6:10; third 11:15
        % 'unroll' 64x14500 data into 1D array
        temp = data.(classes{i})(:,:,j);
        data_classify(k,:) = reshape(temp, [1,num_channels*sample_length*fsnew]);
        k = k+1;
    end
end
% need to shorten arrays for classify()
% apply MATLAB pca()
[coeff, data_classify_pca] = pca(data_classify);
data_classify_pca = data_classify_pca(:,1:19);

% shuffle new dataset
rand_indices = randperm(size(data_classify_pca,1));
data_classify_shuffled = data_classify_pca(rand_indices,:);
labels_shuffled = labels(rand_indices);
predicted = {25};

for i = 1:25
    % permutation - test on all samples
    train_idx = 1:25;
    train_idx(i) = [];
    sample_idx = i;

    % partition data into training and sample sets
    training_data = data_classify_shuffled(train_idx,:);
    sample_data = data_classify_shuffled(sample_idx,:);
    
    % apply classify()
    classify_results = classify(sample_data, training_data, labels_shuffled(train_idx));
    predicted(i) = classify_results;
    % fprintf('Predicted: %s; Actual: %s\n', string(classify_results), string(labels(sample_idx)));
end

% display confusion matrix
figure;
confmat = confusionchart(labels_shuffled, predicted, 'Normalization', 'row-normalized');
title('Confusion Matrix - Reach and Grasp (Participant 3)');

%% classification per participant - shortened sample
% MATLAB classify() - linear discriminant analysis (LDA)
% classify only 0.5s of data at one time
%
% of the 25 data points (5 recordings x 5 fingers), train using 24 and test
% on 1 sample, then permute

% timestamp(s)      expected movement
% ---------------------------------
% 0                 open (resting)
% 2                 close
% 4                 open
% 6                 close
% 8                 open
% etc.

labels = classes;
labels = repelem(labels, num_samples/num_participants); 
new_sample_length = 0.5; % classify half a second of recording
offset = 2*fsnew; % modify to change which slice is being classified

% populate array for classification
data_classify = zeros(length(classes)*num_samples/num_participants, num_channels*new_sample_length*fsnew);
k = 1;
for i=1:length(classes)
    for j=1:5 % first participant 1:5; second 6:10; third 11:15
        % 'unroll' 64x14500 data into 1D array
        temp = data.(classes{i})(:,offset:offset+new_sample_length*fsnew-1,j);
        data_classify(k,:) = reshape(temp, [1,num_channels*new_sample_length*fsnew]);
        k = k+1;
    end
end
% need to shorten arrays for classify()
% apply MATLAB pca()
[coeff, data_classify_pca] = pca(data_classify);
data_classify_pca = data_classify_pca(:,1:19);

% shuffle new dataset
rand_indices = randperm(size(data_classify_pca,1));
data_classify_shuffled = data_classify_pca(rand_indices,:);
labels_shuffled = labels(rand_indices);
predicted = {25};

for i = 1:25
    % permutation - test on all samples
    train_idx = 1:25;
    train_idx(i) = [];
    sample_idx = i;

    % partition data into training and sample sets
    training_data = data_classify_shuffled(train_idx,:);
    sample_data = data_classify_shuffled(sample_idx,:);
    
    % apply classify()
    classify_results = classify(sample_data, training_data, labels_shuffled(train_idx));
    predicted(i) = classify_results;
    % fprintf('Predicted: %s; Actual: %s\n', string(classify_results), string(labels(sample_idx)));
end

% display confusion matrix
figure;
confmat = confusionchart(labels_shuffled, predicted, 'Normalization', 'row-normalized', ...
    'OffDiagonalColor', '#844564', 'DiagonalColor', '#346CAA');
% , 'Normalization', 'row-normalized'
title('Confusion Matrix - During Finger Movement (Participant 1)');

%% classification per participant - 0.5s blocks, stepping through 
% MATLAB classify() - linear discriminant analysis (LDA)
% visualise classifier performance before, during, and after movements
% use sample between 0 and 4 seconds (flexion at 2s)
% 
% for each slice of data:
% of the 25 data points (5 recordings x 5 fingers), train using 24 and test
% on 1 sample, then permute

labels = classes;
labels = repelem(labels, num_samples/num_participants); 
new_sample_length = 4; 
step_distance = 0.05*fsnew;
step_width = 0.5*fsnew; 
offset = 1; % start from beginning of sample 

% populate array for classification
data_classify = zeros(length(classes)*num_samples/num_participants, num_channels, new_sample_length*fsnew);
k = 1;
for i=1:length(classes)
    for j=11:15 % first participant 1:5; second 6:10; third 11:15
        data_classify(k,:,:) = permute(data.(classes{i})(:,offset:offset+new_sample_length*fsnew-1,j), [3,1,2]);
        k = k+1;
    end
end

% step through 0.5s sections of data in 0.05s intervals
% calculate classifier accuracy at each interval
num_slices = new_sample_length*fsnew/step_distance - step_width/step_distance;
accuracy = {num_slices};

data_slice_reshaped = zeros(length(classes)*num_samples/num_participants, num_channels*(step_width+1));
for i=1:num_slices
    data_slice = data_classify(:,:,i*step_distance:i*step_distance+step_width);
    % combine channels and time dimensions
    for j=1:25
        data_slice_reshaped(j,:) = reshape(data_slice(j,:,:), [1,num_channels*(step_width+1)]);
    end

    % need to shorten arrays for classify()
    % apply MATLAB pca()
    [coeff, data_classify_pca] = pca(data_slice_reshaped);
    data_classify_pca = data_classify_pca(:,1:19);

    % shuffle new dataset
    rand_indices = randperm(size(data_classify_pca,1));
    data_classify_shuffled = data_classify_pca(rand_indices,:);
    labels_shuffled = labels(rand_indices);
    predicted = {25};

    for j = 1:25
        % permutation - test on all samples
        train_idx = 1:25;
        train_idx(j) = [];
        sample_idx = j;
    
        % partition data into training and sample sets
        training_data = data_classify_shuffled(train_idx,:);
        sample_data = data_classify_shuffled(sample_idx,:);
        
        % apply classify()
        classify_results = classify(sample_data, training_data, labels_shuffled(train_idx));
        predicted(j) = classify_results;
        % fprintf('Predicted: %s; Actual: %s\n', string(classify_results), string(labels(sample_idx)));
    end

    % calculate accuracy
    num_correct = 0;
    for j=1:25
        if string(predicted(j)) == string(labels_shuffled(j))
            num_correct = num_correct + 1;
        end
    end
    accuracy{i} = num_correct/25;

    if i==39
        figure;
        confmat = confusionchart(labels_shuffled, predicted, 'Normalization', 'row-normalized', ...
            'OffDiagonalColor', '#844564', 'DiagonalColor', '#346CAA');
        title('Confusion Matrix - Picking Up Object (Participant 3)');
    end
end

% visualise how accuracy changes
figure;
plot(linspace(0,new_sample_length,length(accuracy)), cell2mat(accuracy), 'color', '#346CAA');
xlabel('Time(s)');
ylabel('Classifier Accuracy');
title('Classifier Performance at Different Time Intervals (Participant 3)');

% finger movements only
% xline([2,6], 'color', '#346CAA');
% legend('Accuracy', 'Expected Movement (Flexion)');

% reach and grasp only
xline(0.1, 'color', '#93649C');
xline(2.2, '--', 'color', '#844564');
legend('Accuracy', 'Expected Reach', 'Expected Grasp');

%% classification across participants - whole sample
% MATLAB classify() - linear discriminant analysis (LDA)
% of the 25 data points (5 recordings x 5 fingers), train using 24 and test
% on 1 sample, then permute

labels = classes;
labels = repelem(labels, num_samples); 

% populate array for classification
data_classify = zeros(length(classes)*num_samples, num_channels*sample_length*fsnew);
k = 1;
for i=1:length(classes)
    for j=1:15 
        % 'unroll' 64x14500 data into 1D array
        temp = data.(classes{i})(:,:,j);
        data_classify(k,:) = reshape(temp, [1,num_channels*sample_length*fsnew]);
        k = k+1;
    end
end
% need to shorten arrays for classify()
% apply MATLAB pca()
[coeff, data_classify_pca] = pca(data_classify);
data_classify_pca = data_classify_pca(:,1:19);

% shuffle new dataset
rand_indices = randperm(size(data_classify_pca,1));
data_classify_shuffled = data_classify_pca(rand_indices,:);
labels_shuffled = labels(rand_indices);
predicted = {length(classes)*num_samples};

for i = 1:length(classes)*num_samples
    % permutation - test on all samples
    train_idx = 1:length(classes)*num_samples;
    train_idx(i) = [];
    sample_idx = i;

    % partition data into training and sample sets
    training_data = data_classify_shuffled(train_idx,:);
    sample_data = data_classify_shuffled(sample_idx,:);
    
    % apply classify()
    classify_results = classify(sample_data, training_data, labels_shuffled(train_idx));
    predicted(i) = classify_results;
    % fprintf('Predicted: %s; Actual: %s\n', string(classify_results), string(labels(sample_idx)));
end

% display confusion matrix
figure;
confmat = confusionchart(labels_shuffled, predicted, 'Normalization', 'row-normalized', ...
    'OffDiagonalColor', '#844564', 'DiagonalColor', '#346CAA');
title('Confusion Matrix - Reach and Grasp (All Participants)');
