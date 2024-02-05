% given a channel number from the Open Ephys GUI,
% returns the corresponding electrode on the 64-EEG Neuroscan cap
%
% inputs:
% channel - channel number
%
% outputs:
% electrode - electrode name

function electrode = return_electrode(channel)
    electrodes = {
        'FT8','CZ','P7','TP8','P8','P4','OZ','O2','O1','PZ','CP4','C4','T8',...
        'P3','CP3','FC4','TP7','M1','F4','C7','FC3','FP2','M2','HEOL','F7',...
        'HEOR','FP1','F3','FT7','F8','FZ','C3','C1','CP5','C6','PO6','CP2',...
        'P2','P1','PO5','PO3','POZ','PO4','P6','CP6','PO8','PO7','CP1','C2',...
        'FC6','FC1','F6','F1','AF8','GND','FPZ','F5','AF3','AF7','AF4','FC5',...
        'F2','C5','FC2'
        };
    electrode = electrodes{channel};
end 