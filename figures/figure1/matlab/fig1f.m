clc, close all;
addpath(genpath('/Users/tth/Thanh/plotSurface'))
init;

image_name = "CN_group";
% create for loop for other image_name
image_names = {'CN_group', 'MCI_group', 'AD_group'};

v_min = inf;
v_max = -inf;
for i = 1:length(image_names)
    image_name = image_names{i};
    data_path = strcat('../1f/csv/', image_name, '.csv');
    %% read data and plot
    data = readtable(data_path);
    data = table2array(data);
    v_min = min(v_min, min(data));
    v_max = max(v_max, max(data));
end

for i = 1:length(image_names)
    image_name = image_names{i};
    data_path = strcat('../1f/csv/', image_name, '.csv');

    color_map = jet(64);
    color_map = flipud(color_map);

    dir = "./1f/";
    if ~exist(dir, 'dir')
        mkdir(dir)
    end

    %% read data and plot
    data = readtable(data_path);
    data = table2array(data);

    plot_brain_from_brain_weight("weight", ...
        data, "name", image_name, "colormap", color_map,"dir", dir, "min", v_min, "max", v_max);
end


%% plot ttest
% image_names = {"CN-MCI_ttest", "MCI-AD_ttest"}

image_name = "CN-MCI_ttest";
data_path = strcat('../1f/csv/', image_name, '.csv');

data = readtable(data_path);
data = table2array(data);
dir = '1f'

middleValue = 0;
range_thresh = 0.05;
cmin = -4.5
cmax = 20.5
color_map = mycolormap(middleValue,range_thresh,cmin,cmax);

plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", false, "min", cmin, "max", cmax);

%%


image_name = "MCI-AD_ttest";
data_path = strcat('../1f/csv/', image_name, '.csv');

data = readtable(data_path);
data = table2array(data);

middleValue = 2;
range_thresh = 0.1;
cmin = -4.5
cmax = 20.5
color_map = mycolormap_only_positive(middleValue,range_thresh,cmin,cmax);

plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", false, "min", cmin, "max", cmax);

