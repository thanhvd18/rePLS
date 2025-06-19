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

    dir = "./f1/";
    if ~exist(dir, 'dir')
        mkdir(dir)
    end

    %% read data and plot
    data = readtable(data_path);
    data = table2array(data);

    plot_brain_from_brain_weight("weight", ...
        data, "name", image_name, "colormap", color_map,"dir", dir, "min", v_min, "max", v_max);
end
