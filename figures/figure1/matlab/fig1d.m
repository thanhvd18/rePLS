clc, close all;
addpath(genpath('/Users/tth/Thanh/plotSurface'))
init;


image_names = {'group1_cortical_thickness', 'group2_cortical_thickness', 'group3_cortical_thickness', 'group4_cortical_thickness'};

v_min = inf;
v_max = -inf;
for i = 1:length(image_names)
    image_name = image_names{i};
    data_path = strcat('../1d/csv/', image_name, '.csv');
    %% read data and plot
    data = readtable(data_path);
    data = table2array(data);
    v_min = min(v_min, min(data));
    v_max = max(v_max, max(data));
end

for i = 1:length(image_names)
    image_name = image_names{i};
    data_path = strcat('../1d/csv/', image_name, '.csv');

    color_map = jet(64);
    color_map = flipud(color_map);

    dir = "./1d/";
    if ~exist(dir, 'dir')
        mkdir(dir)
    end

    %% read data and plot
    data = readtable(data_path);
    data = table2array(data);

    plot_brain_from_brain_weight("weight", ...
        data, "name", image_name, "colormap", color_map,"dir", dir, "min", v_min, "max", v_max,"plot_full", true);
end


%% plot ttest
% image_names = {"CN-MCI_ttest", "MCI-AD_ttest"}

image_name = "group1-group4_ttest";
data_path = strcat('../1d/csv/', image_name, '.csv');

data = readtable(data_path);
data = table2array(data);

middleValue = 1;
range_thresh = 0.1;
cmin = min(data(:));
cmax = max(data(:));
% color_map = mycolormap(middleValue,range_thresh,cmin,cmax);
color_map = mycolormap_only_positive(middleValue,range_thresh,cmin,cmax);

plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", false);

%%
