clc, close all;
addpath(genpath('/Users/tth/Thanh/plotSurface'))
init;


image_names = {'', 'male_cortical_thickness'};
%% read data and plot
data_path = "../6f/csv/CDR_weight.csv";
data = readtable(data_path);
data = table2array(data);
cmin = min(data(:));
cmax = max(data(:));
[n_regions, n_components] = size(data);

image_name = "CDR_weight";
dir = "./6f/";
if ~exist(dir, 'dir')
    mkdir(dir)
end
color_map = createCustomColormap();
plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", false);


%%

data_path = "../6f/csv/MMSE_weight.csv";
data = readtable(data_path);
data = table2array(data);
if length(data) == 204
    data = data(3:end);
end
cmin = min(data(:));
cmax = max(data(:));

[n_regions, n_components] = size(data);

image_name = "MMSE_weight";
dir = "./6f/";
if ~exist(dir, 'dir')
    mkdir(dir)
end
color_map = createCustomColormap();
plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", false);

