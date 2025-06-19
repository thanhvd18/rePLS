clc, close all;
addpath(genpath('/Users/tth/Thanh/plotSurface'))
init;


image_names = {'', 'male_cortical_thickness'};
%% read data and plot
data_path = "../4a/csv/mean_P.csv";
data = readtable(data_path);
data = table2array(data);
cmin = min(data(:));
cmax = max(data(:));
[n_regions, n_components] = size(data);

i = 1;
data_i = data(:,i);
image_name = strcat("Pmap", num2str(i));
dir = "./4a/";
if ~exist(dir, 'dir')
    mkdir(dir)
end
color_map = mycolormap_blue(cmin,cmax);
plot_brain_from_brain_weight("weight", ...
    data_i, "name", image_name, "colormap", color_map,"dir", dir, "plot_full", true);

%% 

for i = 2:n_components
    data_i = data(:,i);
    image_name = strcat("Pmap", num2str(i));
    middleValue = 0;
    range_thresh = 0.1;
    cmin = min(data_i(:));
    cmax = max(data_i(:));
    color_map = mycolormap(middleValue,range_thresh,cmin,cmax);
    plot_brain_from_brain_weight("weight", ...
        data_i, "name", image_name, "colormap", color_map,"dir", dir,"plot_full", true);
end

