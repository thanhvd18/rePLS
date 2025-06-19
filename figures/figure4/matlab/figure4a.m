
clc, close all;

(addpath("/Users/tth/Thanh/plotSurface/src"))
init;

data_path = "../4a/mean_P.csv";
image_name = "Pmap";
color_map = jet(64);
color_map = flipud(color_map);

dir = ".";
%% read data and plot


data = readtable(data_path);
data = table2array(data);

% plot_brain_from_brain_weight("weight", ...
%     data, "name", image_name, "colormap", color_map,"dir", dir);

plot_brain_from_brain_weight("weight", ...
    data, "name", image_name,"dir", dir);



