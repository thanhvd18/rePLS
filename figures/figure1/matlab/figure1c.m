clc, close all;

addpath(genpath("/Users/tth/Thanh/plotSurface"))
init;

% load csv and set name for image
data_path = "../1c/female_cortical_thickness.csv";
image_name = "female_cortical_thickness";
color_map = jet(64);
% color_map = flipud(color_map);
% color_map = "";
dir = ".";
data = readtable(data_path);
data = table2array(data);

plot_brain_from_brain_weight("weight", ...
    data, "name", image_name, "colormap", color_map,"dir", dir);


% data_path = "../1c/male_cortical_thickness.csv";
% image_name = "male_cortical_thickness";
% color_map = jet(64);
% color_map = flipud(color_map);
% % color_map = "";
% dir = ".";
% data = readtable(data_path);
% data = table2array(data);
% 
% plot_brain_from_brain_weight("weight", ...
%     data, "name", image_name, "colormap", color_map,"dir", dir);




%data_path = "/Users/tth/Thanh/rePLS/rePLS-figures/dev/group1_cortical_thickness.csv";
%image_name = "group1_cortical_thickness" + post_fix;
%color_map = jet(64);
%color_map = flipud(color_map);



%data_path = "/Users/tth/Thanh/rePLS/rePLS-figures/dev/group2_cortical_thickness.csv";
%image_name = "group2_cortical_thickness" + post_fix;
%color_map = jet(64);
%color_map = flipud(color_map);
%
%data_path = "/Users/tth/Thanh/rePLS/rePLS-figures/dev/group3_cortical_thickness.csv";
%image_name = "group3_cortical_thickness" + post_fix;
%color_map = jet(64);
%color_map = flipud(color_map);
%
%data_path = "/Users/tth/Thanh/rePLS/rePLS-figures/dev/group4_cortical_thickness.csv";
%image_name = "group4_cortical_thickness" + post_fix;
%color_map = jet(64);
%color_map = flipud(color_map);

% data_path = "/Users/tth/Thanh/rePLS/rePLS-figures/dev/mean.csv";
% image_name = "group1_cortical_thickness" + post_fix;
% color_map = jet(64);
%color_map = flipud(color_map);

%
% 
% dir = ".";
% data = readtable(data_path);
% data = table2array(data);
% 
% plot_brain_from_brain_weight("weight", ...
%     data, "name", image_name, "colormap", color_map,"dir", dir);