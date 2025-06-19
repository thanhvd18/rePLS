import os
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import cairosvg
from mayavi import mlab
from surfer import Brain
import nibabel as nib
from nibabel import freesurfer as nfs
from scipy.interpolate import interp1d

def combine_images_layout_3(image_paths, output_path, padding=50):
    """
    Combine multiple images into a single image with specified padding.

    Parameters:
    - image_paths: List of paths to the images to be combined.
    - output_path: Path to save the combined image.
    - padding: Padding between images in pixels.
    """
    # Load images
    images = [Image.open(path) for path in image_paths]

    # Rotate the second image by 90 degrees
    images[-1] = images[-1].rotate(-90)

    # Get image dimensions (assuming all are the same size)
    width, height = images[0].size

    # Create a new blank image (3 images wide, 1 row)
    new_width = 3 * width - 2 * padding
    new_height = height

    # Create a blank canvas
    combined = Image.new("RGB", (new_width, new_height))

    # Paste images with specified padding
    combined.paste(images[-1], (width - padding, 0))
    combined.paste(images[0], (0, 0))
    combined.paste(images[1], (2 * width - 2 * padding, 0))

    # Save or show the combined image
    combined.save(output_path)
    # combined.show()



# Convert hex colors to RGB
def hex_to_rgb(hex_color):
    """Convert hex color format '#RRGGBB' to normalized RGB [0, 1]"""
    hex_color = hex_color.lstrip("#")
    return [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

def region2surface(weights, annotation_file):

    """
    Map region weights to the brain surface based on the annotation file.

    Returns:
        ndarray: Surface weights.
    """
    rh = nfs.read_annot(annotation_file)
    atlas_left_cdata = rh[0]

    w_left_cdata = np.zeros(atlas_left_cdata.shape)
    w_right_cdata = np.zeros(atlas_left_cdata.shape)

    # if len(weights) == 202:
    #     weights[0] = np.NAN
    #     weights[1] = np.NAN

    for i in range(int(len(weights) / 2)):
        w_left_cdata[atlas_left_cdata == i] = weights[2 * i]
        w_right_cdata[atlas_left_cdata == i] = weights[2 * i + 1]

    surface_data = np.hstack([w_left_cdata, w_right_cdata])
    return surface_data


def vertex2color(gg_mesh, is_normalize=True, colormap=None):
    """
    Convert surface mesh weights to vertex colors based on a colormap with interpolation.

    Parameters:
        gg_mesh (ndarray): Surface mesh weights.
        is_normalize (bool): Flag to normalize weights.
        colormap (list): List of colors in the colormap.

    Returns:
        ndarray: Interpolated vertex colors in RGB format.
    """
    if colormap is None:
        # colormap = ['#08519c', '#3182bd', '#6baed6', '#bdd7e7', '#eff3ff',
        #             '#CCCCCC', '#fee5d9', '#fcae91', '#fb6a4a', '#de2d26', '#a50f15']
        colormap = ["#0000FF", "#3333FF", "#6666FF", "#9999FF", "#CCCCFF", "#FFFFFF", "#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#FF0000"]

    rgb_colormap = np.array([hex_to_rgb(c) for c in colormap])

    if is_normalize:
        min_val = -1
        max_val = 1
    else:
        min_val = np.min(gg_mesh)
        max_val = np.max(gg_mesh)

    range_ = max_val - min_val
    steps = np.linspace(min_val, max_val, len(colormap))
    interpolator = interp1d(steps, rgb_colormap, axis=0, kind='linear', fill_value='extrapolate')
    vertex_color_rgb = interpolator(gg_mesh.ravel())
    return vertex_color_rgb


def png_path_to_plt_plot(path: str):
    """

    :param path:
    :return:
    """
    # Open an image using PIL
    image = Image.open(path)  # Replace with your image path
    # Convert to NumPy array
    image_array = np.array(image)
    # Display the image
    plt.imshow(image_array)
    plt.axis("off")  # Hide axes for a cleaner display
    plt.show()

def plot_layout_3(save_dir, post_fix, region_data, colorbar=True, colormap='jet_r', alpha=0.9, padding=70, min=None, max=None,wall_color = "#8ef5fc"):
    """
    Plots the left, right, and combined hemispheres of the brain using PySurfer.
    Captures five views: lateral & medial (for each hemisphere), dorsal (for both).

    Parameters:
        post_fix (str): Suffix for output filenames.
        region_data (dict): Data to map onto the brain surface.
        colorbar (bool): Whether to display the color bar.
        colormap (str): Colormap used for visualization.
        alpha (float): Transparency level (0-1).
        padding (int): Padding between images in the combined output.
        min (float): Minimum value for color scale (auto if None).
        max (float): Maximum value for color scale (auto if None).
    """
    #TODO check atlas_dir and subjects_dir
    atlas_dir = os.environ['ATLASDIR']
    mlab.options.offscreen = True  # Enable offscreen rendering
    subject_id = 'fsaverage'
    surf = 'inflated'
    subjects_dir = os.environ['SUBJECTS_DIR']

    if min is None or max is None:
        all_surface_values = []
        for hemi in ['lh', 'rh']:
            atlas_file = os.path.join(atlas_dir, f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot")
            all_surface_values.append(region2surface(region_data, atlas_file))

        min = np.nanmin(np.concatenate(all_surface_values)) if min is None else min
        max = np.nanmax(np.concatenate(all_surface_values)) if max is None else max

    for hemi in ['lh', 'rh']:
        for view in ['lat']:  # Capture two views per hemisphere
            _plot_brain_hemi(save_dir, atlas_dir, hemi, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view,wall_color=wall_color)

    _plot_brain_combined(atlas_dir, save_dir, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view="dorsal")

    combined_image_path = os.path.join(save_dir, f'combined_{post_fix}.png')
    image_paths = [os.path.join(save_dir, f'{hemi}_{view}_{post_fix}.png') for hemi in ['lh', 'rh'] for view in
                   ['lat']]
    image_paths.append(os.path.join(save_dir, f'both_dorsal_{post_fix}.png'))  # Add dorsal view
    combine_images_layout_3(image_paths, combined_image_path, padding=padding)
    return png_path_to_plt_plot(combined_image_path)


def plot_layout_5(save_dir, post_fix, region_data, colorbar=True, colormap='jet_r', alpha=0.9, padding=70, min=None, max=None, wall_color="#8ef5fc"):
    """
    Plots the left, right, and combined hemispheres of the brain using PySurfer.
    Captures five views: lateral & medial (for each hemisphere), dorsal (for both).

    Parameters:
        post_fix (str): Suffix for output filenames.
        region_data (dict): Data to map onto the brain surface.
        colorbar (bool): Whether to display the color bar.
        colormap (str): Colormap used for visualization.
        alpha (float): Transparency level (0-1).
        padding (int): Padding between images in the combined output.
        min (float): Minimum value for color scale (auto if None).
        max (float): Maximum value for color scale (auto if None).
    """
    atlas_dir = os.environ['ATLASDIR']
    mlab.options.offscreen = True  # Enable offscreen rendering
    subject_id = 'fsaverage'
    surf = 'inflated'
    subjects_dir = os.environ['SUBJECTS_DIR']

    # Determine min/max if not provided
    if min is None or max is None:
        all_surface_values = []
        for hemi in ['lh', 'rh']:
            atlas_file = os.path.join(atlas_dir, f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot")
            all_surface_values.append(region2surface(region_data, atlas_file))

        min = np.nanmin(np.concatenate(all_surface_values)) if min is None else min
        max = np.nanmax(np.concatenate(all_surface_values)) if max is None else max

    # Plot each hemisphere with two views: lateral & medial
    for hemi in ['lh', 'rh']:
        for view in ['lat', 'med']:  # Capture two views per hemisphere
            _plot_brain_hemi(save_dir, atlas_dir, hemi, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view,wall_color=wall_color)

    # Plot combined hemispheres with dorsal view
    _plot_brain_combined(atlas_dir, save_dir, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view="dorsal")

    # Combine images into a single file
    combined_image_path = os.path.join(save_dir, f'combined_{post_fix}.png')
    image_paths = [os.path.join(save_dir, f'{hemi}_{view}_{post_fix}.png') for hemi in ['lh', 'rh'] for view in
                   ['lat', 'med']]
    image_paths.append(os.path.join(save_dir, f'both_dorsal_{post_fix}.png'))  # Add dorsal view
    combine_images_layout_5(image_paths, combined_image_path, padding=padding)
    return png_path_to_plt_plot(combined_image_path)
    #return fig object


def combine_images_layout_5(image_paths, output_path, padding=30):
    """
    Combine 5 brain images into a single layout:
    - Top row: LH lateral, Both (front), RH lateral
    - Bottom row: LH medial, RH medial

    Parameters:
    - image_paths: List of 5 image paths in order: [lh_lateral, rh_lateral, both, lh_medial, rh_medial]
    - output_path: Path to save the combined image.
    - padding: Padding between images in pixels.
    """
    # Load images
    images = [Image.open(path) for path in image_paths]

    #rotate image 2 (both) by 90 degrees
    images[-1] = images[-1].rotate(-90)
    # Get image dimensions (assuming all images have the same size)
    width, height = images[0].size

    # Define new canvas size
    new_width = (3 * width) + (2 * padding)  # 3 images wide in top row
    new_height = (2 * height)       # 2 rows with padding in between

    # Create a blank canvas
    combined = Image.new("RGB", (new_width, new_height), "white")

    # Paste images onto the canvas
    # Top row (3 images: LH lateral, Both, RH lateral)
    combined.paste(images[-1], (width + padding, height//2))  # Both (center)
    combined.paste(images[0], (0, 0))  # LH lateral
    combined.paste(images[2], (2 * width + 2 * padding, 0))  # RH lateral

    # Bottom row (2 images: LH medial, RH medial)
    combined.paste(images[1], (0 , height + padding))  # LH medial (shifted left)
    combined.paste(images[3], (2 * width + 2 * padding, height + padding))  # RH medial (shifted right)

    # Save and show the combined image
    combined.save(output_path)
    # combined.show()

def _plot_brain_hemi(save_dir, atlas_dir, hemi, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view,wall_color = "#8ef5fc"):
    """ Helper function to plot a single hemisphere with a specific view (lateral/medial). """
    # wall_color = "#8ef5fc"
    # wall_color = "dark"
    atlas_file = os.path.join(atlas_dir, f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot")
    subjects_dir = os.environ['SUBJECTS_DIR']
    labels, ctab, names = nib.freesurfer.read_annot(atlas_file)
    surface_value = region2surface(region_data, atlas_file)
    if hemi == "lh":
        surface_value = surface_value[:327684//2]
    elif hemi == "rh":
        surface_value = surface_value[327684//2:]
    else:
        raise ValueError("Invalid hemisphere")

    fig = mlab.figure()
    b = Brain('fsaverage', hemi, surf, subjects_dir=subjects_dir, background='white', figure=fig, cortex='low_contrast')

    b.add_annotation((labels, ctab), borders=True, alpha=0.5, color="k", hemi=hemi)
    b.add_data(surface_value, min=min, max=max, alpha=alpha, colormap=colormap, transparent=True, hemi=hemi,
               colorbar=colorbar)
    if wall_color is not None:
        b.add_label("Medial_wall", color=wall_color, alpha=alpha, hemi=hemi)

    # Set the desired view ("lateral" or "medial")
    b.show_view(view)

    # Save and close
    mlab.savefig(os.path.join(save_dir, f'{hemi}_{view}_{post_fix}.png'))
    mlab.close()


def _plot_brain_combined(atlas_dir, save_dir, region_data, post_fix, surf, colormap, alpha, min, max, colorbar, view="dorsal"):
    """ Helper function to plot both hemispheres with a single view. """
    wall_color = "#8ef5fc"
    # wall_color = "dark"
    fig = mlab.figure()
    subjects_dir = os.environ['SUBJECTS_DIR']
    b = Brain('fsaverage', "both", surf, subjects_dir=subjects_dir, background='white', figure=fig,
              cortex='low_contrast')

    for hemi in ["lh", "rh"]:
        atlas_file = os.path.join(atlas_dir, f"{hemi}.Schaefer2018_200Parcels_7Networks_order.annot")
        labels, ctab, names = nib.freesurfer.read_annot(atlas_file)
        surface_value = region2surface(region_data, atlas_file)
        if hemi == "lh":
            surface_value = surface_value[:327684 // 2]
        elif hemi == "rh":
            surface_value = surface_value[327684 // 2:]
        else:
            raise ValueError("Invalid hemisphere")

        b.add_annotation((labels, ctab), borders=True, alpha=0.5, color="k", hemi=hemi, remove_existing=False)
        b.add_data(surface_value, min=min, max=max, alpha=alpha, colormap=colormap, transparent=True, hemi=hemi,
                   colorbar=colorbar)
        b.add_label("Medial_wall", color=wall_color, alpha=alpha, hemi=hemi)

    # Set the desired view ("dorsal")
    b.show_view(view)
    mlab.savefig(os.path.join(save_dir, f'both_{view}_{post_fix}.png'))
    mlab.close()


def custome_colormap(data, v=None, cutoff=0.2):
    if v is None:
        thresh = abs(cutoff * np.min([np.min(data), np.max(data)]))
        v = [np.min(data), -thresh, thresh, np.max(data)]

    color_thresh = (v - np.min(v)) / (np.max(v) - np.min(v))
    color_positions = [(0, (1, 0, 0)), (color_thresh[1], (1, 1, 1)), (color_thresh[2], (1, 1, 1)), (1, (0, 0, 1))]
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", color_positions)
    return custom_cmap


def show_svg(file_path):
    """
    Display an SVG file in matplotlib fig
    """
    # Read the SVG file as bytes
    with open(file_path, "rb") as f:
        svg_data = f.read()

    # Convert SVG to PNG
    img_png = cairosvg.svg2png(bytestring=svg_data)
    img = Image.open(BytesIO(img_png))
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

def show_png(file_path):
    """
    Display a PNG file in matplotlib fig
    """
    img = Image.open(file_path)
    plt.imshow(img)
    plt.axis("off")  # Hide axes
    plt.show()

def show_image_from_folder(folder_path, extension:str = "svg"):
    """
    Display all SVG files in a folder
    """
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)]
    for file in files:
        show_png(file) if extension == "png" else show_svg(file)
