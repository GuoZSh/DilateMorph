import glob
import os, utils

import matplotlib
from matplotlib import cm
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.TransMorph import CONFIGS as CONFIGS_TM
import models.TransMorph as TransMorph
import torch.nn as nn
from visualizer import get_local
get_local.activate()
import matplotlib.pyplot as plt
import matplotlib as mpl
from skimage import color
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
from PIL import Image, ImageDraw
import nibabel as nib
import cv2
torch.cuda.empty_cache()

def main():
    atlas_dir = '/root/regis/IXI_data/atlas.pkl'
    test_dir = '/root/regis/IXI_data/Test/'
    save_path = '/root/regis/TransMorph_Transformer_for_Medical_Image_Registration-main/IXI/Figures/TransMorph/'

    model_idx = -1
    weights = [1, 1]
    model_folder = 'TransMorph_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments_IXI/' + model_folder

    config = CONFIGS_TM['TransMorph']
    model = TransMorph.TransMorph(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(config.img_size, 'bilinear')
    reg_model.cuda()
    reg_model_bilin = utils.register_model(config.img_size, 'bilinear')
    reg_model_bilin.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType((np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x,y),dim=1)
            x_def, flow = model(x_in)
            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            #visulize deformation field
            #visulaize_deformation_field(flow)
            encode_deformation_field(flow,save_path)

            #visulize fixed,moving,moved
            vis_orginal(y, x, x_def, x_seg, y_seg, def_out, save_path)

            #visulize by overlap
            addimage(y,x,x_def,save_path)

            #visulize by grid
            grid_img = mk_grid_img(8, 1, config.img_size)
            def_grid = reg_model_bilin([grid_img.float(), flow.cuda()])
            comput_fig(def_grid,save_path)

            #visualizer by attention
            cache = get_local.cache
            attn_map = cache['WindowAttention.forward']

            visualize_head(attn_map[9][0][2],save_path)
            visualize_grid_to_grid(y[0, 0, :, :], attn_map ,0, (50,160,180,100,200),9,2,save_path)
            get_local.clear()
            stdy_idx += 1
            if stdy_idx == 1:
                break


def color_boundaries(segmentation):
    # Create a colored image
    colored_image = cv2.cvtColor(np.uint8(segmentation), cv2.COLOR_GRAY2BGR)

    # Get unique label values
    unique_labels = np.unique(segmentation)
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)]
    # Detect boundaries for each label and mark with different colors
    for i, label in enumerate(unique_labels):
        if label == 0:  # Ignore background
            continue

        mask = (segmentation == label).astype(np.uint8)
        edges = cv2.Canny(mask, threshold1=0, threshold2=0.9)  # Use Canny edge detection
        color = colors[i % len(colors)]  # Select color in order
        colored_image[edges > 0] = color

    return colored_image

def vis_orginal(fix,moving,moved, x_seg, y_seg, def_out,save_path):
    #print(fix.shape)
    fix = fix.detach().cpu().numpy()[0, 0, 100, :, :]
    moving = moving.detach().cpu().numpy()[0, 0, 100, :, :]
    moved = moved.detach().cpu().numpy()[0, 0, 100, :, :]

    x_seg = x_seg.detach().cpu().numpy()[0, 0, 100, :, :]
    unique_elements_x, counts = np.unique(x_seg, return_counts=True)
    element_count_dict = dict(zip(unique_elements_x, counts))
    #print(element_count_dict)

    y_seg = y_seg.detach().cpu().numpy()[0, 0, 100, :, :]
    unique_elements_y, counts = np.unique(y_seg, return_counts=True)
    element_count_dict = dict(zip(unique_elements_y, counts))
    #print(element_count_dict)

    desire_values = np.union1d(unique_elements_x, unique_elements_y)
    def_out = def_out.detach().cpu().numpy()[0, 0, 100, :, :]
    closest_indices = np.argmin(np.abs(def_out[..., np.newaxis] - desire_values), axis=-1)
    def_out = desire_values[closest_indices]

    unique_elements_out, counts = np.unique(def_out, return_counts=True)
    element_count_dict = dict(zip(unique_elements_out, counts))
    #print(element_count_dict)

    desired_labels = [9, 7, 5]
    x_seg[~np.isin(x_seg,desired_labels)] = 0
    y_seg[~np.isin(y_seg,desired_labels)] = 0
    x_seg = cv2.medianBlur(x_seg.astype(np.uint8), 3)
    y_seg = cv2.medianBlur(y_seg.astype(np.uint8), 3)

    out = np.zeros((def_out.shape[0], def_out.shape[1]))


    for i in desired_labels:
        middle = def_out.copy()
        middle[middle != i] = 0
        middle = cv2.medianBlur(middle.astype(np.uint8), 5) # 使用5x5的中值滤波器
        out = np.where(middle != 0, middle, out)

    unique_elements_out, counts = np.unique(out, return_counts=True)
    element_count_dict = dict(zip(unique_elements_out, counts))
    print(element_count_dict)

    # Define the data and filenames for the loop
    data_list = [
        (fix, seg_fix, 'fix.png'),
        (moving, seg_moving, 'moving.png'),
        (moved, seg_moved, 'moved.png'),
        (y_seg, None, 'ground_truth.png'),
        (x_seg, None, 'input.png'),
        (out, None, 'pred.png')
    ]

    # Loop through the data and save images
    for data, overlay, filename in data_list:
        plt.imshow(data, cmap='gray')
        if overlay is not None:
            plt.imshow(overlay, cmap='viridis', alpha=0.6)
        plt.axis('off')
        plt.savefig(os.path.join(save_path, filename), dpi=300)
        plt.clf()  

def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img

def comput_fig(img,save_path):
    img = img.detach().cpu().numpy()[0, 0, 100, :, :]

    plt.figure(dpi=180)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_path,'grid.png'), dpi=300)

def visualize_head(att_map, save_path):
    ax = plt.gca()
    # Plot the heatmap
    im = ax.imshow(att_map)
    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    plt.savefig(os.path.join(save_path, 'head.png'), dpi=300)

def visualize_grid_to_grid(image, att_map, window_index, query_indexes, layer_index, head_index, save_path, window_size=(5,6,7), alpha=0.8):
    att_map = att_map[layer_index]
    window_number = att_map.shape[0]

    window_number = (round(window_number**(1/3)), round(window_number**(1/3)), round(window_number**(1/3)))
    att_map = att_map[:, head_index, :, :]
    H, W, D = image.shape   # 160, 192, 224
    att_map = np.array(att_map)
    attention = att_map[window_index]

    # Calculate the boundaries of the window and query in the original image
    # 1. Map the window's position range in the original image based on i
    i = window_index

    # Get the window's coordinates based on the index
    i_H_index, i_W_index, i_D_index = np.unravel_index(window_index, window_number)
    h, w, d = int(H/window_number[0]), int(W/window_number[1]), int(D/window_number[2])
    i_H_start, i_W_start, i_D_start = int(i_H_index*h), int(i_W_index*w), int(i_D_index*d)
    i_H_end, i_W_end, i_D_end = i_H_start+h, i_W_start+w, i_D_start+d

    print(f"Range of window_{i} in the original image:\n",
          f"(H: {i_H_start}-{i_H_end}, W: {i_W_start}-{i_W_end}, D: {i_D_start}-{i_D_end})")

    for query_index in query_indexes:
        j = query_index
        # 2. Map the query's position within the window based on j
        j_Hp_start, j_Wp_start, j_Dp_start = np.unravel_index(j, window_size)

        print(f"Position of query_{j} on the window:\n",
              f"(H: {j_Hp_start}, W: {j_Wp_start}, D: {j_Dp_start})\n")

        # Extract the query's weight and reshape it to the window size
        query_weight = attention[j]
        query_weight = query_weight.reshape(window_size[0], window_size[1], window_size[2])
        # Extract the attention slice of the corresponding slice
        query_weight = query_weight[j_Hp_start]
        # Scale the attention map inside the window to the size of the corresponding window in the original image
        query_weight = np.array(query_weight)
        query_weight = Image.fromarray(query_weight).resize((i_W_end-i_W_start, i_D_end-i_D_start), Image.Resampling.LANCZOS)
        query_weight = np.array(query_weight).T  # Convert the PIL image to a NumPy array
        # Extract the image slice
        img = image.detach().cpu().numpy()[(i_H_start+i_H_end)//2, :, :]
        # Extract the window
        img = img[i_W_start:i_W_end, i_D_start:i_D_end]
        query_weight = Image.fromarray(query_weight)

        plt.imshow(img, cmap='gray')
        plt.imshow(query_weight / np.max(query_weight), alpha=alpha, cmap='viridis')
        plt.axis('off')

        plt.savefig(os.path.join(save_path, 'attention_{}'.format(query_index)), dpi=300)


def encode_deformation_field(deformation_field, save_path, output_file="/root/regis/IXI/Flow/flow"):
    """
    Color-encode the entire deformation field and save it as a NIfTI file.

    Parameters:
        deformation_field (numpy.ndarray): Deformation field data with shape (depth, height, width, 3).
        output_file (str): Path to the output NIfTI file.

    Returns:
        None
    """

    # Create a color-encoded deformation field image
    def encode_deformation_field(deformation_field):
        # Normalize the deformation field to range [0, 1]
        deformation_field = deformation_field[0].detach().permute(1, 2, 3, 0).cpu().numpy()

        # Calculate max and min values in the x direction
        x_displacements = deformation_field[..., 0]
        max_x_displacement = round(np.max(x_displacements), 2)
        min_x_displacement = round(np.min(x_displacements), 2)

        # Calculate max and min values in the y direction
        y_displacements = deformation_field[..., 1]
        max_y_displacement = round(np.max(y_displacements), 2)
        min_y_displacement = round(np.min(y_displacements), 2)

        # Calculate max and min values in the z direction
        z_displacements = deformation_field[..., 2]
        max_z_displacement = round(np.max(z_displacements), 2)
        min_z_displacement = round(np.min(z_displacements), 2)

        # Calculate average displacements in x, y, z directions
        mean_x_displacement = round(np.mean(x_displacements), 2)
        mean_y_displacement = round(np.mean(y_displacements), 2)
        mean_z_displacement = round(np.mean(z_displacements), 2)

        # Calculate standard deviations of displacements in x, y, z directions
        std_x_displacement = round(np.std(x_displacements), 2)
        std_y_displacement = round(np.std(y_displacements), 2)
        std_z_displacement = round(np.std(z_displacements), 2)

        # Print results
        print(f"Displacement in x direction: [{min_x_displacement}, {max_x_displacement}]")
        print(f"Displacement in y direction: [{min_y_displacement}, {max_y_displacement}]")
        print(f"Displacement in z direction: [{min_z_displacement}, {max_z_displacement}]")
        print(f"Average displacement in x direction: {mean_x_displacement} ± {std_x_displacement}")
        print(f"Average displacement in y direction: {mean_y_displacement} ± {std_y_displacement}")
        print(f"Average displacement in z direction: {mean_z_displacement} ± {std_z_displacement}")

        deformation_magnitude = np.linalg.norm(deformation_field, axis=3)
        print(f"Average displacement: {round(np.mean(deformation_magnitude), 2)} ± {round(np.std(deformation_magnitude), 2)}")
        max_magnitude = round(np.max(deformation_magnitude), 2)
        print(f"Maximum absolute displacement: {max_magnitude}")

        normalized_magnitude = deformation_magnitude / max_magnitude

        # Create a color map to represent deformation direction
        cmap = ListedColormap(['red', 'green', 'blue'])

        # Encode based on deformation direction and magnitude
        hsv_image = np.zeros(deformation_field.shape[:3] + (3,))
        hsv_image[..., 0] = (np.arctan2(deformation_field[..., 1], deformation_field[..., 0]) + np.pi) / (2 * np.pi)
        hsv_image[..., 1] = 1.0
        hsv_image[..., 2] = normalized_magnitude

        # Convert to RGB color space
        rgb_image = color.hsv2rgb(hsv_image)

        return rgb_image

    # Color-encode the deformation field
    encoded_image = encode_deformation_field(deformation_field)

    data = encoded_image[100, :, :, :]
    data = np.array(data)
    plt.figure(dpi=180)
    plt.axis('off')
    plt.imshow(data, cmap='rainbow', norm=plt.Normalize(vmin=0, vmax=1), alpha=0.8)

    # Save the NIfTI image to file
    # nib.save(nifti_img, output_file)


def addimage(img_1, img_2, img_3, save_path):
    print(img_1.shape)
    img_1 = img_1[0, 0, 100, :, :].cpu()
    img_2 = img_2[0, 0, 100, :, :].cpu()
    img_3 = img_3[0, 0, 100, :, :].cpu()
    
    # Convert grayscale image to color image: use blue for the base and red for the top
    img = np.zeros((192, 224, 3))
    img[:, :, 0] = img_1 * 3  # Blue color
    img1 = img
    img = np.zeros((192, 224, 3))
    img[:, :, 2] = img_2 * 3
    img2 = img  # Red color for the moving image
    
    # Overlap between fixed and moving images
    overlap1 = cv2.addWeighted(img1, 0.5, img2, 0.5, 0)
    
    # Overlap between fixed and registered images
    img[:, :, 2] = img_3 * 3
    img3 = img
    overlap2 = cv2.addWeighted(img1, 0.5, img3, 0.5, 0)
    
    plt.imshow(overlap1)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'overlap1'), dpi=300)
    
    plt.imshow(overlap2)
    plt.axis('off')
    plt.savefig(os.path.join(save_path, 'overlap2'), dpi=300)




if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 0
    torch.cuda.set_device(GPU_iden)
    main()
