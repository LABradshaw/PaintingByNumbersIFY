#! /usr/bin/env python3
import numpy as np
import argparse
import os

import dominant_cluster
import image_utils
import process
import file_utils

def simple_matrix_to_image(mat, palette):
    simple_mat_flat = np.array(
        [[col for col in palette[index]] for index in mat.flatten()])
    return simple_mat_flat.reshape(mat.shape + (3,))


def PBNify(image_path, clusters=20, pre_blur=True):
    image = image_utils.load_image(image_path, resize=True)
    if pre_blur:
        image = process.blur_image(image)

    dominant_colors, quantized_labels, bar_image = dominant_cluster.get_dominant_colors(
        image, n_clusters=clusters, use_gpu=False, plot=True)

    # Create final PBN image
    smooth_labels = process.smoothen(quantized_labels.reshape(image.shape[:-1]))
    pbn_image = dominant_colors[smooth_labels].reshape(image.shape)

    # Create outline image
    outline_image = process.outline(pbn_image)

    return pbn_image, outline_image


def process_image(input_path, output_path, should_outline, num_clusters):
    """
    Processes a single image, generating PBN and outline images based on parameters.
    """
    print(f"Processing image: {input_path}")
    pbn_image, outline_image = PBNify(input_path, num_clusters)
    image_utils.save_image(pbn_image, output_path)

    if should_outline:
        outline_image_path = os.path.splitext(output_path)[0] + "_outline.jpg"
        image_utils.save_image(outline_image, outline_image_path)

def process_batch(directory_path, output_dir, should_outline, num_clusters):
    """
    Processes a batch of images in a specified directory.
    """
    files = file_utils.find_files(directory_path, ["jpg"])
    print(f"Processing batch: {directory_path}")
    print(f"{len(files)} images found")
    
    output = output_dir or 'processed_images'
    if not os.path.exists(output):
            os.makedirs(output)

    for file in files:
        filename = os.path.basename(file).split('.')[0]
        target_location = os.path.join(output, filename)
        target_image = os.path.join(target_location,os.path.basename(file))
        print(f"filename: {filename}")
        print(f"Target location: {target_location}")

        if not os.path.exists(target_location):
            os.makedirs(target_location)
        
        process_image(file, target_image, should_outline, num_clusters)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-image', type=str, help='Path of input image.')
    parser.add_argument('-o', '--output-image', type=str, help='Path of output image.')
    parser.add_argument('-k', '--num-of-clusters', type=int, default=15, help='Number of kmeans clusters for dominant color calculation. Defaults to 15.')
    parser.add_argument('--outline', action="store_true", help='Save outline image containing edges.')
    parser.add_argument('-bd', '--batch-dir', type=str, help='Path of directory for batch processing')
    args = parser.parse_args()

    if args.batch_dir and os.path.exists(args.batch_dir):
        process_batch(args.batch_dir, args.output_image, args.outline, args.num_of_clusters)
    elif args.input_image and args.output_image:
        process_image(args.input_image, args.output_image, args.outline, args.num_of_clusters)
    else:
        print("Invalid arguments. Please specify input and output paths or batch directory path.")

if __name__ == '__main__':
    main()