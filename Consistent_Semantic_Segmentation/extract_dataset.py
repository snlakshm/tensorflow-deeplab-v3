import os
import sys
import h5py
import argparse
import numpy as np
from scipy import io
from PIL import Image

def color_map(N=256, normalized=False):
	'''
	Get the color maps for the segmentation task.
	This is the PYTHON implementation of the VOC cmap extractor.
	'''
	def bitget(byteval, idx):
		return ((byteval & (1 << idx)) != 0)

	dtype = 'float32' if normalized else 'uint8'
	cmap = np.zeros((N, 3), dtype=dtype)
	for i in range(N):
		r = g = b = 0
		c = i
		for j in range(8):
			r = r | (bitget(c, 0) << 7-j)
			g = g | (bitget(c, 1) << 7-j)
			b = b | (bitget(c, 2) << 7-j)
			c = c >> 3

		cmap[i] = np.array([r, g, b])

	cmap = cmap/255 if normalized else cmap
	return cmap

def decode_labels(label_mask, cmap, n, class_mapper):
	"""Decode segmentation class labels into a color image
		Args:
			label_mask (np.ndarray): an (W,H) array of integer values denoting
			the class label at each spatial location.
			plot (bool, optional): whether to show the resulting color image
			in a figure.
		Returns:
			(np.ndarray, optional): the resulting decoded color image.
	"""

	label_mask = label_mask.astype(int)
	r = label_mask.copy()
	g = label_mask.copy()
	b = label_mask.copy()
	for ll in range(0, n+1):
		if ll==0:
			convert_ll = 0
		else:
			convert_ll = class_mapper[ll-1]
			
		r[label_mask == ll] = cmap[convert_ll, 0]
		g[label_mask == ll] = cmap[convert_ll, 1]
		b[label_mask == ll] = cmap[convert_ll, 2]
	rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
	rgb[:, :, 0] = r
	rgb[:, :, 1] = g
	rgb[:, :, 2] = b
	return rgb

def extract_dataset(args):
	'''
	Extract the images and the masks from the dataset Mat file.
	Convert the classes for each pixel in the labels to the mask color scheme of VOC.
	Store all the files in the directories specified by the user.
	'''
	hf = h5py.File(args.mat_file, 'r')

	images = np.asarray(hf['images'])
	labels = np.asarray(hf['labels'])
	depths = np.asarray(hf['depths'])
	names_encoded = np.asarray(hf['names'])
	names_encoded = names_encoded[0]

	segmentation_mapping = io.loadmat(args.class_mapping)
	class_mapper = segmentation_mapping['mapClass'][0]

	min_depth = np.min(depths)
	max_depth = np.max(depths)

	cmap = color_map(14, False)
	# for i in range(cmap.shape[0]):
	# 	print(cmap[i,:])

	# Create target Directory if don't exist
        if not os.path.exists(args.root_dir):
                os.mkdir(args.root_dir)
	path = os.path.join(args.root_dir, args.image_dir)
	if not os.path.exists(path):
		os.mkdir(path)
		print("Directory " , path ,  " Created ")
	else:    
		print("Directory " , path ,  " already exists")

	path = os.path.join(args.root_dir, args.label_dir)
	if not os.path.exists(path):
		os.mkdir(path)
		print("Directory " , path ,  " Created ")
	else:    
		print("Directory " , path ,  " already exists")

	path = os.path.join(args.root_dir, args.depth_dir)
	if not os.path.exists(path):
		os.mkdir(path)
		print("Directory " , path ,  " Created ")
	else:    
		print("Directory " , path ,  " already exists")

	for i in range(images.shape[0]):
	#for i in range(10):
		# Save the image
		image = images[i,:,:,:]
		image = image.transpose([2,1,0])
		im = Image.fromarray(image)
		image_path = os.path.join(args.root_dir, args.image_dir, str(i+1).zfill(4) + '.jpg')
		im.save(image_path)
		# Encode the mask and then save it.
		label = labels[i,:,:]
		label = label.transpose([1,0])
		mask = decode_labels(label, cmap, names_encoded.shape[0], class_mapper)
		mask_im = Image.fromarray(mask.astype(np.uint8))
		mask_path = os.path.join(args.root_dir, args.label_dir, str(i+1).zfill(4) + '.jpg')
		mask_im.save(mask_path, subsampling=0, quality=100)
		# Normalize the depth maps and then save it. 
		depth = depths[i,:,:]
		depth = depth.transpose([1,0])
		depth = (depth - min_depth) / (max_depth - min_depth)
		depth = depth*255
		depth_im = Image.fromarray(depth.astype(np.uint8))
		depth_path = os.path.join(args.root_dir, args.depth_dir, str(i+1).zfill(4) + '.jpg')
		depth_im.save(depth_path)
		print(str(i+1).zfill(4) + "/" + str(images.shape[0]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="config")
	parser.add_argument(
		'--mat_file', 
		nargs='?', 
		type=str, 
		default='nyu_depth_v2_labeled.mat', 
		help='The data file to read from.'
	)
	parser.add_argument('--root_dir', nargs='?', type=str, default='NYU/',    
                        help='The data root directory..')
	parser.add_argument('--image_dir', nargs='?', type=str, default='images/',    
                        help='The directory to store the images in.')
	parser.add_argument('--label_dir', nargs='?', type=str, default='labels/',    
                        help='The directory to store the labels in.')
	parser.add_argument('--depth_dir', nargs='?', type=str, default='depth/',    
                        help='The directory to store the depth maps in.')
	parser.add_argument('--class_mapping', nargs='?', type=str, default='classMapping13.mat',    
                        help='The file that has the segmentation class mapping.')

	args = parser.parse_args()
	extract_dataset(args)
