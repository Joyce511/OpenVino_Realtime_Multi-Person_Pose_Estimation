from __future__ import print_function
import sys
import os
from argparse import ArgumentParser
import cv2 as cv
import time

from openvino.inference_engine import IENetwork, IEPlugin

import numpy as np
import scipy
import PIL.Image
import math
import caffe
from config_reader import config_reader
import util
import copy
import matplotlib
import pylab as plt
from numpy import ma
from scipy.ndimage.filters import gaussian_filter

def main():

	param, model = config_reader()
	model_xml = "/home/yue/Realtime_Multi-Person_Pose_Estimation/model/_trained_COCO/pose_iter_440000.xml"
	model_bin = os.path.splitext(model_xml)[0] + ".bin"
	prob_threshold = 0.5
	labels_map = None

	print("Initializing plugin for CPU device...")
	plugin = IEPlugin(device="CPU", plugin_dirs=None)

	print("Adding CPU extenstions...")
	plugin.add_cpu_extension("/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/libcpu_extension_sse4.so")

	print("Reading IR...")
	net = IENetwork.from_ir(model=model_xml, weights=model_bin)
	input_blob = next(iter(net.inputs))
	out_blob = next(iter(net.outputs))

	print("Loading IR to the plugin...")
	exec_net = plugin.load(network=net, num_requests=2)
	# Read and pre-process input image
	n, c, h, w = net.inputs[input_blob]
	print((n,c,h,w))

	input_stream = "../sample_video/dance.mp4"
	cap = cv.VideoCapture(input_stream)
	cur_request_id = 0
	next_request_id = 1

	if (cap.isOpened()== False): 
	  print("Error opening video stream or file")
		
	print("Starting inference in async mode...")
	is_async_mode = True
	render_time = 0


	# find connection in the specified sequence, center 29 is in the position 15
	limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
			   [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
			   [1,16], [16,18], [3,17], [6,18]]
	# the middle joints heatmap correpondence
	mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
			  [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
			  [55,56], [37,38], [45,46]]
	colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
			  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
			  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
	fps = 0
	
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		
		initial_w = cap.get(3)
		initial_h = cap.get(4)
		
		image = cv.resize(frame, (w, h))

		image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
		image = image.reshape((1,3,368,368))/256 - 0.5
		img_show = frame;
		# Main sync point:
		# in the truly Async mode we start the NEXT infer request, while waiting for the CURRENT to complete
		# in the regular mode we start the CURRENT request and immediately wait for it's completion
		inf_start = time.time()
		
		res = exec_net.infer(inputs={input_blob: image})
		
		inf_end = time.time()
		det_time = inf_end - inf_start

		# Parse detection results of the current request
		res_heatMap = res['Mconv7_stage6_L2']
		res_paf = res['Mconv7_stage6_L1']

		#Process outputs
		res_heatMap = np.squeeze(res_heatMap, axis=0)
		res_paf = np.squeeze(res_paf, axis=0)

		# extract outputs, resize, and remove padding
		heatmap = np.transpose(res_heatMap, (1,2,0)) # output 1 is heatmaps

		heatmap = cv.resize(heatmap, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_CUBIC)

		paf = np.transpose(res_paf, (1,2,0)) # output 0 is PAFs

		paf = cv.resize(paf, (frame.shape[1], frame.shape[0]), interpolation=cv.INTER_CUBIC)

		heatmap_avg = heatmap 
		paf_avg = paf 


		U = paf_avg[:,:,16] * -1
		V = paf_avg[:,:,17]
		X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
		M = np.zeros(U.shape, dtype='bool')
		M[U**2 + V**2 < 0.5 * 0.5] = True
		U = ma.masked_array(U, mask=M)
		V = ma.masked_array(V, mask=M)

		all_peaks = []
		peak_counter = 0

		for part in range(19-1):
			x_list = []
			y_list = []
			map_ori = heatmap_avg[:,:,part]
			map = gaussian_filter(map_ori, sigma=3)

			map_left = np.zeros(map.shape)
			map_left[1:,:] = map[:-1,:]
			map_right = np.zeros(map.shape)
			map_right[:-1,:] = map[1:,:]
			map_up = np.zeros(map.shape)
			map_up[:,1:] = map[:,:-1]
			map_down = np.zeros(map.shape)
			map_down[:,:-1] = map[:,1:]

			peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
			peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0])) # note reverse
			peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
			id = range(peak_counter, peak_counter + len(peaks))
			peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

			all_peaks.append(peaks_with_score_and_id)
			peak_counter += len(peaks)


		connection_all = []
		special_k = []
		mid_num = 10

		for k in range(len(mapIdx)):
			score_mid = paf_avg[:,:,[x-19 for x in mapIdx[k]]]
			candA = all_peaks[limbSeq[k][0]-1]
			candB = all_peaks[limbSeq[k][1]-1]
			nA = len(candA)
			nB = len(candB)
			indexA, indexB = limbSeq[k]
			if(nA != 0 and nB != 0):
				connection_candidate = []
				for i in range(nA):
					for j in range(nB):
						vec = np.subtract(candB[j][:2], candA[i][:2])
						norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
						vec = np.divide(vec, norm)

						startend = list(zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), \
									   np.linspace(candA[i][1], candB[j][1], num=mid_num)))

						vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] \
										  for I in range(len(startend))])
						vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] \
										  for I in range(len(startend))])
						score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
						score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*frame.shape[0]/norm-1, 0)
						criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
						criterion2 = score_with_dist_prior > 0
						if criterion1 and criterion2:
							connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

				connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
				connection = np.zeros((0,5))
				for c in range(len(connection_candidate)):
					i,j,s = connection_candidate[c][0:3]
					if(i not in connection[:,3] and j not in connection[:,4]):
						connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
						if(len(connection) >= min(nA, nB)):
							break

				connection_all.append(connection)
			else:
				special_k.append(k)
				connection_all.append([])
		# last number in each row is the total parts number of that person
		# the second last number in each row is the score of the overall configuration
		subset = -1 * np.ones((0, 20))
		candidate = np.array([item for sublist in all_peaks for item in sublist])

		for k in range(len(mapIdx)):
			if k not in special_k:
				partAs = connection_all[k][:,0]
				partBs = connection_all[k][:,1]
				indexA, indexB = np.array(limbSeq[k]) - 1

				for i in range(len(connection_all[k])): #= 1:size(temp,1)
					found = 0
					subset_idx = [-1, -1]
					for j in range(len(subset)): #1:size(subset,1):
						if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
							subset_idx[found] = j
							found += 1

					if found == 1:
						j = subset_idx[0]
						if(subset[j][indexB] != partBs[i]):
							subset[j][indexB] = partBs[i]
							subset[j][-1] += 1
							subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
					elif found == 2: # if found 2 and disjoint, merge them
						j1, j2 = subset_idx
						print ("found = 2")
						membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
						if len(np.nonzero(membership == 2)[0]) == 0: #merge
							subset[j1][:-2] += (subset[j2][:-2] + 1)
							subset[j1][-2:] += subset[j2][-2:]
							subset[j1][-2] += connection_all[k][i][2]
							subset = np.delete(subset, j2, 0)
						else: # as like found == 1
							subset[j1][indexB] = partBs[i]
							subset[j1][-1] += 1
							subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

					# if find no partA in the subset, create a new subset
					elif not found and k < 17:
						row = -1 * np.ones(20)
						row[indexA] = partAs[i]
						row[indexB] = partBs[i]
						row[-1] = 2
						row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
						subset = np.vstack([subset, row])
		deleteIdx = [];
		for i in range(len(subset)):
			if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
				deleteIdx.append(i)
		subset = np.delete(subset, deleteIdx, axis=0)


		stickwidth = 2
		cmap = matplotlib.cm.get_cmap('hsv')

		for i in range(18):
			rgba = np.array(cmap(1 - i/18. - 1./36))
			rgba[0:3] *= 255
			for j in range(len(all_peaks[i])):
				cv.circle(img_show, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)

		to_plot = cv.addWeighted(frame, 0.3, img_show, 0.7, 0)

		for i in range(17):
			for n in range(len(subset)):
				index = subset[n][np.array(limbSeq[i])-1]
				if -1 in index:
					continue
				cur_canvas = img_show.copy()
				Y = candidate[index.astype(int), 0]
				X = candidate[index.astype(int), 1]
				mX = np.mean(X)
				mY = np.mean(Y)
				length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
				angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
				polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
				cv.fillConvexPoly(cur_canvas, polygon, colors[i])
				img_show = cv.addWeighted(img_show, 0.4, cur_canvas, 0.6, 0)


		#################################################################################################################
		# Draw performance stats

		inf_time_message = "Inference time: N\A for async mode" if is_async_mode else \
						   "Inference time: {:.3f} ms".format(det_time * 1000)
		render_time_message = "OpenCV rendering time: {:.3f} ms".format(render_time * 1000)
		async_mode_message = "Async mode is on. Processing request {}".format(cur_request_id) if is_async_mode else \
							 "Async mode is off. Processing request {}".format(cur_request_id)
		fps_message = "FPS: {:.1f}".format(fps);

		cv.putText(img_show, inf_time_message, (15, 15), cv.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
		cv.putText(img_show, render_time_message, (15, 30), cv.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
		cv.putText(img_show, fps_message, (15, 45), cv.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)
		
		render_start = time.time()
		cv.imshow("Detection Results", img_show)
		render_end = time.time()
		render_time = render_end - render_start
		fps = float(60/(render_time*1000))
		
		key = cv.waitKey(1)
		if key == 27:
			break
		if (9 == key):
			is_async_mode = not is_async_mode
			print("Switched to {} mode".format("async" if is_async_mode else "sync"))

		if is_async_mode:
			cur_request_id, next_request_id = next_request_id, cur_request_id
			
	cv.destroyAllWindows()
	del exec_net
	del plugin


if __name__ == '__main__':
	sys.exit(main() or 0)