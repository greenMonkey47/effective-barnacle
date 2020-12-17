import os
import sys
import math
import yaml
import cv2 as cv2
import numpy as np
import time
import datetime
import pandas as pd
from configparser import ConfigParser,ExtendedInterpolation

import utilities as ut

from Editor import Editor


def distance(p0, p1):
	return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)


def play(e,video_path,rectangle,gaze_tracks,final_track):

	vidcap = cv2.VideoCapture(video_path)
	index = 0
	while True:
		ret,orig_frame = vidcap.read()
		cv2.rectangle(orig_frame,
					(int(rectangle[index][0]),
					int(rectangle[index][1])),
					(int(rectangle[index][2]), int(rectangle[index][3])),
					(0, 0, 255),
					2)
		for p in gaze_tracks:
			gaze_point = (int(float(gaze_tracks[p][index][0]) *
							  e.normFactorX +
							  float(e.gazeXOffset)),
						  int(float(gaze_tracks[p][index][1]) *
							  e.normFactorY +
							  float(e.gazeYOffset)))
			cv2.circle(orig_frame,
					   gaze_point,
					   color=(0, 255, 0),
					   radius=5,
					   thickness=6)

		frame = cv2.resize(orig_frame,(1066,600))
		frame_text = 'Frame : '+str(index)
		shot_text = 'Shot : '+final_track[index]
		cv2.putText(frame, frame_text,
					(50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.putText(frame, shot_text,
					(50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
		cv2.imshow('mat',frame)
		if(cv2.waitKey(int(1000/e.fps))&0xff==ord('q')):
			vidcap.release()
			break
			cv2.destroyAllWindows()
		index +=1



def computeRythmCost(editor,previous_frame,prev_shot,shot_duration):
	time = shot_duration[previous_frame][prev_shot]
	# time = shot_duration[pres_shot][previous_frame][prev_shot]
	return analyticRythmCost(editor,time)
	

def analyticRythmCost(editor,time,variance=0.82,rythm_factor=3):
	if(time==0):
		return 1
	y = np.exp((-(np.log(time)-np.log(editor.fps*rythm_factor))**2/2*(variance**2)))
	rythm_cost = -1*y + 1
	return rythm_cost
	


def computeGazeCostforFrame(editor,shot_tracks,gaze_tracks,frame_number):
	gaze_cost = {key:0 for key in editor.shot_keys}
	
	## GAZE FOR ONE SHOTS
	one_shots = editor.getAllNShots(1)

	for shot in one_shots:
		centerX = 0.5*(shot_tracks[shot][frame_number][0] + \
			shot_tracks[shot][frame_number][2])
		centerY = 0.5*(shot_tracks[shot][frame_number][1] + \
			shot_tracks[shot][frame_number][3])

		center_point = [centerX,centerY]
		cost_for_present_shot = 0

		if(not (centerX < 25 or centerY < 25)):
			for track_number in gaze_tracks:
			
				single_gaze_track = gaze_tracks[track_number]
				gaze_point = [float(single_gaze_track[frame_number][0]) *
								editor.normFactorX+float(editor.gazeXOffset),
								float(single_gaze_track[frame_number][1]) *
								editor.normFactorY+float(editor.gazeYOffset)]

				if(int(gaze_point[0]) > editor.width or int(gaze_point[1]) > editor.height):
					continue
				cost_for_present_shot += distance(center_point, gaze_point)

			if(cost_for_present_shot != 0):
				gaze_cost[shot] = 1/float(cost_for_present_shot)
			else:
				gaze_cost[shot] = 0
		else:
			gaze_cost[shot] = 0

	# for n-shot, the unary cost is defined as follows
	for i in range(2, len(editor.actors)+1):
		n_shots = editor.getAllNShots(i)
		cost_for_present_shot = 0

		for n_shot in n_shots:
			lower_n_shots = editor.getAllNShots(i-1)
			lower_n_shots = [s
							for s in lower_n_shots
							if editor.containedActors(s, n_shot)]

			lower_n_shots = sorted(lower_n_shots,
									key=lambda x: shot_tracks[x][frame_number][0])

			cost_for_present_shot = (gaze_cost[lower_n_shots[0]] +
									gaze_cost[lower_n_shots[-1]] -
									abs(gaze_cost[lower_n_shots[0]] -
										gaze_cost[lower_n_shots[-1]]))

		gaze_cost[n_shot] = cost_for_present_shot

	return gaze_cost
def addDynamicCosts(editor,dp_cost,gaze_cost,shot_tracks,present_frame,backtrack,shot_duration):
	# present_frame is to be computed
	# previous_frame has been computed
	print(present_frame)
	previous_frame = present_frame-1
	if(present_frame==0):
		for key in editor.shot_keys:
			dp_cost[present_frame][key] = gaze_cost[present_frame][key]
			backtrack[present_frame][key] = key
			for k in editor.shot_keys:
				# shot_duration[key][present_frame][key] = 1
				shot_duration[present_frame][key] = 1
	else:
		for pres_shot in editor.shot_keys:
			min_cost = np.inf
			update_cost = {key:0 for key in editor.shot_keys}
			for prev_shot in editor.shot_keys:
				#gaze
				update_cost[prev_shot] = dp_cost[previous_frame][prev_shot]+gaze_cost[present_frame][pres_shot]

				#rythm
				update_cost[prev_shot] += computeRythmCost(editor,previous_frame,prev_shot,shot_duration)

				# print(prev_shot,pres_shot,
				# 	'gaze: ',gaze_cost[present_frame][pres_shot],
				# 	'rythm: ',computeRythmCost(editor,previous_frame,prev_shot,shot_duration)
				# 		,'dp: ',update_cost[prev_shot])

				#transition cost
				# update_cost[prev_shot] += transition_cost_matrix[prev_shot][prev_shot]
					
				if(update_cost[prev_shot]<min_cost):
					min_cost = update_cost[prev_shot]
					backtrack[present_frame][pres_shot] = prev_shot

			if(backtrack[present_frame][pres_shot] == pres_shot):
				# increase timer for a shot
				shot_duration[present_frame][pres_shot] = shot_duration[previous_frame][pres_shot] + 1
			else:
				if(shot_duration[previous_frame][pres_shot] > 0):
					# reset timer on shot change
					shot_duration[present_frame][pres_shot] = 0

			dp_cost[present_frame][pres_shot] += min_cost

	return dp_cost,backtrack,shot_duration

def main():

	e = Editor('configs/dos6_all.ini')

	gaze_cost = [{key:0 for key in e.shot_keys} for i in range(0,e.no_of_frames)]
	dp_cost = [{key:0 for key in e.shot_keys} for i in range(0,e.no_of_frames)]
	backtrack = [{key: '' for key in e.shot_keys} for i in range(0,e.no_of_frames)]
	shot_duration = [{key: 0 for key in e.shot_keys} for i in range(0, e.no_of_frames)]

	final_track = ['']*e.no_of_frames

	frames_in_range = e.no_of_frames

	for frame in range(0,frames_in_range):
		g = computeGazeCostforFrame(e,e.shot_tracks,e.gaze_tracks,frame)	
		for key in e.shot_keys:
			gaze_cost[frame][key] = g[key]

	max_gaze = np.max([np.max([gaze_cost[frame][key] for key in e.shot_keys]) for frame in range(0,e.no_of_frames)])
	gaze_cost = [{key:max_gaze-gaze_cost[frame][key] for key in e.shot_keys} for frame in range(0,e.no_of_frames)]
	max_gaze = np.max([np.max([gaze_cost[frame][key] for key in e.shot_keys]) for frame in range(0, e.no_of_frames)])

	gaze_cost = [{key:(gaze_cost[frame][key]/max_gaze) for key in e.shot_keys} for frame in range(0,e.no_of_frames)]

	transition_cost_matrix = {k:{key:0.1 for key in e.shot_keys}for k in e.shot_keys}


	## ADDING RYTHM COST
	for frame in range(0,10):#frames_in_range):
		dp_cost, backtrack,shot_duration = addDynamicCosts(e,dp_cost,gaze_cost,e.shot_tracks,frame,backtrack,shot_duration)
		print(dp_cost[frame],backtrack[frame],shot_duration[frame])

	exit()
	last_frame_shot = e.shot_keys[np.argmin([dp_cost[e.no_of_frames-1][key] for key in e.shot_keys])]
	final_track[frames_in_range-1] = last_frame_shot

	#print(backtrack)
	for frame in range(frames_in_range-2,-1, -1):
		final_track[frame] = backtrack[frame][final_track[frame+1]]

	#print(final_track)

	cropped_window = [e.shot_tracks[shot][frame] for frame,shot in enumerate(final_track)]

	videoName = e.videoName
	video_path = e.video[videoName]

	play(e,video_path,cropped_window,e.gaze_tracks,final_track)

if __name__ == '__main__':
	main()