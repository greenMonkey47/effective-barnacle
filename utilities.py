import cvxopt as cvxopt
from cvxopt import spmatrix
import scipy as scipy
import scipy.sparse
import cvxpy as cvx
import pandas as pd
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from itertools import combinations
import pysubs2
from configparser import ConfigParser

import editing
import os,sys
# def optimised_cropping_window(frame,incl_region, vlambda1=1000, vlambda2=1, vlambda3=1000):
def optimiser(frame,incl_region, vlambda1=1000, vlambda2=1, vlambda3=1000):

	ar = 1.7779

	incl_x1 = incl_region[:,0]
	incl_y1 = incl_region[:,1]
	incl_x2 = incl_region[:,2]
	incl_y2 = incl_region[:,3]

	frame_x1 = frame[:,0]
	frame_y1 = frame[:,1]
	frame_x2 = frame[:,2]
	frame_y2 = frame[:,3]

	frame_x = 0.5*(frame_x1+frame_x2)
	frame_y = 0.5*(frame_y1+frame_y2)
	frame_s = 0.5*(frame_y2-frame_y1)

	n = frame_x.size

	e = np.mat(np.ones((1, n)))
	D1 = scipy.sparse.spdiags(np.vstack((-e,e)), range(2), n-1, n)
	D1 = D1.tocoo()
	D2 = scipy.sparse.spdiags(np.vstack((e,-2*e,e)), range(3), n-2, n)
	D2 = D2.tocoo()
	D3 = scipy.sparse.spdiags(np.vstack((-e,3*e,-3*e,e)), range(4), n-3, n)
	D3 = D3.tocoo()

	fx = cvx.Variable(n)
	fy = cvx.Variable(n)
	fs = cvx.Variable(n)

	constraints = []

	# for i in xrange(n):
	#	 constraints += [
	#		 fx[i] - ar*fs[i] >  0,
	#		 fx[i] - ar*fs[i] <= incl_x1[i],
	#		 fx[i] + ar*fs[i] <  1920,
	#		 fx[i] + ar*fs[i] >= incl_x2[i],
	#		 fy[i] - fs[i]	>  0,
	#		 fy[i] - fs[i]	<= incl_y1[i],
	#		 fy[i] + fs[i]	<  1080,
	#		 fy[i] + fs[i]	>= incl_y2[i],
	#	 ]
	constraints += [
		fx - ar*fs >  0,
		# fx - ar*fs <= incl_x1,
		fx + ar*fs <  1920,
		# fx + ar*fs >= incl_x2,
		fy - fs	>  0,
		# fy - fs	<= incl_y1,
		fy + fs	<  1080,
		# fy + fs	>= incl_y2,
	]

	obj = cvx.Minimize(0.5*( 
						  cvx.sum_squares(fx - frame_x)
						+ cvx.sum_squares(fy - frame_y)
						+ cvx.sum_squares(fs - frame_s))
						+ vlambda1*cvx.norm(D1*fx,1) 
						+ vlambda2*cvx.norm(D2*fx,1) 
						+ vlambda3*cvx.norm(D3*fx,1)
						+ vlambda1*cvx.norm(D1*fy,1) 
						+ vlambda2*cvx.norm(D2*fy,1) 
						+ vlambda3*cvx.norm(D3*fy,1)
						+ vlambda1*cvx.norm(D1*fs,1) 
						+ vlambda2*cvx.norm(D2*fs,1) 
						+ vlambda3*cvx.norm(D3*fs,1)
						)

	prob = cvx.Problem(obj,constraints)
	# prob.solve(max_iters=10000, verbose=True)
	prob.solve()
	# prob.solve(max_iters=10000)

	# print('Solver status: ', prob.status)
	# if (prob.status != cvx.OPTIMAL or prob.status != cvx.OPTIMAL_INACCURATE):
		# raise Exception("Solver did not converge!")
		# print("didnt converge excactly")


	# END OPTIMISATION

	fx = np.asarray(fx.value)
	fy = np.asarray(fy.value)
	fs = np.asarray(fs.value)

	opt_x1 = fx - ar*fs
	opt_y1 = fy - fs
	opt_x2 = fx + ar*fs
	opt_y2 = fy + fs

	opt_region = np.column_stack((opt_x1,opt_y1,opt_x2,opt_y2))
	return opt_region
	pas
#RETURNS OPTIMISED CROPPING WINDOW FOR AN ACTOR WITH ABSEN
def optimised_cropping_window(frame,incl_region, vlambda1=1000, vlambda2=1, vlambda3=1000):

	splt_idx, segments = get_actor_segments(incl_region)

	frame_fragments = []
	for i in range(1,len(splt_idx)+1):
		if i!=len(splt_idx):
			temp = np.array(frame[splt_idx[i-1]:splt_idx[i]])
			frame_fragments.append(temp)
		else:
			temp = np.array(frame[splt_idx[i-1]:len(frame)])
			frame_fragments.append(temp)

	frame_fragments = np.asarray(frame_fragments) #SYNCED FRAME FRAGMENTS WITH SEGEMENTED INCLUSION REGION

	opt = []
	for idx,segment in enumerate(segments):
		if(is_zero_segment(segment)):
			opt.append(segment)
		else:
			# t_opt = optimised_cropping_window(frame_fragments[idx],segment,vlambda1,vlambda2,vlambda3) #old name
			t_opt = optimiser(frame_fragments[idx],segment,vlambda1,vlambda2,vlambda3)
			opt.append(t_opt)
	opt = np.asarray(opt)
	opt = np.concatenate((opt), axis=0)
	print("Done")
	return opt

def play(videoName,rectangle,index=5,start_from_frame=0):

	center_x = 0.5*(rectangle[:,0]+rectangle[:,2])
	center_y = 0.5*(rectangle[:,1]+rectangle[:,3])

	cap = cv2.VideoCapture(videoName)
	cap.set(1,start_from_frame)
	if(start_from_frame!=0):
		index=start_from_frame
	# index=1

	while True:
		ret, orig_frame = cap.read()
		index+=1

		cv2.rectangle(orig_frame,(int(rectangle[index][0]),int(rectangle[index][1])),
								(int(rectangle[index][2]),int(rectangle[index][3])),
								(0,0,255),2)

		cv2.circle(orig_frame, (int(center_x[index]),int(center_y[index])), 1 , 0x111, thickness=5, lineType=8, shift=0)
		frame_text = 'Frame : '+str(index)
		frame = cv2.resize(orig_frame,(1366,768))
		cv2.putText(frame,frame_text,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0))
		# frame = cv2.resize(orig_frame,(1920,1080))
		cv2.imshow(videoName,frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		pass
	cap.release()
	cv2.destroyAllWindows()
	pass

def get_rectangle(track,sep=' '):
	cw= pd.read_csv(track, sep)
	cw.columns = ['x1','y1','x2','y2']
	x1,y1,x2,y2 = cw['x1'].values , cw['y1'].values,cw['x2'].values,cw['y2'].values
	rectangle = np.column_stack((x1,y1,x2,y2))
	return rectangle
	pass

def plot_graph(given, calculated):
	t = np.arange(0,len(given))
	plt.figure(figsize=(6, 6))
	plt.plot(t, given, 'k:', linewidth=1.0)
	plt.plot(t, calculated, 'b-', linewidth=2.0)
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.show()

def cropped_play(videoName,rectangle):

	##temporary
	# fourcc = cv2.VideoWriter_fourcc(*'DIVX')  # 'x264' doesn't work
	# out = cv2.VideoWriter('output.mp4', fourcc, 25.0, (1423,800))

	cap = cv2.VideoCapture(videoName)
	index=5

	while True:
		ret, orig_frame = cap.read()
		index+=1

		# cv2.rectangle(orig_frame,(int(rectangle[index][1]),int(rectangle[index][3])),
								# (int(rectangle[index][0]),int(rectangle[index][2])),
								# (0,0,255),2)
		# cropped_frame = orig_frame[0:1920,0:1080]
		if(rectangle[index][0]==0 and rectangle[index][2]==0 or rectangle[index][1]==0 and rectangle[index][3]==0 ):
			cropped_frame = orig_frame[0:1,0:1]
		else:
			cropped_frame = orig_frame[int(rectangle[index][1]):int(rectangle[index][3]),int(rectangle[index][0]):int(rectangle[index][2])]
		cropped_frame = cv2.resize(cropped_frame, (int(1.7779*720),720))
		cv2.imshow(videoName,cropped_frame)

		#temp
		# out.write(cropped_frame)

		if cv2.waitKey(25) & 0xFF == ord('q'):
			break
		pass
	cap.release()
	# out.release()
	cv2.destroyAllWindows()
	pass

def get_inclusion_region(videoName, actor, shot_composition):
	track = 'inclusion_regions/'+videoName+'/'+videoName+'-'+actor+'-'+shot_composition+'-incl.txt'
	rect = get_rectangle(track)
	return rect
	pas

def get_correct_frame(incl_region,aspect_ratio,width_margin=0.5):

	incl_x1 = incl_region[:,0]
	incl_y1 = incl_region[:,1]
	incl_x2 = incl_region[:,2]
	incl_y2 = incl_region[:,3]	

	incl_width = incl_x2 - incl_x1
	incl_height = incl_y2 - incl_y1

	w_mask = incl_width<incl_height*aspect_ratio
	h_mask = incl_height<incl_width/aspect_ratio

	w = []
	h = []
	for i in range(len(w_mask)):
		if(w_mask[i]):
			w.append(incl_height[i]*aspect_ratio)
		else:
			w.append(incl_width[i])
		pass
	for i in range(len(h_mask)):
		if(h_mask[i]):
			h.append(incl_width[i]/aspect_ratio)
			pass
		else:
			h.append(incl_height[i])
		pass

	w = np.asarray(w)
	h = np.asarray(h)

	incl_x = 0.5*(incl_x1+incl_x2)
	incl_y = 0.5*(incl_y1+incl_y2)
	# incl_s = 0.5*(incl_y2-incl_y1)

	rx1 = incl_x - width_margin*w
	ry1 = incl_y - (width_margin*w/aspect_ratio)

	rx2 = incl_x + width_margin*w

	ry2 = incl_y + (width_margin*w/aspect_ratio)

	r = np.column_stack((rx1,ry1,rx2,ry2))
	return r
	pas
# BAD FUNCTION DESIGN, BUT WORKING
# def get_original_track(videoName,actor):
def get_original_track(video,actor):
	# track = '../Other Resources/datasets/theatre videos/tracks/'+video.name+'/'+video.name+'-'+actor+'.txt'
	# track = '../Other Resources/datasets/theatre videos/tracks/'+videoName+'/'+videoName+'-'+actor+'.txt'
	track = video.track_path+actor+'.txt'
	# rect = get_rectangle(track)
	rect = get_rectangle(track,',')
	return rect
	pass

def generate_one_shot_rushes(video):
	shots = ['MS','FS']
	for actor in video.actors:
		for shot in shots:
			print("Solving for "+video.name+"-"+actor+"-"+shot+"...")
			incl_region = get_inclusion_region(video.name,actor,shot)
			frame = get_correct_frame(incl_region,1.7779)
			opt_region = optimised_cropping_window(frame,incl_region,1000,10,1000)
			np.savetxt('shots/'+video.name+'/'+video.name+'-'+actor+'-'+shot+'.txt',opt_region,delimiter=' ')
	pass

def generate_one_shot_incl(video):
	shots = ['MS','FS']
	for shot in shots:
		for actor in video.actors:
			# rect = get_original_track(video.name,actor)
			rect = get_original_track(video,actor)
			bx1, by1, bx2, by2 = rect[:,0],rect[:,1],rect[:,2],rect[:,3]
			incl_x1 = 0.98*bx1
			incl_y1 = 0.98*by1
			incl_x2 = 1.02*bx2

			bounding_box_size = by2-by1

			if(shot=='MS'):
				# incl_y2 =1.1*by2
				incl_y2 =by2 + 0.8*bounding_box_size
				incl_y2[incl_y2<25] = 0
			else:
				incl_y2[incl_y2<25] = 0
				incl_y2[incl_y2>1] =800
			incl_region = np.column_stack((incl_x1,incl_y1,incl_x2,incl_y2))

			incl_region[incl_region<25] = 0
			np.savetxt('inclusion_regions/'+video.name+'/'+video.name+'-'+actor+'-'+shot+'-incl.txt',incl_region,delimiter=' ')
	pass

def generate_n_shot_incl(video):
	shots = ['MS','FS']
	for shot in shots:
		for i in range(2,len(video.actors)+1):
			combos = combinations(video.actors,i)
			for combo in combos:
				incl_regions = []
				name=""
				# incl_size = 1000000000000
				for actor in combo:
					actor_incl = get_inclusion_region(video.name,actor,shot)
					incl_regions.append(actor_incl)
					name+=actor+'-'
				incl_regions = np.asarray(incl_regions)
				incl_x1 = minimum(*incl_regions[:,:,0])
				incl_y1 = minimum(*incl_regions[:,:,1])
				incl_x2 = maximum(*incl_regions[:,:,2])
				incl_y2 = maximum(*incl_regions[:,:,3])
				incl_region = np.column_stack((incl_x1,incl_y1,incl_x2,incl_y2))
				incl_region = modify_for_absent_actor(incl_region)
				np.savetxt('inclusion_regions/'+video.name+'/'+video.name+'-'+name+shot+'-incl.txt',incl_region,delimiter=' ')
	pass
# if an actor is absent, in inclusion region min x1 and min y1 will become zero, so that multi-shot doesnt exit, correcting th
def modify_for_absent_actor(incl):
	for i in range(incl.shape[0]):
		if(incl[i][0]<=25 and incl[i][1]<=25):
			incl[i][2]=0
			incl[i][3]=0
	return incl

def generate_n_shot_rushes(video):
	shots = ['MS','FS']
	for shot in shots:
		for i in range(2,len(video.actors)+1):
			combos = combinations(video.actors,i)
			for combo in combos:
				file_name = ''
				for actor in combo:
					file_name+=actor+'-'
				file_name =  file_name[:-1]
				print("Solving for "+video.name+"-"+file_name+"-"+shot+"...")
				incl_region = get_inclusion_region(video.name,file_name,shot)
				frame = get_correct_frame(incl_region,1.7779)
				opt_region = optimised_cropping_window(frame,incl_region,600,10,600)
				np.savetxt('shots/'+video.name+'/'+video.name+'-'+file_name+'-'+shot+'.txt',opt_region,delimiter=' ')

def minimum(*arg): 
	res = []
	for a in arg:
		if(len(res)==0):
			res = a
		else:
			res = np.column_stack((res,a))
	r = np.amin(res,axis=1)
	return r
	pas

def maximum(*arg): 
	res = []
	for a in arg:
		if(len(res)==0):
			res = a
		else:
			res = np.column_stack((res,a))
	r = np.amax(res,axis=1)
	return r
	pass
# SEPERATES THE PARTS WHERE ACTOR IS PRESENT AND ABSENT
def get_actor_segments(track):
			
	segments = []
	seg = []
	split_index = [0]
	flag=3
	for idx,record in enumerate(track):
		if ((record[0]<=25 and record[2]<=25) or (record[1]<=25 and record[3]<=25)):
			if flag==1:
				split_index.append(idx)
				segments.append(np.asarray(seg))
				seg = []
			seg.append(record)
			flag=0
			pass
		else:
			if flag==0:
				split_index.append(idx)
				segments.append(np.asarray(seg))
				seg = []
				flag=1
			seg.append(record)
			flag=1
	segments.append(np.asarray(seg))
	return [np.asarray(split_index),np.asarray(segments)]
	pass
	
def is_zero_segment(segment):
	if ((segment[0][0]==0 and segment[0][2]==0) or (segment[0][1]==0 and segment[0][3]==0)):
		return True
	return False
# FRAMES GENERATOR IN A VID
def frame_generator(videoName, dest):
	vidcap = cv2.VideoCapture(videoName)
	success,image = vidcap.read()
	count = 0
	while success:
		file_name = ""
		file_name += str(count).zfill(4)+'.jpg'
		file = dest+file_name
		cv2.imwrite(file, image)	 # save frame as JPEG file	  
		success,image = vidcap.read()
		print('written file :  ', file)
		count += 1

def reduce_resolution(resolution, frame_rate, output_name, dest):
	pass

def loadConfig(CONFIG_FILE): # need to see how we can re-use the editing .py function 

	config_reader = ConfigParser()
	config_reader.read(CONFIG_FILE)

	shots = eval(config_reader.get('performers', 'shots'))
	actors = config_reader.get('performers', 'actors')
	subtitle_file = config_reader.get('video', 'subtitle')
	subtitle = pysubs2.load(subtitle_file)

	videoName = config_reader.get('video', 'name')
	video_frames = {videoName: config_reader.get('video', 'frames')}

	shot_keys = list(shots.keys())
	shot_tracks = {}



	frames = [file
			  for file in os.listdir(video_frames[videoName])
			  if file.endswith('.jpg')]
	frames.sort()

	for shot in shot_keys:
		shot_tracks[shot] = get_rectangle(shots[shot])

	return shot_tracks,shot_keys,subtitle,frames

def createBaseline(CONFIG_FILE):

	shot_tracks,shot_keys,subtitle,frames = loadConfig(CONFIG_FILE)

	fps=30 #have to change this probably add this in the config file


	actors = []
	
	one_shots = [shot for shot in shot_keys if shot.count('-')==1]
	main_shot = [shot for shot in shot_keys if ( shot.count('-')==len(one_shots) and 'FS' in shot)]

	final_track = shot_tracks[main_shot[0]]

	print(main_shot)

	for i in subtitle:
		#print(pysubs2.time.ms_to_frames(i.start,25))
		#print(i.name,one_shots)
		actor_id = [j for j,st in enumerate(one_shots) if i.name.lower() in st]

		start_frame = pysubs2.time.ms_to_frames(i.start,fps)
		end_frame = pysubs2.time.ms_to_frames(i.end,fps)
		print(start_frame,end_frame)
		final_track[start_frame:end_frame,:] = shot_tracks[one_shots[actor_id[0]]][start_frame:end_frame,:]

	op_resolution_w = 1920
	op_resolution_h = 1080

	cropped_window = final_track
	input_video = 'data/videos/music_intro.webm'
	output_video_name = 'temp'

	framerate = 30
	#framerate = 24
	# framerate = 59.94

	output_video_name = output_video_name+'.mp4'

	fourcc = cv2.VideoWriter_fourcc(*'MPEG')  # 'x264' doesn't work
	out = cv2.VideoWriter('temp.mp4', fourcc, framerate, (1920, 1080))
	cap = cv2.VideoCapture(input_video)
	index = 0

	no_of_frames = len(frames)

	#print(cropped_window)
	#print(no_of_frames)
	print('Rendering Video...')

	while index in range(no_of_frames-4):
		ret, orig_frame = cap.read()
		index += 1

		if(cropped_window[index][0] == 0 and
			cropped_window[index][2] == 0 or
			cropped_window[index][1] == 0 and
				cropped_window[index][3] == 0):
			cropped_frame = orig_frame[0:1, 0:1]
		else:
			#print(cropped_window[index])

			cropped_frame = orig_frame[int(cropped_window[index][1]):
										int(cropped_window[index][3]),
										int(cropped_window[index][0]):
										int(cropped_window[index][2])]
			#print(orig_frame.shape)
		# cropped_frame = cv2.resize(cropped_frame, (int(1.7779*720),720))
		cropped_frame = cv2.resize(cropped_frame,
								   (op_resolution_w, op_resolution_h))
		out.write(cropped_frame)

		sys.stdout.write('\r')
		percentage = float(index/no_of_frames)*100
		sys.stdout.write(str('%0.2f' % percentage))

	cap.release()
	out.release()	

def main():
	pass

if __name__ == '__main__':
	
	CONFIG_FILE = 'configs/music_intro.ini'

	createBaseline(CONFIG_FILE)