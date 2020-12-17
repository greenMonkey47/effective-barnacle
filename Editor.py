
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

class Editor(object):
	def __init__(self,config_file):

		config_reader = ConfigParser(interpolation=ExtendedInterpolation())
		config_reader.read(config_file)


		self.DEBUG_MODE = config_reader.getint('exec', 'DEBUG_MODE', fallback=0)
		self.VERBOSE = config_reader.getint('exec', 'VERBOSE', fallback=1)
		self.COST_PLOT = config_reader.getint('exec', 'COST_PLOT', fallback=1)

		self.fps = config_reader.getfloat('video','fps',fallback=30)

		self.output_video_name = config_reader.get('exec', 'output_video_name',
											  fallback='output.mp4')
		self.transition_factor = config_reader.getfloat('parameters',
														'transition_factor',
														fallback=1)
		self.high_min_shot_duration_factor = config_reader.getfloat('parameters',
														'high_min_shot_duration_factor',
														fallback=100)
		self.low_min_shot_duration_factor = config_reader.getfloat('parameters',
													   'low_min_shot_duration_factor',
													   fallback=0.01)
		self.low_rythm_factor = config_reader.getfloat('parameters',
											  'low_rythm_factor',
											  fallback=0.01)
		self.high_rythm_factor = config_reader.getfloat('parameters',
											  'high_rythm_factor',
											  fallback=100)
		self.min_shot_duration = int(self.fps*config_reader.getfloat('parameters',
												 'min_shot_duration',
												 fallback=2))
		self.max_shot_duration = int(self.fps*config_reader.getfloat('parameters',
												 'max_shot_duration',
												 fallback=4))

		self.low_overlap_factor = config_reader.getfloat('parameters',
													'low_overlap_factor',
													fallback=0.01)
		self.low_overlap_percentage = config_reader.getfloat('parameters',
														'low_overlap_percentage',
														fallback=0.3)
		self.mid_overlap_factor = config_reader.getfloat('parameters',
													'mid_overlap_factor',
													fallback=1)
		self.mid_overlap_percentage = config_reader.getfloat('parameters',
														'mid_overlap_percentage',
														fallback=0.5)
		self.high_overlap_factor = config_reader.getfloat('parameters',
													 'high_overlap_factor',
													 fallback=100)
		self.low_to_high_shot_context_factor = config_reader.getfloat('parameters',
																 'low_to_high_shot_context_factor',
																 fallback=0.01)
		self.high_to_low_shot_context_factor = config_reader.getfloat('parameters',
																 'high_to_low_shot_context_factor',
																 fallback=100)
		self.shot_composition_factor = config_reader.getfloat('parameters',
														'shot_composition_factor',
														fallback=0.01)
		self.MS_composition_threshold = config_reader.getfloat('parameters','MS_composition_threshold',fallback=0.4)												   
		self.FS_composition_threshold = config_reader.getfloat('parameters','FS_composition_threshold',fallback=0.4)													

		self.establishing_shot_time = int(self.fps*config_reader.getfloat('parameters',
													  'establishing_shot_time',
													  fallback=4))
		self.gazeXOffset = config_reader.getint('parameters',
										   'gazeXOffset',
										   fallback=0)
		self.gazeYOffset = config_reader.getint('parameters',
										   'gazeYOffset',
										   fallback=0)
		# incase the resolution of gaze recording is different from the
		# original videoresolution
		# for dos6 it is 1366x768
		# rest is 1920x1080

		self.normFactorX = config_reader.getint('parameters',
										   'normFactorX',
										   fallback=1920)
		self.normFactorY = config_reader.getint('parameters',
										   'normFactorY',
										   fallback=1080)

		self.videoName = config_reader.get('video', 'name')
		self.basedir = config_reader.get('video', 'basedir')

		self.video = {self.videoName: config_reader.get('video', 'path')}
		self.audio = {self.videoName: config_reader.get('video', 'audio')}
		self.video_frames = {self.videoName: config_reader.get('video', 'frames')}
		self.gaze = {self.videoName: config_reader.get('video', 'gaze')}
		self.width = config_reader.getint('video', 'width')
		self.height = config_reader.getint('video', 'height')

		# shots = eval(config_reader.get('performers', 'shots'))
		self.shots = config_reader['shots']
		self.actors = config_reader.get('performers', 'actors')

		self.shot_keys = list(self.shots.keys())

		self.normFactorX = self.width/self.normFactorX
		self.normFactorY = self.height/self.normFactorY


		self.shot_tracks = {}
		for shot in self.shot_keys:
			self.shot_tracks[shot] = ut.get_rectangle(self.shots[shot])


		with open(self.gaze[self.videoName], 'r') as fp:
			self.gaze_tracks  = yaml.load(fp)
		
		self.frames = [file for file in os.listdir(self.video_frames[self.videoName]) if file.endswith('.jpg')]
		self.frames.sort()

		self.min_shot_length = min([len(list(self.shot_tracks[key]))for key in self.shot_keys])
		self.no_of_frames = min(len(self.frames), self.min_shot_length)


	def findNShot(self,shot):
		return shot.count('-')
	
	def getAllNShots(self,n):
 
		l = list(self.shots.keys())
		l = [shot for shot in self.shot_keys if(self.findNShot(shot) == n)]
		return l

	def actorsInNShot(self,shot):
  
		t = shot.split('-')
		return t[0:-1]


	def containedActors(self,a, b):
   
		a = selactorsInNShot(a)
		b = self.actorsInNShot(b)
		return set(a).issubset(set(b))

