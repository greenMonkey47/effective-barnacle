import os
import sys
import math
import yaml
import cv2 as cv2
import numpy as np
import time
import datetime
import pandas as pd
import utilities as ut

from pathlib import Path
import matplotlib.pyplot as plt
from configparser import ConfigParser,ExtendedInterpolation
from colorama import init, Fore, Style, Back
init()
print(Style.BRIGHT)

# CONFIG_FILE = 'configs/kala.ini'
# CONFIG_FILE = 'configs/dance_all.ini'
# CONFIG_FILE = 'configs/dance2_all.ini'
# CONFIG_FILE = 'configs/dos1_all.ini'
CONFIG_FILE = 'configs/dos6_all.ini'
# CONFIG_FILE = 'configs/seq3f_all.ini'
# CONFIG_FILE = 'configs/seq5f_all.ini'
# CONFIG_FILE = 'configs/song1_all.ini'
# CONFIG_FILE = 'configs/song2_all.ini'

# CONFIG_FILE = 'configs/new_dance.ini'
# CONFIG_FILE = 'configs/new_dance2.ini'
# CONFIG_FILE = 'configs/new_dos1.ini'
# CONFIG_FILE = 'configs/new_dos6.ini'
# CONFIG_FILE = 'configs/new_seq3f.ini'
# CONFIG_FILE = 'configs/new_seq5f.ini'

# CONFIG_FILE = 'configs/song1.ini'
# CONFIG_FILE = 'configs/song2_all.ini'
# CONFIG_FILE = 'configs/albert1.ini'
#CONFIG_FILE = 'configs/music_intro.ini'
# CONFIG_FILE = 'configs/braunfels_1.ini' 
# CONFIG_FILE = 'configs/braunfels_2.ini' 
# CONFIG_FILE = 'configs/chappell_1.ini'
# CONFIG_FILE = 'configs/chappell_2.ini'
# CONFIG_FILE = 'configs/carol_1.ini' 
# CONFIG_FILE = 'configs/carol_2.ini' 
# CONFIG_FILE = 'configs/cinderella_1.ini'
# CONFIG_FILE = 'configs/legally_blonde_1.ini'
# CONFIG_FILE = 'configs/legally_blonde_2.ini'

# TODO
# IMP create the shots section using configparser
# Fix the videoname redundancy
# Add rendering framerate to config-DONE
# Create a config updater and a template config
# op_resolution variable binding in render function
# Complete MS-FS selection and add it to config
# remove video width and height to optimiser in utilities hard coding
# DONE - Check if utilities is needed
# Change os.system commands to subprocess module
##

def calculateGazeCost(frame_number, shot_tracks, gaze_tracks):
    """Gaze cost for all shots in a given frame

    Calculating the gaze cost in a frame for all shots based on the gaze
    points

    Parameters
    ----------
    frame_number : int
                   The frame number in which the gaze cost is to be
                   calculated
    shot_tracks : dict
                  A dictionary containing all the shot tracks, as key
                  value pairs, where key is the shot name and value is
                  an array of rectangle coords for all frames
    gaze_tracks : dict
                  Key-value pairs where key is the the ith gaze point
                  and value is the (x,y) coordinate of the point
    Returns
    -------
    gaze_cost : dict
                A key-value pair where the keys are the shot-keys and
                values are the gaze cost corresponding to  the frame
                number
    Notes
    -----
    The gaze cost is calculated in the following way, for one-shots (ie
    rushes with only a single performer) distance from gaze point to the
    center of the rush rectangle is calculated and summed for all gaze
    points (say D). Gaze cost = 1/D. If the performer isn't present in
    the frame (rectangle = (0,0,0,0)) then a default cost of 0.00001 is
    given (this is emperical, the gaze cost was usually obsereved to be
    in 10e-2 to 10e-3 range, so any number less than that can also work)
    For n-shots (rushes containing more than one performer), the
    following formula is used:
    Say H is the gaze cost of n-shot and L1 and L2 are the gaze cost of
    two n-1 shots which when combined make the n-shot, then
        H = L1 + L2 - abs(L1 - L2)
    where abs is the absolute gives the value

    """

    no_of_shots = len(list(shot_tracks.keys()))
    gaze_cost = {key: 0 for key in shot_keys}

    # assign unary cost for all 1-shots
    # then for every n-shot from 2 to n, assign cost using n-1 shot

    one_shots = getAllNShots(1)

    for shot in one_shots:
        centerX = (shot_tracks[shot][frame_number, 0] +
                   shot_tracks[shot][frame_number, 2]) * 0.5
        centerY = (shot_tracks[shot][frame_number, 1] +
                   shot_tracks[shot][frame_number, 3]) * 0.5
        center_point = [centerX, centerY]

        cost_for_present_shot = 0.00001
        if(not (centerX < 25 or centerY < 25)):
            for track_number in gaze_tracks:
            # for track_number in range(1,2):
                single_gaze_track = gaze_tracks[track_number]
                gaze_point = [float(single_gaze_track[frame_number][0]) *
                              normFactorX+float(gazeXOffset),
                              float(single_gaze_track[frame_number][1]) *
                              normFactorY+float(gazeYOffset)]

                if(int(gaze_point[0]) > width or int(gaze_point[1]) > height):
                    continue
                cost_for_present_shot += distance(center_point, gaze_point)
            if(cost_for_present_shot != 0.00001):
                gaze_cost[shot] = 1/float(cost_for_present_shot)
            else:
                # in case all gaze points are out of bounds
                gaze_cost[shot] = cost[shot][frame_number-1]
        else:
            gaze_cost[shot] = float(cost_for_present_shot)

    # for n-shot, the unary cost is defined as follows
    for i in range(2, len(actors)+1):
        n_shots = getAllNShots(i)
        cost_for_present_shot = 0

        for n_shot in n_shots:
            lower_n_shots = getAllNShots(i-1)
            lower_n_shots = [s
                             for s in lower_n_shots
                             if containedActors(s, n_shot)]
            lower_n_shots = sorted(
                lower_n_shots,
                key=lambda x: shot_tracks[x][frame_number][0])
            cost_for_present_shot = (gaze_cost[lower_n_shots[0]] +
                                     gaze_cost[lower_n_shots[1]] -
                                     abs(gaze_cost[lower_n_shots[0]] -
                                         gaze_cost[lower_n_shots[1]]))

            gaze_cost[n_shot] = cost_for_present_shot

        pass
    return gaze_cost


def addStaticCosts(present_frame):
    static_cost = {key: 0 for key in shot_keys}
    for shot in shot_keys:
        # Shot composition cost
        static_cost[shot] = calculateShotCompositionCost(shot, present_frame)
    return static_cost
    # pass

def addDynamicCosts(present_frame):
    """Dynamic cost addition for a frame

    Calculates the different dynamic costs (such as state transition
    cost, rythm cost, overlap cost) for all possible state transitions.
    Then updates the backtrack array, wrt to the state transition with
    minimum total dynamic cost incurred.

    Parameters
    ----------
    present_frame : int
                    The frame number in which the gaze cost is to be
                    calculated
    Notes
    -----
    It is assumed that the dynamic cost array(`dp_cost`) is a global
    array.
    For a given state in present_frame, total cost from state overlap,
    state transition and rythm are added to the initial gaze_cost of the
    state. Then, the state with the minimum total is recorded in the
    backtracking array.

    """

    previous_frame = present_frame-1

    if(present_frame == 0):
        for key in shot_keys:
            dp_cost[key][present_frame] = cost[key][present_frame]
            back_track[key][present_frame] = key
    else:
        for shot in shot_keys:
            min_cost = np.inf
            update_cost = {key: 0 for key in shot_keys}
            for prev_shot in shot_keys:
                # Previous dp cost
                update_cost[prev_shot] = (dp_cost[prev_shot][previous_frame] +
                                          cost[shot][present_frame])
                # Transition cost
                update_cost[prev_shot] += calculateTransitionCost(
                    prev_shot,
                    shot,
                    present_frame)
                # Rythm cost
                update_cost[prev_shot] += calculateRythmCost(shot,
                                                             previous_frame)
                # Min shot duration Cost
                update_cost[prev_shot] += calculateMinShotDurationCost(shot,                                                        previous_frame)
                # Calculate Context cost
                # update_cost[prev_shot] += calculateContextCost(prev_shot,shot)

                # Overlap cost
                if(prev_shot != shot):
                    update_cost[prev_shot] += calculateOverlapCost(
                        present_frame,
                        prev_shot,
                        shot)
                

                if(update_cost[prev_shot] < min_cost):
                    min_cost = update_cost[prev_shot]
                    back_track[shot][present_frame] = prev_shot

            if(back_track[shot][present_frame] == shot):
                # increase timer for a shot
                shot_duration[shot][present_frame] = shot_duration[shot][previous_frame] + 1
            else:
                if(shot_duration[shot][previous_frame] > 0):
                    # reset timer on shot change
                    shot_duration[shot][present_frame] = 0

            dp_cost[shot][present_frame] = (dp_cost[shot][present_frame] +
                                            min_cost)

def calculateContextCost(prev_shot,shot):
    prev_shot_n = findNShot(prev_shot)
    shot_n = findNShot(shot)
    if(shot==prev_shot):
        return 0
    elif(shot_n==prev_shot_n):
        return high_to_low_shot_context_factor*cost_norm_factor
    elif (shot_n>prev_shot_n):
        return low_to_high_shot_context_factor*cost_norm_factor
    else:
        return high_to_low_shot_context_factor*cost_norm_factor
        pass
    pass

    # context_cost_matrix = {key:{shot:0 for shot in shot_keys} for key in shot_keys}
    # for shot_i in shot_keys:
    #     for shot_j in shot_keys:
    #         i = findNShot(shot_i)
    #         j = findNShot(shot_j)
    #         if(shot_i==shot_j):
    #             context_cost_matrix[shot_i][shot_j] = 0
    #         elif(i<j):
    #             context_cost_matrix[shot_i][shot_j] = 0.01*cost_norm_factor
    #         elif(i==j):
    #             context_cost_matrix[shot_i][shot_j] = 0.01*cost_norm_factor
    #         elif(i>j):
    #             context_cost_matrix[shot_i][shot_j] = 100*cost_norm_factor
    # return context_cost_matrix[prev_shot][shot]

def loadConfig(config_file):
    """Loads a config file for a video

    This initializes all the global variable used in this file, by
    reading them from the provided config file

    Parameters
    ----------
    config_file : str
                  Name/path of the config file for a particular video
                  to be edited. Eg: 'film1.ini'
    Notes
    -----
    Below is the comprehensive description of all the global variables
    DEBUG_MODE : int
                 This variable if set, allows printing the actual cost
                 stored wrt frame number in each state. Also provides
                 some metadata about the files loaded while execution
                 for easier debugging. If unset, will render the final
                 edited video as an mp4 file
    output_video_name : str
                        Name that will be used to render the edited
                        video. Used by renderVideoFromCroppedWindow().
                        Default: output.mp4
    low_transition_factor : float
                            The multuplier to cost_norm_factor, used by
                            analyticTransitionFunction() to calculate the
                            tranistion cost when shot duration is more than
                            `min_shot_duration`
                            Default: 0.01
    high_transition_factor : float
                             The multuplier to cost_norm_factor, used by
                             analyticTransitionFunction() to calculate the
                             tranistion cost when shot duration is less than
                             `min_shot_duration`
                             Default: 1
    video_frames : dict
                   Path to folder containing all the video frames as
                   images (preferred jpg) as key-value where key is the
                   video name (TBF). Used by loadFrames()
    rythm_factor : float
                   A multiplier to cost_norm_factor used by the
                   analyticRythmFunction() to calculate the rythm cost
                   for a state.
                   Default: 2
    min_shot_duration : int
                        The minimum duration (in frames) a shot is to persist 
                        before transitioning to another shot
                        Default: 60
    low_overlap_factor : float
                         A multiplier for cost_norm_factor to calculate
                         overlap cost when overlap_ratio between two
                         frames is less than low_overlap_percentage.
                         Used by analyticOverlapFunction() to
                         calculate overlap cost for a state transition.
                         Default: 0.01
    mid_overlap_factor : float
                         A multiplier for cost_norm_factor to calculate
                         overlap cost when overlap_ratio between two
                         frames is less than mid_overlap_percentage and
                         greater than low_overlap_percentage.
                         Used by analyticOverlapFunction()
                         to calculate overlap cost for a state
                         transition.
                         Default: 10
    high_overlap_factor : float
                          A multiplier for cost_norm_factor to calculate
                          overlap cost when overlap_ratio between two
                          frames is more than mid_overlap_percentage.
                          Used by analyticOverlapFunction() to
                          calculate overlap cost for a state transition.
                          Default: 1000
    low_overlap_percentage : float
                             A number between 0 and 1 which signifies
                             the lower tolerance limit for overlap_ratio
                             between two frames while calculating the
                             overlap cost.
                             Used by analyticOverlapFunction().
                             Default: 0.3
    mid_overlap_percentage : float
                             A number between 0 and 1 which signifies
                             the mid tolerance limit for overlap_ratio
                             between two frames while calculating the
                             overlap cost.
                             Used by analyticOverlapFunction().
                             Default: 0.5
    audio : dict
            A key-value where key is the videoname and the value
            specifies the path to the audio file for the video for
            rendering. Used by addAudioToVideo()
    establishing_shot_time : int
                             Signifies the min number of frames from 0
                             during which the wide shot has to be used.
                             Default: 60
    videoName : str
                Name of the input video. (This is useful as the rushes
                are prefixed with videoName for convienience)
    video : dict
            A key-value pair, where the key is the video name and the
            value is the path of the video. (TBF)
    shots : dict
            A set of keys and values where the keys are the rush names
            and the values correspong to the path of the tracks
    actors : list
             A list containing the names of all the performers
    width : int
            Width of the input video (in pixels).
    height : int
             Height of the input video (in pixels)
    gaze : dict
           A key-value pair, where the key is the video name and the
           value is the path to the gaze tracks. (TBF)
    normFactorX : int
                  The screen width when the gaze was recorded. This
                  is used to match the gaze points to the actual video
                  when the resolutions (gaze and video) differ.
                  This is updated to their ratio.
                  Default: 1920
    normFactorY : int
                  The screen height when the gaze was recorded. This
                  is used to match the gaze points to the actual video
                  when the resolutions (gaze and video) differ.
                  This is updated to their ratio.
                  Default: 1080
    gazeXOffset : int
                  A number (in pixels), to correct the deviation in gaze
                  points in X direction,in case the calibration was
                  incorrect.
                  Default: 0
    gazeYOffset : int
                  A number (in pixels), to correct the deviation in gaze
                  points in Y direction,in case the calibration was
                  incorrect.
                  Default: 0
    shot_keys : list
                List of the names of the shots

    """
    config_reader = ConfigParser(interpolation=ExtendedInterpolation())
    config_reader.read(config_file)

    global DEBUG_MODE, \
    VERBOSE, \
    COST_PLOT, \
    output_video_name, \
    video_frames, \
    low_rythm_factor, \
    high_rythm_factor, \
    low_overlap_factor, \
    high_overlap_factor, \
    audio, \
    low_overlap_percentage, \
    mid_overlap_percentage, \
    mid_overlap_factor, \
    establishing_shot_time, \
    videoName, \
    basedir, \
    video, \
    fps, \
    shots, \
    actors, \
    width, \
    gaze, \
    gazeXOffset, \
    gazeYOffset, \
    normFactorX, \
    normFactorY, \
    shot_keys, \
    height, \
    min_shot_duration, \
    max_shot_duration, \
    transition_factor, \
    low_min_shot_duration_factor, \
    high_min_shot_duration_factor, \
    low_to_high_shot_context_factor, \
    high_to_low_shot_context_factor,\
    shot_composition_factor, \
    MS_composition_threshold, \
    FS_composition_threshold

    DEBUG_MODE = config_reader.getint('exec', 'DEBUG_MODE', fallback=0)
    VERBOSE = config_reader.getint('exec', 'VERBOSE', fallback=1)
    COST_PLOT = config_reader.getint('exec', 'COST_PLOT', fallback=1)

    fps = config_reader.getfloat('video','fps',fallback=30)

    output_video_name = config_reader.get('exec', 'output_video_name',
                                          fallback='output.mp4')
    transition_factor = config_reader.getfloat('parameters',
                                                    'transition_factor',
                                                    fallback=1)
    high_min_shot_duration_factor = config_reader.getfloat('parameters',
                                                    'high_min_shot_duration_factor',
                                                    fallback=100)
    low_min_shot_duration_factor = config_reader.getfloat('parameters',
                                                   'low_min_shot_duration_factor',
                                                   fallback=0.01)
    low_rythm_factor = config_reader.getfloat('parameters',
                                          'low_rythm_factor',
                                          fallback=0.01)
    high_rythm_factor = config_reader.getfloat('parameters',
                                          'high_rythm_factor',
                                          fallback=100)
    min_shot_duration = int(fps*config_reader.getfloat('parameters',
                                             'min_shot_duration',
                                             fallback=2))
    max_shot_duration = int(fps*config_reader.getfloat('parameters',
                                             'max_shot_duration',
                                             fallback=4))

    low_overlap_factor = config_reader.getfloat('parameters',
                                                'low_overlap_factor',
                                                fallback=0.01)
    low_overlap_percentage = config_reader.getfloat('parameters',
                                                    'low_overlap_percentage',
                                                    fallback=0.3)
    mid_overlap_factor = config_reader.getfloat('parameters',
                                                'mid_overlap_factor',
                                                fallback=1)
    mid_overlap_percentage = config_reader.getfloat('parameters',
                                                    'mid_overlap_percentage',
                                                    fallback=0.5)
    high_overlap_factor = config_reader.getfloat('parameters',
                                                 'high_overlap_factor',
                                                 fallback=100)
    low_to_high_shot_context_factor = config_reader.getfloat('parameters',
                                                             'low_to_high_shot_context_factor',
                                                             fallback=0.01)
    high_to_low_shot_context_factor = config_reader.getfloat('parameters',
                                                             'high_to_low_shot_context_factor',
                                                             fallback=100)
    shot_composition_factor = config_reader.getfloat('parameters',
                                                    'shot_composition_factor',
                                                    fallback=0.01)
    MS_composition_threshold = config_reader.getfloat('parameters','MS_composition_threshold',fallback=0.4)                                                   
    FS_composition_threshold = config_reader.getfloat('parameters','FS_composition_threshold',fallback=0.4)                                                    

    establishing_shot_time = int(fps*config_reader.getfloat('parameters',
                                                  'establishing_shot_time',
                                                  fallback=4))
    gazeXOffset = config_reader.getint('parameters',
                                       'gazeXOffset',
                                       fallback=0)
    gazeYOffset = config_reader.getint('parameters',
                                       'gazeYOffset',
                                       fallback=0)
    # incase the resolution of gaze recording is different from the
    # original videoresolution
    # for dos6 it is 1366x768
    # rest is 1920x1080

    normFactorX = config_reader.getint('parameters',
                                       'normFactorX',
                                       fallback=1920)
    normFactorY = config_reader.getint('parameters',
                                       'normFactorY',
                                       fallback=1080)

    videoName = config_reader.get('video', 'name')
    basedir = config_reader.get('video', 'basedir')

    video = {videoName: config_reader.get('video', 'path')}
    audio = {videoName: config_reader.get('video', 'audio')}
    video_frames = {videoName: config_reader.get('video', 'frames')}
    gaze = {videoName: config_reader.get('video', 'gaze')}
    width = config_reader.getint('video', 'width')
    height = config_reader.getint('video', 'height')

    # shots = eval(config_reader.get('performers', 'shots'))
    shots = config_reader['shots']
    actors = config_reader.get('performers', 'actors')

    shot_keys = list(shots.keys())

    normFactorX = width/normFactorX
    normFactorY = height/normFactorY

    pass

def distance(p0, p1):
    """Calculates Euclidean distance

    Parameters
    ----------
    p0,p1 : list
            list containing the x and y coordinate
    Returns
    -------
    A float which the distance between p0 and p1

    """
    return math.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)
    pass


def loadShotTracks():
    """Loads the Shot tracks into a dictionary

    Loads the shot tracks from `shots` by reading from the paths 
    contained in it into a dictionary

    Returns
    -------
    shot_tracks : dict
                  A set of key-values where the keys are the shot names 
                  and values are the bounding rectangles

    """
    print(Fore.WHITE+'Loading Shot Tracks ...')
    shot_tracks = {}
    for shot in shot_keys:
        DEBUG_MODE and print(Fore.WHITE+'Loading shot '+shot +
                             ' from : ' + shots[shot])
        shot_tracks[shot] = ut.get_rectangle(shots[shot])
    print(Fore.GREEN+'Done')
    return shot_tracks


def loadGazeTracks():
    """Loads the gaze tracks into a dictionary

    Loads the gaze tracks from `gaze` by reading from the path contained
    in it into a dictionary

    Returns
    -------
    gaze_tracks : dict
                  A set of key-values where the keys are the gaze point 
                  number and values are the gaze point coordinates

    """
    print(Fore.WHITE+'Loading Gaze Tracks ...')
    with open(gaze[videoName], 'r') as fp:
        DEBUG_MODE and print(Fore.WHITE+"Tracks filename: ", gaze[videoName])
        gaze_tracks  = yaml.load(fp)
    print(Fore.GREEN+'Done')
    return gaze_tracks


def loadFrames():
    """Loads the video frames into an array

    Loads the video frames from `video_frames` by reading from the path 
    contained in it into a list

    Returns
    -------
    frames : list
             List containing the all the frame paths.

    Notes
    -----
    This variable is useful in rendering the final video.

    """
    print(Fore.WHITE+'Loading Frames ...')
    DEBUG_MODE and print(Fore.WHITE+'Loading Video Frames from ' +
                         video_frames[videoName])
    frames = [file
              for file in os.listdir(video_frames[videoName])
              if file.endswith('.jpg')]
    frames.sort()
    print(Fore.GREEN+'Done')
    return frames


def shotComposition(shot):
    """Returns the type of shot

    Parameters
    ----------
    shot : str
           Shot name which follows the naming convention given in the   
           example, eg: 'actor1-actor2-...-actorN-shot_composition'

    Returns
    -------
    The last two characters in the name of the shot provided

    """
    return shot[-2::1]
    pass


def findNShot(shot):
    """Returns the number of performers in the shot

    Parameters
    ----------
    shot : str
           Shot name which follows the naming convention given in the   
           example, eg: 'actor1-actor2-...-actorN-shot_composition'

    Returns
    -------
    The count of number of performers in the shot

    """
    return shot.count('-')
    pass


def actorsInNShot(shot):
    """Returns the list of performers in the shot

    Parameters
    ----------
    shot : str
           Shot name which follows the naming convention given in the   
           example, eg: 'actor1-actor2-...-actorN-shot_composition'

    Returns
    -------
    The list of performers in the given shot

    """
    t = shot.split('-')
    return t[0:-1]


def containedActors(a, b):
    """Checks whether shot `b` contains shot `a`

    Checks whether all the performers in `a` occur in `b` or not

    Parameters
    ----------
    a, b : str
           Both are shot names following the convention given below
           convention: 'actor1-actor2-...-actorN-shot_composition'

    Returns
    -------
    A bool ascertaining whether performers in `a` are contained in `b`

    """
    a = actorsInNShot(a)
    b = actorsInNShot(b)
    return set(a).issubset(set(b))


def getAllNShots(n):
    """Get all shots containing `n` performers

    Parameters
    ----------
    n : int
        The number of required perfomers
    Returns
    -------
    l : list
        The list of all shot names containing exactly n performers

    """
    l = list(shots.keys())
    l = [shot for shot in shot_keys if(findNShot(shot) == n)]
    return l


def getAllNShotsWithActors(n, actors):
    """Get all shots containing `n` performers and 'actors'

    Parameters
    ----------
    n : int
        The number of required perfomers
    actors : list
             The list of all compulsory performers
    Returns
    -------
    l : list
        The list of all shot names containing exactly n performers and
        the compulsory actors

    """
    l = list(shots.keys())
    l = [shot for shot in shot_keys if(findNShot(shot) == n and
                                       set(actors).issubset(
        actorsInNShot(shot)))]
    return l


def calculateTransitionCost(prev_shot, pres_shot, frame_number):
    """Calculate transition cost from prev_shot to pres_shot in frame

    Parameters
    ----------
    prev_shot : str
                The name of previous state
    pres_shot : str
                The name of present state
    frame_number : int
                   The frame at which transition cost is to be
                   calculated
    Returns
    -------
    A float which is the transition cost from prev_shot to pres_shot
    for frame_number

    Notes
    -----
    This is achieved by creating a matrix between all pair of
    states ('transition_matrix'). It is populated using the function
    analyticTransitionFunction()

    """
    # transition_matrix = {key:
    #                      {k: 0 for k in shot_keys}
    #                      for key in shot_keys}
    # for key in shot_keys:
    #     for k in shot_keys:
    #         if key == k:
    #             transition_matrix[key][k] = 0
    #         else:
    #             transition_matrix[key][k] = analyticTransitionFunction(
    #                 shot_duration[prev_shot][frame_number])
    # print(transition_matrix[time][prev_shot][pres_shot])
    # return transition_matrix[prev_shot][pres_shot]

    # time = int(shot_duration[prev_shot][frame_number-1])
    # return transition_matrix[time][prev_shot][pres_shot]
    return transition_matrix[prev_shot][pres_shot]

def calculateMinShotDurationCost(shot,previous_frame):
    time = int(shot_duration[shot][previous_frame])
    min_shot_duration_cost = analyticMinShotDurationFunction(time)
    return min_shot_duration_cost

def getAllShotCompositionOfNShot(shot):
    """Get all the existing shot compositions of a given shot

    Given an n-shot, it finds all the existing shot compositions for
    rushes containing exactly the same performers in `shot`. Eg:
    'actor1-actor-2-MS' and 'actor1-actor-2-FS' are two possible shots
    for 'actor1-actor2', so the function returns `['MS','FS']`

    Parameters
    ----------
    shot : str
           Name of the required shot
    Returns
    -------
    shots : list
            A list of all existing shot compositions
    """
    n = findNShot(shot)
    shots = getAllNShots(n)
    shots = [s for s in shots if containedActors(s, shot)]
    return shots


def analyticRythmFunction(time):
    """Calculates the rythm cost for duration mathematically

    Parameters
    ----------
    time : int
           The duration for which the rythm cost is to be calculated
    Returns
    -------
    rythm_cost : float
                 The calculated rythm cost
    Notes
    -----
    For calculation of rythm cost, a simple linearly growing cost
    function is used, which is normalised such that the cost reaches 
    `cost_norm_factor` after 500 frames (assuming rythm factor is 1),
    this is arbitrary (ie 500) and can be changed according to need, but a 
    better approach would be to increase rythm factor instead of the 
    constant used as that can be changed from the config file for every 
    specific video and hence more convienient

    """
    # rythm_cost = time*rythm_factor*cost_norm_factor/500

    # low_rythm_percentage = 36
    # mid_rythm_percentage = 50
    # low_rythm_factor = 0.01
    # mid_rythm_factor = 1
    # high_rythm_factor = 1

    # if(time <= low_rythm_percentage):
    #     rythm_cost = low_rythm_factor*cost_norm_factor
    # elif(time > low_rythm_percentage and time <= mid_rythm_percentage):
    #     slope = ((mid_rythm_factor - low_rythm_factor) *
    #              cost_norm_factor /
    #              (mid_rythm_percentage - low_rythm_percentage))
    #     rythm_cost = (time *
    #                     slope +
    #                     (low_rythm_factor *
    #                      cost_norm_factor -
    #                      low_rythm_percentage * slope))
    # else:
    #     rythm_cost = high_rythm_factor*cost_norm_factor

    #new rythm function starts here
    lrf = cost_norm_factor*low_rythm_factor
    hrf = cost_norm_factor*high_rythm_factor
    max_shot_duration = 120
    rythm_cost = -hrf/(1+np.exp(time-max_shot_duration))+lrf+hrf

    return rythm_cost


# def analyticTransitionFunction(time):
def analyticMinShotDurationFunction(time):
    """Calculates the state transition cost

    Parameteres
    -----------
    time : int
            No of frames, which is a measure of amount of duration
    Returns
    -------
    transition_cost : float
                      The calculated transition cost
    Notes
    -----
    This function calculates the analytical transition cost between two
    different states. The function graphically has a high initial cost (ie
    when `time` is less than `min_shot_duration`), which decreases after the
    threshold is crossed. This is to prevent high frequency transitions

    """

    min_shot_duration_cost = 0
    # if(time < min_shot_duration):
    #     min_shot_duration_cost = high_transition_factor * cost_norm_factor
    # else:
    #     min_shot_duration_cost = low_transition_factor * cost_norm_factor

    #new transition function starts here (To be named min shot duration function)
    high_factor = high_min_shot_duration_factor * cost_norm_factor
    low_factor = low_min_shot_duration_factor * cost_norm_factor
    min_shot_duration_cost = -high_factor/(1+np.exp(-time+min_shot_duration)) + low_factor+high_factor
    return min_shot_duration_cost


def analyticOverlapFunction(ratio):
    """Calculates the overlap cost incurred for a given ratio

    Parameteres
    -----------
    ratio : float
            A ratio, which is essentially the overlap ratio between two
            different shot rectangles at a frame
    Returns
    -------
    overlap_cost : float
                   The cost incurred for a given ratio
    Notes
    -----
    This function mathematically calculates the overlap cost that should
    be incurred given the overlap ratio. This function is a piecewise
    function containing contant and linear parts. The following example
    explains the function : 
    If `low_overlap_percentage` = 0.3 and `mid_overlap_percentage` = 0.5
    then this is a piecewise function which is const from 0 - 0.3, 
    linear between 0.3 and 0.5 and returns very high const after 0.5.
    The multipliers (for cost_norm_factor) used are
    `low_overlap_factor`,`mid_overlap_factor` and `high_overlap_factor`
    respectively.

    """

    overlap_cost = 0
    if(ratio <= low_overlap_percentage):
        overlap_cost = low_overlap_factor*cost_norm_factor
    elif(ratio > low_overlap_percentage and ratio <= mid_overlap_percentage):
        slope = ((mid_overlap_factor - low_overlap_factor) *
                 cost_norm_factor /
                 (mid_overlap_percentage - low_overlap_percentage))
        overlap_cost = (ratio *
                        slope +
                        (low_overlap_factor *
                         cost_norm_factor -
                         low_overlap_percentage * slope))
    else:
        overlap_cost = high_overlap_factor*cost_norm_factor

    return overlap_cost


def initialiseGlobalMatrices():
    global shot_composition_cost, transition_matrix

    shot_composition_cost = {
        'MS': lambda width_ratio: shot_composition_factor*cost_norm_factor
        if width_ratio > MS_composition_threshold else 0,
        'FS': lambda width_ratio: shot_composition_factor*cost_norm_factor
        if width_ratio < FS_composition_threshold else 0
    }

    # transition_matrix = [{key:
    #                       {k: 0 for k in shot_keys}
    #                       for key in shot_keys}
    #                      for i in range(no_of_frames)]
    # for t in range(no_of_frames):
    #     for key in shot_keys:
    #         for k in shot_keys:
    #             if(k == key):
    #                 transition_matrix[t][key][k] = 0
    #             else:
    #                 transition_matrix[t][key][k] = analyticTransitionFunction(
    #                     t)

    transition_matrix = {key: {k:0 for k in shot_keys} for key in shot_keys}
    for k in shot_keys:
        for key in shot_keys:
            if(k==key):
                transition_matrix[k][key]=0
            else:
                transition_matrix[k][key]=transition_factor*cost_norm_factor


def analyticShotCompositionFunction(width_ratio, shot_composition):
    # shot_composition_cost = {
    #     'MS': lambda width_ratio: 0.1*cost_norm_factor
    #                  if width_ratio > 0.3 else 0,
    #     'FS': lambda width_ratio: 0.1*cost_norm_factor
    #                  if width_ratio < 0.3 else 0
    # }
    return shot_composition_cost[shot_composition.upper()](width_ratio)
    # if(width_ratio<0.25):
    #   return 0.001*cost_norm_factor
    # else:
    #   return 0.001*cost_norm_factor


def findShotComposition(shot):
    return shot.split('-')[-1]
    pass


def calculateShotCompositionCost(shot, frame_number):

    # print(findShotComposition(all_compositions[0])=="FS")
    # width_ratios = {composition:
                      # (shot_tracks[composition][frame_number][2] -
                      #  shot_tracks[composition][frame_number][0]) /
                      # width
                      # for composition in all_compositions}
    # print(width_ratios)
    if(len(actorsInNShot(shot)) > 1):
        shot_composition = findShotComposition(shot)
        width_ratio = ((shot_tracks[shot][frame_number][2] -
                        shot_tracks[shot][frame_number][0]) /
                       width)
        return analyticShotCompositionFunction(width_ratio,
                                               shot_composition)
    else:
        return 0
    pass


def calculateOverlapCost(frame_number, from_shot, to_shot):
    """Calculates the overlap cost for two shots in a frame

    At a particular frame number, calculates the frame overlap cost
    between two shots.

    Parameters
    ----------
    frame_number : int
                   The required frame number to calculate the cost
    from_shot : str
                Name of the shot from which the transition is expected
    to_shot : str
              Name of the shot to which the transition is exprected
    Returns
    -------
    A float, which is the overlap cost. If the area of the to_shot is
    less than 100 pixels**2, then the overlap is considered to be 100%.
    as this undesirable so we assign a high cost to if by default

    """
    # x1 y1 x2 y2
    frame1 = shot_tracks[from_shot][frame_number-1]
    frame2 = shot_tracks[to_shot][frame_number]

    overlap_x1 = max(frame1[0], frame2[0])
    overlap_y1 = max(frame1[1], frame2[1])
    overlap_x2 = min(frame1[2], frame2[2])
    overlap_y2 = min(frame1[3], frame2[3])

    overlap_area = (overlap_x2-overlap_x1)*(overlap_y2-overlap_y1)
    frame1_area = (frame1[2]-frame1[0])*(frame1[3]-frame1[1])
    frame2_area = (frame2[2]-frame2[0])*(frame2[3]-frame2[1])

    if(frame2_area <= 100):
        overlap_ratio = 1  # shot doesnt exit, never cut to this shot
    else:
        overlap_ratio = overlap_area/frame2_area
    overlap_cost = analyticOverlapFunction(overlap_ratio)
    return overlap_cost


def calculateRythmCost(pres_shot, frame_number):
    """Calculates the rythm cost for a shot in a frame

    Uses the analyticRythmFunction() to calculate the cost

    Parameters
    ----------
    pres_shot : str
                The name of the shot for which rythm cost is to be 
                calculated
    frame_number : int
                   The frame number where rythm cost is to be calculated
    Returns
    -------
    A float which is the rythm cost

    """

    return analyticRythmFunction(shot_duration[pres_shot][frame_number])
    pass


def generateCroppedWindow(final_track):
    """Generates the edited cropping window from `final_track`

    Parameters
    ----------
    final_track : list
                  This is a list containing states (shot names), and is
                  the length of `no_of_frames`
    Returns
    -------
    A numpy array, with the final rectangle coordinates (x1,y1,x2,y2)
    for the video to be rendered

    """
    cropped_window = []
    wide_shot = [0, 0, width, height]
    # print(final_track)
    for frame_number, shot in enumerate(final_track):
        if shot == '':
            cropped_window.append(wide_shot)
        else:
            cropped_window.append(shot_tracks[shot][frame_number])

    for frame in range(establishing_shot_time):
        cropped_window[frame] = wide_shot

    # for frame in range(len(cropped_window)):
    #     if (cropped_window[frame][2]<25 or cropped_window[frame][3]<25):
    #         for i in range(-30,31,1):
    #             cropped_window[frame+i] = wide_shot 

    return np.asarray(cropped_window)


def renderVideoFromCroppedWindow(input_video,
                                 cropped_window,
                                 output_video_name,
                                 index=0):
    """Renders an edited video in mp4 format from input video

    Creates an mp4 file with the name given in `output_video_name` which
    is the edited video

    Parameters
    ----------
    input_video : str
                  path for the video to be edited
    cropped_window : numpy array
                     Coordinates of the rectangle in each frame, which
                     will be cropped for the edited video to be
                     generted
    output_video_name : str
                        Desired name for output edited video.
                        Eg: 'output', 'john' (ommit the file format)
    index : int
            Offset for if any delay exists in bounding rectangles and
            the video frames
            Default: 0

    Notes
    -----
    Output resolution by default is 1920x1080, and can be changed by 
    changing the variables `op_resolution_w` and `op_resolution_h` which
    are the width and height respectively. When the debug mode is set, it
    renders a video with the final cropped window, with frame number and
    the shot it represents.

    """

    op_resolution_w = 1920
    op_resolution_h = 1080

    framerate = fps
    # framerate = 23.98
    # framerate = 59.94

    stamp = datetime.datetime.today().strftime('%H%M%d%m%Y')
    
    output_video_name = output_video_name+'_'+stamp+'.mp4'
    if(DEBUG_MODE):
        output_video_name = 'debug_'+output_video_name

    output_video_name = '../Outputs/'+videoName+'/'+output_video_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'x264' doesn't work
    out = cv2.VideoWriter('../Outputs/'+videoName+'/'+'temp.mp4', fourcc, framerate, (1920, 1080))

    cap = cv2.VideoCapture(input_video)
    # index=5 #offset to sync coordinate shot track and video

    print('Rendering Video...')

    # while index in range(no_of_frames-4):
    for index,frame in enumerate(frames):
        # ret, orig_frame = cap.read()
        orig_frame = cv2.imread(basedir+'/video_frames/'+frame)
        # index += 1

        if DEBUG_MODE:
            for p in gaze_tracks:
                gaze_point = (int(float(gaze_tracks[p][index][0]) *
                                  normFactorX +
                                  float(gazeXOffset)),
                              int(float(gaze_tracks[p][index][1]) *
                                  normFactorY +
                                  float(gazeYOffset)))
                cv2.circle(orig_frame, gaze_point,
                           color=(0, 255, 0),
                           radius=5,
                           thickness=6)

            cv2.rectangle(orig_frame,
                          (int(cropped_window[index][0]),
                           int(cropped_window[index][1])),
                          (int(cropped_window[index][2]),
                           int(cropped_window[index][3])),
                          (0, 0, 255), 2)
            cropped_frame = orig_frame
            frame_text = 'Frame : '+str(index)
            shot_text = 'Shot : '+final_track[index]
            cv2.putText(cropped_frame, frame_text,
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))
            cv2.putText(cropped_frame, shot_text,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0))

        else:
            if(cropped_window[index][0] == 0 and
                cropped_window[index][2] == 0 or
                cropped_window[index][1] == 0 and
                    cropped_window[index][3] == 0):
                cropped_frame = orig_frame[0:1, 0:1]
            else:
                cropped_frame = orig_frame[int(cropped_window[index][1]):
                                           int(cropped_window[index][3]),
                                           int(cropped_window[index][0]):
                                           int(cropped_window[index][2])]

        # cropped_frame = cv2.resize(cropped_frame, (int(1.7779*720),720))
        cropped_frame = cv2.resize(cropped_frame,
                                   (op_resolution_w, op_resolution_h))
        out.write(cropped_frame)

        # sys.stdout.write('\r')
        percentage = float(index/no_of_frames)*100
        # sys.stdout.write(str('%0.2f' % percentage))
        print(str('%0.2f' % percentage),end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # adding audio

    sys.stdout.write('\r')
    s = '100 : Done'
    sys.stdout.write(s)

    if(audio[videoName] != 'NA'):
        output_video_name = addAudioToVideo('../Outputs/'+videoName+'/'+'temp.mp4',
                                            audio[videoName],
                                            output_video_name)
    else:
        shell_command = 'rename'+'../Outputs/'+videoName+'/'+'temp.mp4 '+output_video_name
        os.system(shell_command)

    print(Fore.GREEN+'\nRendered Video : '+output_video_name)
    video_stats = os.stat(output_video_name)
    video_size = float(video_stats.st_size)/(1024*1024)
    print(Fore.GREEN+'Size : '+str('%2f' % video_size)+'M')
    print(Fore.GREEN+'Resolution : ' + str(op_resolution_w) +
          'x'+str(op_resolution_h))
    print(Fore.GREEN+'Audio file : '+audio[videoName])

    printParameters()
    pass

def printParameters():
    print(Fore.WHITE+'Parameters used : ')
    print(Fore.CYAN+'Low Min Shot Duration Factor : ' +
          str(low_min_shot_duration_factor))
    print('High Min Shot Duration Factor : '+str(high_to_low_shot_context_factor))
    print('Low to high shot context Factor : '+str(low_to_high_shot_context_factor))
    print('High to low shot context Factor : '+str(high_min_shot_duration_factor))
    print('Transition Factor : '+str(transition_factor))
    print('Min Shot Duration : '+str(min_shot_duration))
    print('Max Shot Duration : '+str(max_shot_duration))
    print('Low Overlap Percentage : '+str(low_overlap_percentage))
    print('Mid Overlap Percentage : '+str(mid_overlap_percentage))
    print('Low Overlap Factor : '+str(low_overlap_factor))
    print('Mid Overlap Factor : '+str(mid_overlap_factor))
    print('High Overlap Factor : '+str(high_overlap_factor))
    print('Low Rythm Factor : '+str(low_rythm_factor))
    print('High Rythm Factor : '+str(high_rythm_factor))
    print('Shot Composition Factor : '+str(shot_composition_factor))
    print('MS composition threshold : '+str(MS_composition_threshold))
    print('FS composition threshold : '+str(FS_composition_threshold))
    print('Gaze X Offset : '+str(gazeXOffset))
    print('Gaze Y Offset : '+str(gazeYOffset))
    print('Norm X Factor : '+str(normFactorX))
    print('Norm Y Factor : '+str(normFactorY))
    print('Establishing frames : '+str(establishing_shot_time))
    pass

def addAudioToVideo(input_video, audio, output_video_name):
    """Adding audio to the edited video

    Parameters
    ----------
    input_video : str
                  Path for input video
    audio : str
            Path for the audio file
    output_video_name : str
                        Desired output video name
    Returns
    -------
    output_video_name : str
                        The name of the renedered video after
                        audio addition

    Notes
    -----
    This function takes in the input video path (which is actually
    temp.mp4 if called from renderVideoFromCroppedWindow()), adds the 
    audio using ffmpeg and generated the new file. The old file is 
    DELETED

    """
    # ffmpeg does not take path, but works with quotes, so we are
    # converting the string to quotes (single quotes do not work)

    audio = repr(audio)
    audio = audio.replace('\'','"')
    
    input_video = repr(input_video)
    input_video = input_video.replace('\'','"')
    
    output_name = output_video_name
    shell_command = ('ffmpeg -i ' +
                     str(audio) +
                     ' -i ' +
                     str(input_video) +
                     ' -codec copy -shortest ' +
                     str(output_video_name))
    os.system(shell_command)
    if(not os.path.isfile(output_video_name)):
        print(Fore.RED+'Could not add audio, exiting.. your output video is ../Outputs'+videoName+'/temp.mp4')
        exit(0)
    else:
        os.system('del '+input_video)
    return output_name


def playWithGaze(videoName, rectangle, gaze_tracks, index=0):
    """ Plays a video with visible bounding rectangle and gazepoints 

    Parameters
    ----------
    videoName : str
                Path to the video
    rectangle : numpy array
                Coordinates of a bounding rectangle to be plotted in
                the format (x1,y1,x2,y2) where the corners used are
                top left and bottom right in that order
    gaze_tracks : dict
                  Key-value where keys are the point numbers and 
                  values are the coordinates for the points
    index : int
            A number to synchronize any delays in frames and bouning
            boxed. 
            Default: 0
    Notes
    -----
    Plays the video using opencv imshow function, this function is
    used when `DEBUG_MODE` is set, and the video played is of (1280,720)
    resolution for convenience.

    """
    cap = cv2.VideoCapture(videoName)

    while index < no_of_frames-1:
        ret, orig_frame = cap.read()
        index += 1

        for p in gaze_tracks:
            gaze_point = (int(float(gaze_tracks[p][index][0]) *
                              normFactorX +
                              float(gazeXOffset)),
                          int(float(gaze_tracks[p][index][1]) *
                              normFactorY +
                              float(gazeYOffset)))
            cv2.circle(orig_frame,
                       gaze_point,
                       color=(0, 255, 0),
                       radius=5,
                       thickness=6)

        cv2.rectangle(orig_frame,
                      (int(rectangle[index][0]),
                       int(rectangle[index][1])),
                      (int(rectangle[index][2]), int(rectangle[index][3])),
                      (0, 0, 255),
                      2)

        cropped_frame = cv2.resize(orig_frame, (int(1.7779*740), 740))
        frame_text = 'Frame : '+str(index)
        shot_text = 'Shot : '+final_track[index]
        cv2.putText(cropped_frame, frame_text,
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        cv2.putText(cropped_frame, shot_text,
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, 255)
        # cropped_frame = cv2.resize(orig_frame, (1920,1080))
        cv2.imshow('video', cropped_frame)

        # temp
        # out.write(cropped_frame)

        if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
            break
        pass
    cap.release()
    # out.release()
    cv2.destroyAllWindows()
    pass

def display_cost(frame_number, matrix): return {key:
                                                matrix[key][frame_number]
                                                for key in matrix}

if __name__ == '__main__':
    # will return the unary_cost values at a particular frame number
    # given in argument

    print(Fore.WHITE+'Loading config file : '+CONFIG_FILE)
    loadConfig(CONFIG_FILE)
    print(Fore.GREEN+'Done')

    DEBUG_MODE and print(Fore.CYAN+'Running in Debug Mode')
    VERBOSE and print(Fore.CYAN+'Running in Verbose Mode')
    COST_PLOT and print(Fore.CYAN+'Costs incurred graph will be plotted')

    if DEBUG_MODE:
        print(Fore.WHITE+'Video : '+videoName)
        print('Actors : '+str(actors))
        printParameters()


    gaze_tracks = loadGazeTracks()
    shot_tracks = loadShotTracks()
    frames = loadFrames()

    min_shot_length = min([len(list(shot_tracks[key]))for key in shot_keys])
    no_of_frames = min(len(frames), min_shot_length)
    no_of_shots = len(list(shot_tracks.keys()))

    final_track = ['']*no_of_frames

    # INITIALIZING UNARY COST MATRIX
    zeros_list = np.zeros(no_of_frames)

    # no. of shots * no of frames matrix
    cost = {key: list(zeros_list) for key in shot_keys}

    #REDUNDANT
    # cost_breakup_matrix = [{key:
    #                         {k: np.zeros(4) for k in shot_keys}
    #                         for key in shot_keys}
    #                        for i in range(no_of_frames)]

    # DYNAMIC COSTS INITIALISATION
    dp_cost = {key: list(np.zeros(no_of_frames)) for key in shot_keys}
    back_track = {key: [-1]*no_of_frames for key in shot_keys}
    shot_duration = {key: list(np.zeros(no_of_frames)) for key in shot_keys}

    frames_in_range = no_of_frames

    # FRAME LOOP STARTS HERE
    # ADDING UNARY COST FOR A FRAME

    print(Fore.WHITE+'Computing Unary costs...')

    for present_frame in range(frames_in_range):
        g = calculateGazeCost(present_frame, shot_tracks, gaze_tracks)
        for key in shot_keys:
            cost[key][present_frame] = g[key]
        pass

    # NORMALISING GAZE COST
    maxi = -10000000
    maxi = np.max([np.max(cost[key]) for key in cost])

    cost = {key: np.subtract(maxi, cost[key]) for key in cost}

    # to normalize other costs
    cost_norm_factor = np.max([np.max(cost[key]) for key in cost])

    initialiseGlobalMatrices()

    # ADDING STATIC COST (COSTS)
    # for present_frame in range(establishing_shot_time, frames_in_range):
    #     s = addStaticCosts(present_frame)
    #     for key in shot_keys:
    #         cost[key][present_frame] += s[key]

    print(Fore.GREEN+'Done')
    print(Fore.WHITE+'Computing dynamic costs...')
    # ADDING DYNAMIC COSTS
    for present_frame in range(establishing_shot_time, frames_in_range):

        VERBOSE and print(Fore.RED+"Frame : ", present_frame)
        addDynamicCosts(present_frame)
        VERBOSE and print(Fore.YELLOW+"Unary Cost : " +
                             str(display_cost(present_frame, cost)))
        VERBOSE and print(Fore.MAGENTA+"Dp Cost : " +
                             str(display_cost(present_frame, dp_cost)))
        VERBOSE and print(Fore.CYAN+"back_track : " +
                             str(display_cost(present_frame, back_track)))
        VERBOSE and print(Fore.WHITE+"shot_duration : " +
                             str(display_cost(present_frame, shot_duration)))

    print(Fore.GREEN+'Done')

    last_state = [key
                  for key in dp_cost
                  if (dp_cost[key][no_of_frames-1] ==
                      min(display_cost(
                          no_of_frames-1,
                          dp_cost).values()))][0]
    # BACKTRACKING
    final_track[frames_in_range-1] = last_state

    for frame in range(frames_in_range-2, establishing_shot_time-1, -1):
        final_track[frame] = back_track[final_track[frame+1]][frame]

    for frame in range(establishing_shot_time, -1, -1):
        # final track shots until establishing shot, these are dummy,
        # and actualy track will be replaced by wide shot
        final_track[frame] = final_track[frame+1]

    VERBOSE and print([str(idx)+':'+shot
                          for idx, shot in enumerate(final_track)])

    if COST_PLOT:
        fig = plt.figure(figsize=(20,20))
        ax = fig.subplots()
        ax = plt.subplot(321)
        
        #transition cost
        ax.set_title("Min Shot Duration Cost")
        x = np.linspace(0, no_of_frames,2000)
        y = [analyticMinShotDurationFunction(t) for t in x]
        cnf = [cost_norm_factor for t in x]
        plt.plot(x, y, label='Transition cost')
        # plt.plot(x, cnf, 'r-', label='Cost norm Factor')
        # plt.legend()

        ax = plt.subplot(322)
        # rythm cost
        ax.set_title("Rythm Cost")
        x = np.linspace(0, no_of_frames,2000)
        y = [analyticRythmFunction(t) for t in x]
        cnf = [cost_norm_factor for t in x]
        plt.plot(x, y, label='Rythm Cost')
        # plt.plot(x, cnf, 'r-', label='Cost norm Factor')
        plt.legend()

        ax = plt.subplot(323)
        # overlap cost
        ax.set_title("Overlap Potential")
        x = np.linspace(0, 1, 200)
        y = [analyticOverlapFunction(t) for t in x]
        cnf = [cost_norm_factor for t in x]
        plt.plot(x, y, label='Overlap potential')
        plt.xlabel('Overlap ratio')
        plt.ylabel('Overlap potential')
        # plt.plot(x, cnf, 'r-', label='Cost norm Factor')
        # plt.legend()

        ax = plt.subplot(324)
        # total dpcost
        ax.set_title("DP Cost")
        x = np.arange(0, frames_in_range)

        for idx, key in enumerate(shot_keys):
            plt.plot(x,
                     np.asarray(dp_cost[key][:frames_in_range]),
                     c=np.random.rand(3),
                     label=key)
        # plt.legend()
        ax = plt.subplot(325)
        ax.set_title("Gaze potential")
        x = np.arange(0, frames_in_range)

        for idx, key in enumerate(shot_keys):
            plt.plot(x,
                     np.asarray(cost[key][:frames_in_range]),
                     c=np.random.rand(3),
                     label=key)
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Gaze potential')
        cnf = [cost_norm_factor for t in x]
        # plt.plot(x, cnf, 'r-', label='Cost norm Factor')
        plt.show()

    cropped_window = generateCroppedWindow(final_track)

    if DEBUG_MODE:
        playWithGaze(video[videoName], cropped_window, gaze_tracks)
        # ut.cropped_play(video[videoName], cropped_window)
        # renderVideoFromCroppedWindow(video[videoName],
        #                              cropped_window,
        #                              output_video_name)
    else:
        renderVideoFromCroppedWindow(video[videoName],
                                     cropped_window,
                                     output_video_name)
