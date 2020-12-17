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
# import automatic_editing_lib as editlib
from pathlib import Path
import matplotlib.pyplot as plt
# from configparser import ConfigParser, ExtendedInterpolation
from colorama import init, Fore, Style, Back
import editing as ed
init()
print(Style.BRIGHT)


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

def computeGazeCostforFrame(shot_tracks,gaze_tracks,frame_number):
    gaze_cost = {key:0 for key in shot_keys}
    
    ## GAZE FOR ONE SHOTS
    one_shots = ed.getAllNShots(1)
    for shot in one_shots:
        centerX = 0.5*(shot_tracks[shot][frame_number][0] + \
            shot_tracks[shot][frame_number][2])
        centerY = 0.5*(shot_tracks[shot][frame_number][1] + \
            shot_tracks[shot][frame_number][3])

        center_point = [centerX,centerY]
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
        n_shots = ed.getAllNShots(i)
        cost_for_present_shot = 0

        for n_shot in n_shots:
            lower_n_shots = ed.getAllNShots(i-1)
            lower_n_shots = [s
                            for s in lower_n_shots
                            if ed.containedActors(s, n_shot)]
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

# def computeRythmCost(pres_shot,previous_frame,prev_shot):
def computeRythmCost(previous_frame,prev_shot):
    time = shot_duration[previous_frame][prev_shot]
    # time = shot_duration[pres_shot][previous_frame][prev_shot]
    #return time/1000 
    return analyticRythmCost(time,cost_norm_factor)
    pass

def analyticRythmCost(time,cost_norm_factor,variance=0.82):
    if(time==0):
        return 100*cost_norm_factor
    y = np.exp((-(np.log(time)-np.log(fps*rythm_factor))**2/2*(variance**2)))
    rythm_cost = -100*cost_norm_factor*y + 100*cost_norm_factor
    return rythm_cost
    pass


def play(video_path,rectangle,gaze_tracks):

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

        frame = cv2.resize(orig_frame,(1066,600))
        frame_text = 'Frame : '+str(index)
        shot_text = 'Shot : '+final_track[index]
        cv2.putText(frame, frame_text,
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.putText(frame, shot_text,
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.imshow('mat',frame)
        if(cv2.waitKey(int(1000/fps))&0xff==ord('q')):
            vidcap.release()
            break
            cv2.destroyAllWindows()
        index +=1

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
    out = cv2.VideoWriter('../Outputs/'+videoName+'/' +
                          'temp.mp4', fourcc, framerate, (1920, 1080))

    cap = cv2.VideoCapture(input_video)
    # index=5 #offset to sync coordinate shot track and video

    print('Rendering Video...')

    while index in range(0,no_of_frames):
    # for index in range(0,no_of_frames):
        ret, orig_frame = cap.read()
        # orig_frame = cv2.imread(basedir+'/video_frames/'+frames[index])
        index += 1

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
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))
            cv2.putText(cropped_frame, shot_text,
                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0))

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
        print(str('%0.2f' % percentage), end='\r')

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # adding audio

    sys.stdout.write('\r')
    s = '100 : Done'
    sys.stdout.write(s)

    if(audio[videoName] != 'NA'):
        output_video_name = ed.addAudioToVideo('../Outputs/'+videoName+'/'+'temp.mp4',
                                            audio[videoName],
                                            output_video_name)
    else:
        shell_command = 'rename'+'../Outputs/' + \
            videoName+'/'+'temp.mp4 '+output_video_name
        os.system(shell_command)

    print(Fore.GREEN+'\nRendered Video : '+output_video_name)
    video_stats = os.stat(output_video_name)
    video_size = float(video_stats.st_size)/(1024*1024)
    print(Fore.GREEN+'Size : '+str('%2f' % video_size)+'M')
    print(Fore.GREEN+'Resolution : ' + str(op_resolution_w) +
          'x'+str(op_resolution_h))
    print(Fore.GREEN+'Audio file : '+audio[videoName])

    ed.printParameters()
    pass

def addDynamicCosts(dp_cost,gaze_cost,shot_tracks,present_frame,backtrack):
    # present_frame is to be computed
    # previous_frame has been computed

    previous_frame = present_frame-1
    if(present_frame==0):
        for key in shot_keys:
            dp_cost[frame][key] = gaze_cost[frame][key]
            backtrack[frame][key] = key
            for k in shot_keys:
                # shot_duration[key][present_frame][key] = 1
                shot_duration[present_frame][key] = 1
    else:
        for pres_shot in shot_keys:
            min_cost = np.inf
            update_cost = {key:0 for key in shot_keys}
            for prev_shot in shot_keys:
                #gaze
                update_cost[prev_shot] = dp_cost[previous_frame][prev_shot]+gaze_cost[present_frame][pres_shot]

                #rythm
                # update_cost[prev_shot] += computeRythmCost(pres_shot,previous_frame,prev_shot)
                
                update_cost[prev_shot] += computeRythmCost(previous_frame,prev_shot)

                #transition cost
                # update_cost[prev_shot] += transition_cost_matrix[prev_shot][prev_shot]
                    
                print("gaze_cost: ",gaze_cost[present_frame][pres_shot],"Rythm: ",computeRythmCost(previous_frame,prev_shot),"DP_prev:",dp_cost[previous_frame][prev_shot],pres_shot,prev_shot,update_cost[prev_shot],)

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

            dp_cost[frame][pres_shot] += min_cost
    pass

#--------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------

ed.loadConfig('configs/dos6_all.ini')
DEBUG_MODE = ed.DEBUG_MODE
basedir = ed.basedir
width = ed.width
height = ed.height
actors = ed.actors
videoName = ed.videoName
video_path = ed.video[videoName]
fps = ed.fps
rythm_factor = 3

## LOADING GAZE TRACK
gaze_tracks = ed.loadGazeTracks()

## LOADING FRAMES
frames = ed.loadFrames()

## LOADING SHOTS
shot_tracks = ed.loadShotTracks()
shot_keys = ed.shot_keys

normFactorX = ed.normFactorX
normFactorY = ed.normFactorY
gazeXOffset = ed.gazeXOffset
gazeYOffset = ed.gazeYOffset

min_shot_length = min([len(list(shot_tracks[key]))for key in shot_keys])
no_of_frames = min(len(frames), min_shot_length)



gaze_cost = [{key:0 for key in shot_keys} for i in range(0,no_of_frames)]
dp_cost = [{key:0 for key in shot_keys} for i in range(0,no_of_frames)]
backtrack = [{key: '' for key in shot_keys} for i in range(0, no_of_frames)]
# shot_duration = {k:[{key: 0 for key in shot_keys} for i in range(0, no_of_frames)] for k in shot_keys}
shot_duration = [{key: 0 for key in shot_keys} for i in range(0, no_of_frames)]
final_track = ['']*no_of_frames

frames_in_range = no_of_frames

## COMPUTING GAZE COST FOR EACH FRAME
for frame in range(0,frames_in_range):
    g = computeGazeCostforFrame(shot_tracks,gaze_tracks,frame)    
    for key in shot_keys:
        gaze_cost[frame][key] = g[key]

    # print(shot_keys[max_gaze_shot])

## COMPUTE NORMALISATION COST
print('Computing normalisation factor')
max_gaze = np.max([np.max([gaze_cost[frame][key] for key in shot_keys]) for frame in range(0,no_of_frames)])
gaze_cost = [{key:max_gaze-gaze_cost[frame][key] for key in shot_keys} for frame in range(0,no_of_frames)]
cost_norm_factor = max_gaze = np.max([np.max([gaze_cost[frame][key] for key in shot_keys]) for frame in range(0, no_of_frames)])

transition_cost_matrix = {k:{key:0.1 for key in shot_keys}for k in shot_keys}


## ADDING RYTHM COST
for frame in range(0,10):
    addDynamicCosts(dp_cost,gaze_cost,shot_tracks,frame,backtrack)
    print(Fore.RED+'Frame: '+str(frame))
    print(Fore.WHITE)
    print([shot_duration[frame][key] for key in shot_keys])

exit()
last_frame_shot = shot_keys[np.argmin([dp_cost[no_of_frames-1][key] for key in shot_keys])]
final_track[frames_in_range-1] = last_frame_shot

for frame in range(frames_in_range-2,-1, -1):
    final_track[frame] = backtrack[frame][final_track[frame+1]]


## THIS PART COMPUTES ALL OTHER POSSIBLE TRACKS, BUT FIX SHOT DURATION FIRST
# f = -1*np.ones((len(shot_keys),frames_in_range))
# print(f.shape)
# for idx,key in enumerate(shot_keys):
#     f[idx,frames_in_range-1] = idx

# for idx,key in enumerate(shot_keys):
#     for frame in range(frames_in_range-2, -1, -1):
#         f[idx,frame] = shot_keys.index(backtrack[frame][shot_keys[int(f[idx,frame+1])]])

# x = np.arange(0,no_of_frames)
# for idx, key in enumerate(shot_keys):
#     plt.plot(x,np.asarray(f[idx,:]),c=np.random.rand(3),label=key)
# # plt.legend()
# plt.xlabel('Frames')
# plt.ylabel('shots')
# plt.show()

# for frame in range(0,frames_in_range):
#     min_gaze_shot = np.argmin([gaze_cost[frame][key] for key in shot_keys])
    # final_track[frame] = shot_keys[min_gaze_shot]
    # max_gaze = -100000000
    # for key in shot_keys:
    #     if(max_gaze<gaze_cost[frame][key]):
    #         max_gaze_shot = key
    #         max_gaze = gaze_cost[frame][key]
    # final_track[frame] = max_gaze_shot

print(final_track)

# shot_duration_list = []
# shot = final_track[0]
# count = 0
# for frame in range(1,frames_in_range):
#     if(final_track[frame]==shot):
#         count+=1
#     else:
#         d = {shot:count/fps}
#         shot_duration_list.append(d)
#         shot = final_track[frame]
#         count = 1
# d = {shot:count/fps}
# shot_duration_list.append(d)


# print(shot_duration_list)

cropped_window = [shot_tracks[shot][frame] for frame,shot in enumerate(final_track)]

play(video_path,cropped_window,gaze_tracks)
# renderVideoFromCroppedWindow(video_path,cropped_window,'music_intro_gaze_only')
