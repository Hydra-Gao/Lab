# -*- coding: utf-8 -*-
## Description
# Project sine wave gratings and/or dotfield stimuli to assess tuning to 
# rotational or translational optic flow across specified velocity values

# Last updated: 2025-12-04
# @author: ZH, modfied upon VBB prior codes, NOT TESTED WITH DOTFIELD!!! SINE-WAVE ONLY
# Psychopy version: 2022.2.4
# Python version: 3.8.10
# OS version: Windows 10 Pro 21H2 19044.2006

## Package import -- do not edit
from psychopy import visual, core, event
import numpy as np
import pandas as pd
import itertools
import glob
import os
import math
import random
import time
import serial

############################### THINGS TO EDIT #################################
# Set file number
filenumber = 1 # will append as 3-digit integer to file name (e.g., 1 -> "001")
time_of_day = "TG884_19.02_1st     " # change to nlx recording time

# Directory where the stimlog csv will be saved. Automatically resolves to where 
# the script is located and then a /test_runs folder within that dir
filepath = os.path.join(os.path.dirname(__file__), 'stimulus_logs')
# Alternatively, supply a filepath as follows:
#filepath = 'C:/path/to/directory'
#filepath = '/Users/thalassoma/Dropbox/code_scripts_examples/python_scripts/psychopy/2023-03-29 surya sine waves/test_runs' 

## Set head angle as measured from stereotax, in degrees
measured_head_angle = 90
TN_offset = 0 # calculated following assumption that the eye-beak plane is 25 degrees below the retino-centric TN plane (earth horizontal) during flight
# We'll use 90 - head_angle as the actual rotation (clockwise)
programmed_head_angle = 90 - measured_head_angle + TN_offset

## Key experimental parameters
total_replicates = 5 # integer, total number of sweeps of each stimulus
#grating_speeds = [4,32, 128, 256] # sine gratings, deg/s
grating_speeds = [32] # sine gratings, deg/s
#dotfield_speeds = [1024] 1
dotfield_speeds = None 
dotsize = 2.1 # visual angle size of each dot, in degrees
num_dots = 200 # number of on-screen dots

## Setup for screens and stimuli
# Screen resolution
screen_res =  (3840,1080) # or (1920,1080) or (3072,1920) etc (2560,1440)
aspect_ratio = (32, 9)
photodiode_window_size = [0,0]
vis_ang = 84 # visual angle at 30 cm away from screen
dotsize_px = (screen_res[0]/vis_ang) * dotsize

# Stimulus window
win_pos = visual.Window(
    monitor  = "ASUS", #"macbook", pre-register this!
    size = screen_res,
    pos=(0, 0),
    screen = 1,
    color = [1,1,1],
    units = 'pix',
    allowGUI = False,
    #waitBlanking = True,
    fullscr = False,
    # checkTiming = False
    )
# Vector of windows
windows = [win_pos] #[win_pos,win_neg] if more than one screen etc
#fps = win_pos.getActualFrameRate
fps = 144 #119.982 #59.951 #143.973 refresh rate in Hz
# Grating stimuli
# grat_pos = visual.GratingStim(
#     win = win_pos,
#     tex = "sin",
#     texRes = 1024,
#     interpolate = True,
#     units = "pix",
#     pos = (0,0),
#     # Set size to 2 times the width of the screen resolution
#     # This ensures that the screen is filled, and does not seem to distort
#     # the sf
#     size = screen_res[0]*2)
# grat_pos.autoLog = False  # Or we'll get many messages about phase change
#ZH : added for translational/rotational
grat_left = visual.GratingStim(
    win = win_pos,
    tex = "sin",
    texRes = 1024,
    interpolate = True,
    units = "pix",
    pos = (-screen_res[0]/4,0),  # ZH: for left vs right stimulus to be centered in their respective half
    size = ((screen_res[0]/.3), screen_res[1]*1.777)) # ZH: factor of 1.777 ensures vertical stim is filled
grat_right = visual.GratingStim(
    win = win_pos,
    tex = "sin",
    texRes = 1024,
    interpolate = True,
    units = "pix",
    pos = (screen_res[0]/4,0),
    size = ((screen_res[0]/2), screen_res[1]*1.777))
grat_left.autoLog  = False  # Or we'll get many messages about phase change
grat_right.autoLog = False
# Dotfield stim
dotf_pos = visual.DotStim(
    win = win_pos,
    nDots = num_dots*10,
    units = "pix",
    coherence = 1,
    signalDots = 'same',
    fieldSize = (screen_res[0]*5, screen_res[1]*5),
    dotSize = dotsize_px,
    dotLife = 5000, # eternity
    color = "black",) 
dotf_pos.autoLog = False  # Or we'll get many messages about phase change
# Blank stimuli
blank_pos = visual.Rect(
    win = win_pos,
    units = "pix",
    fillColor = 'white',
    size = screen_res)
blank_pos.autoLog = False  # Or we'll get many messages about phase change
# Photodiode window
photod_pos = visual.Rect(
    win = win_pos,
    pos = [-1*(screen_res[0]/2), -1*(screen_res[1]/2)], # bottom left corner
    units = "pix",
    fillColor = 'white',
    size = photodiode_window_size)

## Setup for directions and durations
desired_directions = [0, 90, 180, 270]
directions = [x+programmed_head_angle for x in desired_directions]
# durations should be integers
blank_duration  = 1 # in seconds
static_duration = 4 # in seconds
moving_duration = 6 # in seconds
state_type_vec = ["blank","static","moving"]
state_type_durations = [blank_duration, static_duration, moving_duration]
speed_factor = [-1, 1] # ZH: used to randomize the speed of left/right moving stimuli

# TTL pulse setup
ser = serial.Serial('COM4', write_timeout = 1)
print(ser.name)

################################# MAIN SCRIPT ##################################
# Things after this point (hopefully) won't need to be edited

## Check that either grating and/or dotfield speeds have been given
if not grating_speeds:
    if not dotfield_speeds:
        exit("ERROR: Speeds for gratings and dotfields cannot both be None")

## Set up csv saving into the correct directory
# Get current working directory
cwd = os.getcwd()
# Check against `filepath` and change dir if needed
if cwd != filepath:
    os.chdir(filepath)
else:
    pass
wd = os.getcwd()
print(wd)
# Get date
filedate = time.strftime("%Y-%m-%d")
# Get number into leading zero format
numled = str(filenumber).zfill(3)
# Before running stimulus, search for any files which match the `filename`
stimname =  filedate + '_' + time_of_day + '_' + 'spatemp' + '.csv'
file_present = glob.glob(stimname)
# Throw a warning if overwriting might occur and autoincrement
if file_present:
    print("WARNING: Stim log file with this name already exists! Automatically using garbage hex instead")
    numled = hex(random.getrandbits(16)) #str(filenumber+1).zfill(3)
    stimname =  filedate + '_' + time_of_day + '_' + 'spatemp' + '_' + numled + '.csv'
    print("New file name is ", stimname)
else:
    pass
# Return to original working directory if needed
newwd = os.getcwd()
if newwd != wd:
    os.chdir(wd)
    wd_two = os.getcwd()
    print(wd_two)
else:
    pass
    
pd.set_option("display.max_columns", None)
## Setup for grating velocities (i.e., phases) and SFs
# `phases` and `directions` much each be a list. Even if only one number is
# used, it must be wrapped in square brackets.
if grating_speeds:
    tfs = [math.sqrt(i/4) for i in grating_speeds] # TFs based on speed
    phases = [(i*3)/(fps*3) for i in tfs] # phases computed relative to refresh rate
    scw = screen_res[0] # screen width (1920)
    sfs = [math.sqrt(1/(4*i)) * (vis_ang/scw) for i in grating_speeds]
    pha_sf = pd.DataFrame(
                {'phases': [round(x,6) for x in phases], #phases
                 'sfs': [round(x,7) for x in sfs] #sfs
                })
    pha_sf_list = pha_sf.values.tolist()
    # Use itertools to make iterated sets against direction
    s = list(itertools.product(pha_sf_list, directions))

## Setup for dotfields
if dotfield_speeds:
    dot_pixperframe = [(i/fps) * (screen_res[0]/vis_ang) for i in dotfield_speeds]
    dot_df = pd.DataFrame(
    {'pixperframe': [round(x,6) for x in dot_pixperframe] #speeds
    })
    dot_df_list = dot_df.values.tolist()
    # Use itertools to make iterated sets against direction
    t = list(itertools.product(dot_df_list, directions)) 

## SET UP RANDOMIZATION OF STIMULI
## If there are gratings
if grating_speeds:
    ## But no dotfields
    if not dotfield_speeds:
        # Make total_replicates number of blocks, each of which is randomized
        phases_sfs_dirs_types_list = []
        for i in range(0, total_replicates):
            # Random shuffle
            d = random.sample(s, len(s))
            q = list(itertools.product(itertools.chain(d), state_type_vec))
            a = [(a, b, c, d) for ([a, b], c), d in q]
            phases_sfs_dirs_types_list = phases_sfs_dirs_types_list + a
        stimdat = pd.DataFrame(phases_sfs_dirs_types_list,columns=("GratingStim_phase", "GratingStim_sf","Stimulus_orientation", "Stimulus_state"))
        #ZH: added left/right direction
        type_map = {"blank":blank_duration, "static":static_duration, "moving":moving_duration}
        stimdat['Duration_sec'] = stimdat['Stimulus_state'].map(type_map)
        stimdat['Pattern'] = "gratings"
        stimdat['DotStim_speedpx'] = np.nan
        stimdat['Direction'] = stimdat['Stimulus_orientation'] - programmed_head_angle
        stimdat['Speed'] = 1/(4*(((screen_res[0]*stimdat['GratingStim_sf'])/vis_ang)**2))
        stimdat['Speed'] = [round(x) if x > 1 else round(x, 4) for x in stimdat['Speed']]
        stimdat['Left_speed_factor']  = [np.random.choice(speed_factor, size=len(directions))[0] for x in stimdat['Direction']]
        stimdat['Right_speed_factor'] = [np.random.choice(speed_factor, size=len(directions))[0] for x in stimdat['Direction']]
        stimdat = stimdat.reindex(sorted(stimdat.columns), axis=1)
        print(stimdat)
        print("dot speeds not supplied")
    ## Both gratings and dotfields
    else:
        ## Combine the two lists
        p = [(a, b, c) for ([a, b], c) in s]
        p_df = pd.DataFrame (p, columns = ['phases', 'sfs', 'directions'])
        p_df['Pattern'] = "gratings"
        p_df['Speed'] = 1/(4*(((screen_res[0]*p_df['sfs'])/vis_ang)**2))
        q = [(a, b) for ([a], b) in t]
        q_df = pd.DataFrame (q, columns = ['pixperframe', 'directions'])
        q_df['Pattern'] = "dotfield"
        q_df['Speed'] = (vis_ang*fps*q_df['pixperframe'])/screen_res[0]
        pq_df = pd.concat([p_df,q_df], axis=0, ignore_index=True)
        pql = pq_df.values.tolist()
        
        # Make total_replicates number of blocks, each of which is randomized
        phases_sfs_dirs_types_list = []
        for i in range(0, total_replicates):
            # Random shuffle
            d = random.sample(pql, len(pql))
            q = list(itertools.product(itertools.chain(d), state_type_vec))
            a = [(a, b, c, d, e, f, g) for ([a, b, c, d, e, f], g) in q]
            phases_sfs_dirs_types_list = phases_sfs_dirs_types_list + a
        stimdat = pd.DataFrame(phases_sfs_dirs_types_list,columns=("GratingStim_phase", "GratingStim_sf","Stimulus_orientation", "Pattern" , "Speed", "DotStim_speedpx", "Stimulus_state"))  
        type_map = {"blank":blank_duration, "static":static_duration, "moving":moving_duration}
        stimdat['Duration_sec'] = stimdat['Stimulus_state'].map(type_map)
        stimdat['Direction'] = stimdat['Stimulus_orientation'] - programmed_head_angle
        stimdat['Speed'] = [round(x) if x > 1 else round(x, 4) for x in stimdat['Speed']]
        stimdat = stimdat.reindex(sorted(stimdat.columns), axis=1)
        print(stimdat)
        print('both detected')
    
## If there are no gratings
if not grating_speeds:
    ## But if there are dotfields
    if dotfield_speeds:
        # Make total_replicates number of blocks, each of which is randomized
        phases_sfs_dirs_types_list = []
        for i in range(0, total_replicates):
            # Random shuffle
            d = random.sample(t, len(t))
            q = list(itertools.product(itertools.chain(d), state_type_vec))
            a = [(a, b, c) for (([a], b), c) in q]
            phases_sfs_dirs_types_list = phases_sfs_dirs_types_list + a
        stimdat = pd.DataFrame(phases_sfs_dirs_types_list,columns=("DotStim_speedpx", "Stimulus_orientation", "Stimulus_state"))

        type_map = {"blank":blank_duration, "static":static_duration, "moving":moving_duration}
        stimdat['Duration_sec'] = stimdat['Stimulus_state'].map(type_map)
        stimdat['Pattern'] = "dotfield"
        stimdat['GratingStim_phase'] = np.nan
        stimdat['GratingStim_sf'] = np.nan
        stimdat['Direction'] = stimdat['Stimulus_orientation'] - programmed_head_angle
        stimdat['Speed'] = (vis_ang*fps*stimdat['DotStim_speedpx'])/screen_res[0]
        stimdat['Speed'] = [round(x) if x > 1 else round(x, 4) for x in stimdat['Speed']]
        stimdat = stimdat.reindex(sorted(stimdat.columns), axis=1)
        print(stimdat)
        print("grating speeds not supplied")
    else:
        print('neither found')


## Create columns with NaNs to fill with times of stimulus events on each screen
# `Stimulus_` times will be relative to onset
# `UniversalTime_` will be the actual Unix time
Stimulus_start = pd.Series(np.nan, index=list(range(len(stimdat))), dtype="float")
Stimulus_end = pd.Series(np.nan, index=list(range(len(stimdat))), dtype="float")
UniversalTime_start = pd.Series(np.nan, index=list(range(len(stimdat))), dtype="float")
UniversalTime_end = pd.Series(np.nan, index=list(range(len(stimdat))), dtype="float")


## Initialize timers and variable containing monitor refresh rates
# Create a timer 
timer = core.Clock()
# Create a countdown timer (just a flip of the clock)
countDown = core.CountdownTimer()
# Create a trial clock
trialClock = core.Clock()

sessionStart = trialClock.getTime()
sessionUnixStart = int( time.time_ns() / 10 ** 6 ) # divide by 1 000 000 to convert from nanoseconds to milliseconds

## Generate and render stimuli
j = 0
keep_going = True
while keep_going is True:
    if j > len(stimdat)-1 :
        break
    for d in range(len(windows)):
        #windows[d].flip()
        d += 1
    for e in range(len(windows)):
        windows[e].setMouseVisible(False), # hide mouse cursor
        e += 1
    timer = core.Clock()
    countDown = core.CountdownTimer()
    direct = []
    duration = []
    phase = []
    sf = []
    pattern = []
    px_frame = []
    orientation = []
    Stimulus_state = []
    direct         = stimdat.loc[j,'Direction']
    duration       = stimdat.loc[j,'Duration_sec']
    phase          = stimdat.loc[j,'GratingStim_phase']
    sf             = stimdat.loc[j,'GratingStim_sf']
    pattern        = stimdat.loc[j,'Pattern']
    px_frame       = stimdat.loc[j,'DotStim_speedpx']
    orientation    = stimdat.loc[j,'Stimulus_orientation']
    Stimulus_state = stimdat.loc[j,'Stimulus_state']
    speed_f_L      = stimdat.loc[j,'Left_speed_factor']
    speed_f_R      = stimdat.loc[j,'Right_speed_factor']
    
    Stimulus_start[j] = trialClock.getTime()
    UniversalTime_start[j] = int(time.time_ns() / 10 ** 6)
    if Stimulus_state == "blank": 
        countDown.reset(duration)
        #countDown.add(duration)
        #total_frames = round(duration * fps)
        photod_pos.fillColor = 'white'
        win_pos.flip()
        while duration >= countDown.getTime() > 0:
            #if 0 <= timer.getTime() < duration:
            #for frame_n in range(total_frames):
                #if 0 <= frame_n < total_frames:
            blank_pos.draw()
            photod_pos.draw()
            win_pos.flip()
    elif Stimulus_state == "static":
        duration = duration + np.random.rand(1)
        countDown.reset(duration)
        #countDown.add(duration)
        timer.reset()
        if pattern == "gratings":
            # grat_pos.setPhase(0, '+')
            # grat_pos.setSF(sf)
            # grat_pos.setOri(orientation)
            grat_left.setPhase(0, '+')
            grat_right.setPhase(0, '+')
            grat_left.setSF(sf)
            grat_right.setSF(sf)
            grat_left.setOri(orientation)
            grat_right.setOri(orientation)
        elif pattern == "dotfield":
            dotf_pos.speed = 0
            dotf_pos.dir = orientation
        photod_pos.fillColor = 'white'
        win_pos.flip()
        while duration >= countDown.getTime() > 0:
            #if 0 <= timer.getTime() < duration:
            if pattern == "gratings":
                # grat_pos.phase += 0
                # grat_pos.draw()

                grat_left.phase  +=  0
                grat_right.phase +=  0
                grat_left.draw()
                grat_right.draw()
                photod_pos.draw()
                win_pos.flip()
            elif pattern == "dotfield":
                dotf_pos.speed += 0
                dotf_pos.draw()
                photod_pos.draw()
                win_pos.flip()
    elif Stimulus_state == "moving":
        countDown.reset(duration)
        #countDown.add(duration)
        ## Write to serial COM port
        ser.write(1)
        photod_pos.fillColor = 'red'
        if pattern == "gratings":
            grat_left.setPhase(phase, '+')
            grat_right.setPhase(phase, '+')
        elif pattern == "dotfield":
            dotf_pos.speed = px_frame
        #win_pos.flip()
        while duration >= countDown.getTime() > 0:
            #if 0 <= timer.getTime() < duration:
            if pattern == "gratings":
                # grat_pos.phase += phase
                # grat_pos.draw()
                # grat_left.phase += phase
                # grat_right.phase -= phase
                driftClock = core.Clock()
                t = driftClock.getTime()
                #left_dir  = speed_f_L * phase
                #right_dir = speed_f_R * phase
                # Opposing directions:
                if orientation == 0 or orientation == 180:
                    grat_left.phase  -= phase
                    grat_right.phase += phase
                    grat_left.draw()
                    grat_right.draw()
                else:
                    grat_left.phase  += phase
                    grat_right.phase += phase
                    grat_left.draw()
                    grat_right.draw()

                photod_pos.draw()
                win_pos.flip()
            elif pattern == "dotfield":
                #dotf_pos.speed += px_frame
                dotf_pos.draw()
                photod_pos.draw()
                win_pos.flip()
    else: 
        print("all loops bypassed")
        pass
    #win_pos.flip()
    Stimulus_end[j] = trialClock.getTime()
    UniversalTime_end[j] = int(time.time_ns() / 10 ** 6)
    timer.reset()
    countDown.reset()
    j += 1
    if j > len(stimdat)-1 :
        break
    keys = event.getKeys()
    if len(keys) > 0:
        keep_going = False
        ## Close connection to COM port
        ser.close()
for f in range(len(windows)):
    windows[f].close()
    f += 1

## Close connection to COM port (just in case)
ser.close()

## Get end time
sessionUnixEnd = int( time.time_ns() / 10 ** 6 )
sessionEnd = trialClock.getTime()-sessionStart

## Add time data to stimulus log data frame
stimdat['Stimulus_start']= Stimulus_start
stimdat['Stimulus_end']= Stimulus_end
stimdat['UniversalTime_start']= UniversalTime_start
stimdat['UniversalTime_end']= UniversalTime_end

## Drop any stimulus log rows whose time slots are empty
stimdat = stimdat.dropna(subset = ['Stimulus_start','Stimulus_end','UniversalTime_start', 'UniversalTime_end'])

## Add a row with the trial start time to the top of the data log
stimdat.loc[-1] = ['NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'NaN','NaN', 'NaN', 'NaN', 'NaN', 'INCEPTION', sessionStart, 'NaN', sessionUnixStart, 'NaN']  # adding a row
stimdat.index = stimdat.index + 1  # shifting index
stimdat.sort_index(inplace=True) 
#print(stimdat)

## Double-check the directory and write file
# Get current working directory
wd = os.getcwd()
#print(wd)
# Check it against specified save directory and change directory if need be
if wd != filepath:
    os.chdir(filepath)
else:
    pass
# Write the csv
stimdat.to_csv(stimname, index=False)
print("Exported as ", stimname)
# Check if directory was changed and if so, return to original working directory
nwd = os.getcwd()
if nwd != wd:
    os.chdir(wd)
else:
    pass

core.quit()