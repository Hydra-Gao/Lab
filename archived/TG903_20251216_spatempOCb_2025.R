## Make sure you run this first:Y
source("01_packages_and_functions.R")

## NOTE: several packages have name conflicts. Therefore, you will see a lot
## of explicit naming of functions (package::function(stuff)) below, especially
## for dplyr.

#### Session metadata ####
## Session-specific metadata that you NEED to edit or check:
# Location in which data that have been processed by phy(llum) are stored
path <- "./"

# Head-fixed stimulus settings
resol <- 1920
vis_angle <- 84
refresh <- 165

# Additional metadata
animal <- "tg903"
session <- "2025-12-17_spatemp_left_after"
day <- "2025.12.17"

# stereotax_correction: the amount by which the polar coordinate system of the
# stimlog Direction column must be rotated counterclockwise to correct for the
# non-horizontal orientation of the retinocentric temporal-nasal plane. Should
# generally be a positive number less than 90.
stereotax_correction <- 0

## Define new directories where analysis output will be placed
## Don't edit unless you understand what these mean
analysis_output_dir <- file.path(path, "analysis_output")
rap_sheets_dir <- file.path(analysis_output_dir, "rap_sheets")
## Create the directories (if they don’t exist)
dir.create(analysis_output_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(rap_sheets_dir, showWarnings = FALSE, recursive = TRUE)

#### IMPORT SPIKES DATA ####
## Regardless of whether spatemp and/or flights were recorded, the "spikes.csv"
## essentially works as the master record. We will first import it and deal with
## some basic cleaning steps.

# cluster_info: extended info on clusters, including shank location!
# events: timing of TTL pulses relative to neuropixels recording onset
# spikes: spike times, clusters, sites, depths, and widths

cluster_info <- 
  read_tsv(paste0(path,"cluster_info.tsv")) %>%
  mutate(clust_ids = cluster_id) %>% 
  dplyr::mutate(group = replace_na(group, "mua"))
events <- 
  read_csv(paste0(path,"events.csv"))
spikes_import <- 
  read_csv(paste0(path,"spikes.csv")) %>%
  # merge in the cluster info
  left_join(cluster_info)

## The following deals with a kilosort quirk that sometimes duplicates spikes 
## of the same cluster within the same millisecond bin. These duplicates need
## to be removed, but it is crucial do to so only for the single units ("good").
## MUAs and other units should not be touched.

## Quick check for duplicate bins for “good” rows only
dupes_in_ms <- 
  dtplyr::lazy_dt(spikes_import) %>% 
  dplyr::filter(group == "good") %>% 
  dplyr::mutate(ms_bin = base::floor(spike_time_sec * 1000)) %>% 
  dplyr::count(clust_ids, ms_bin, name = "spikes_in_bin") %>% 
  dplyr::filter(spikes_in_bin > 1) %>% 
  tibble::as_tibble()
nrow(dupes_in_ms) > 0 # If true, there are dupes

## Drop duplicate spikes inside each (cluster, ms) for “good” while leaving 
## other rows unchanged
good_dedup <- 
  dtplyr::lazy_dt(spikes_import) %>% 
  dplyr::filter(group == "good") %>% 
  dplyr::mutate(ms_bin = base::floor(spike_time_sec * 1000)) %>% 
  dplyr::arrange(clust_ids, ms_bin, spike_time_sec) %>% 
  dplyr::distinct(clust_ids, ms_bin, .keep_all = TRUE) %>% 
  dplyr::select(-ms_bin) %>%      # drop helper col
  dplyr::as_tibble()

other_rows <- 
  dtplyr::lazy_dt(spikes_import) %>% 
  dplyr::filter(group != "good") %>%
  dplyr::as_tibble()

spikes <- 
  dplyr::bind_rows(good_dedup, other_rows) %>% 
  tibble::as_tibble() %>%
  dplyr::arrange(spike_time_sec)

## If all worked, this should return TRUE
(base::NROW(spikes_import) - base::NROW(spikes)) ==
  (base::sum(dupes_in_ms$spikes_in_bin) - base::NROW(dupes_in_ms))


#### _Spatemp data ####
# You likely need to hand-edit the title of the stimlog file, since they
# vary substantially.
# stimlog: stimulus log file (made by psychopy or matlab script)
spatemp_stimlog <- 
  read_csv(paste0(path,"2025-12-17_tg908_HA2_upstairs_leftcorrect_spatemp.csv")) %>%
  ## We now need to correct for head angle in the stereotax. First, keep
  ## the original values in the csv as "Direction_recorded".
  rename(Direction_recorded = Direction) %>%
  ## Next, we subtract the offset angle from each value and then take the
  ## modulus against 360. This keeps all values between 0 and 360, inclusive.
  mutate(
    Direction = (Direction_recorded - stereotax_correction) %% 360
  )

#### STIMULUS TIMING ####
## We'll label all events (even those beyond full spatemp)
## These are based on the "events" channel being cross-referenced against Eric's
## lab notebook
full_spatemp_stim_df <- 
  tibble(
    event_start = events$event_time_sec,           
    event_end   = lead(events$event_time_sec)      
  ) %>% 
  mutate(
    event_start_hms   = format(.POSIXct(event_start, tz = "UTC"), "%H:%M:%S"),
    event_duration_sec = round(event_end - event_start, 4),
    event_duration_hms = if_else(
      is.na(event_duration_sec),
      NA_character_,
      format(.POSIXct(event_duration_sec, tz = "UTC"), "%H:%M:%S")
    )
  )

#### FULL SPATEMP PROCESSING ####
#### _Join data sets ####
## Resolve relative timing of stimlog and events
## 
## What follows in this section should generally work, so long as hitting
## the "record" button in openephys or spikeglx happened BEFORE you began 
## running the stimulus.
## 
## If you started the stimulus before you pressed record in the spike recording
## software, come talk to VBB.

# Clean the stimlog
# Custom function from script 01
# THIS ONLY WORKS WITH CERTAIN PSYCHOPY-MADE STIMULI
# Talk to VBB if this needs adjustment
cleaned_stimlog <- clean_stimlog(spatemp_stimlog,
                                 resol = resol,
                                 vis_angle = vis_angle,
                                 refresh = refresh)

# Extract motion onsets
motion_timing <-
  cleaned_stimlog %>%
  dplyr::filter(Stimulus_state == "moving")

# should be fully sampled: 5 replicates per Speed*Direction
motion_timing %>%
  dplyr::count(Speed, Direction, name = "replicates") %>%
  dplyr::arrange(Speed, Direction)

# Is motion_timing the same length as events?
nrow(motion_timing) == nrow(full_spatemp_stim_df)
# We assume each event corresponds to each TTL (which indicates motion onset)
# There are 121 such TTL events in these files

# When is the first motion stimulus?
first_motion_onset <- 
  motion_timing %>%
  dplyr::slice(1) %>% # Take first entry
  pull(Stimulus_start)

# Determine "recording offset" from events
# full_spatemp_stim_df$event_start[1] equates to motion_timing[1,1]
recording_offset <- full_spatemp_stim_df$event_start[1] - first_motion_onset

# Revise timing of stimlog based on events channel
updated_stimlog <-
  cleaned_stimlog %>%
  mutate(Stimulus_start = Stimulus_start + recording_offset, 
         Stimulus_end = Stimulus_end + recording_offset) %>%
  group_by(Speed, Direction) %>%
  mutate(
    Replicate = cumsum(
      Stimulus_state == "blank" & 
        (lag(Stimulus_state, default = "moving") == "moving")
    )
  ) %>%
  ungroup()

#### _Filter spikes ####
## Optionally extract out only the relevant time slice.
## Since there is nothing else in this recording, we'll leave it as is.
spikes_select <-
  spikes 

# Get the unique clust_ids
clust_ids <- unique(spikes_select$clust_ids)

#### _Argument adjustments ####
## Because of variation in stimuli, these need to be determined appropriately
## for what was actually displayed.

## EDIT THESE BY HAND
# Duration of motion epoch; assumed to be a constant
motion_duration <- 10
# When binning for MSR plot, specify bin width (in seconds):
bin_width <- 0.1   # 0.01 = 10 ms; 0.1 = 100 ms ...etc.

# Should rap sheets be exported?
export_plots <- TRUE

## These two should work automatically
# Number of expected sweeps of EACH direction*speed condition
# "Sweeps" and "Replicates" are synonymous in this context
n_sweeps <- updated_stimlog %>% select(Replicate) %>% max
# Is "blank" a stimulus condition?
blank_present <- "blank" %in% updated_stimlog$Stimulus_state


#### _Match stimlog and make rap sheets ####
source(file = "02_rap_sheets_maker_2025.R")


#### _Export labelled data ####
# Combine all the processed subsets into one data frame
labeled_spikes <- 
  bind_rows(result_list) %>%
  ungroup()
write_csv(labeled_spikes, 
          file = paste0(
            analysis_output_dir, "/", 
            animal, "_", session, "_fullspatemp_labeled_spikes.csv")
)

# Summary of data
all_summary_data <-
  bind_rows(polar_data) %>%
  ungroup() %>%
  dplyr::select(
    i, clust_ids, spikeDepths_updated, spikeWidths_updated,
    Speed, Direction, Replicate, vector_sum, si, inverse_CV, peak_count,
    APPD, PD, AP, mean_spike_rate, everything()) %>%
  arrange(i)
write_csv(all_summary_data, 
          file = paste0(
            analysis_output_dir, "/", 
            animal, "_", session, "_fullspatemp_all_summary_data.csv")
)

# Speed-specific summary of clusters (across Directions and Replicates)
cluster_summaries <-
  all_summary_data %>%
  ungroup() %>%
  dplyr::select(-c(Direction, Replicate)) %>%
  group_by(clust_ids, Speed) %>%
  summarize_all(mean) %>%
  dplyr::select(
    i, clust_ids, spikeDepths_updated, spikeWidths_updated,
    Speed, vector_sum, si, inverse_CV, peak_count,
    APPD, PD, AP, mean_spike_rate, everything()) %>%
  arrange(i) %>%
  ungroup() 
write_csv(cluster_summaries, 
          file = paste0(
            analysis_output_dir, "/", 
            animal, "_", session, "_fullspatemp_cluster_summaries.csv")
)


#### _Import labelled data ####
output_files <- 
  list.files(analysis_output_dir, full.names = TRUE)

cluster_summaries <- 
  read_csv(
    file = output_files[str_detect(
      output_files, "_fullspatemp_cluster_summaries\\.csv$")]
  )

all_summary_data <- 
  read_csv(
    file = output_files[str_detect(
      output_files, "_fullspatemp_all_summary_data\\.csv$")]
  )

labeled_spikes <- 
  read_csv(
    file = output_files[str_detect(
      output_files, "_fullspatemp_labeled_spikes\\.csv$")]
  )


#### _Cross-cluster analyses ####

#### __location and si by speed ####
loc_and_si <-
  cluster_summaries %>%
  drop_na(sh) %>%
  ggplot(aes(x = spikeWidths_updated, y = spikeDepths_updated, fill = si)) +
  geom_point(aes(size = si), pch = 21) +
  scale_fill_viridis_c(option = "B") +
  #coord_fixed() +
  #facet_wrap(~Speed, nrow = 1) +
  facet_grid(rows = vars(sh), cols = vars(Speed)) +
  xlab("probe width") +
  ylab("probe depth") +
  ggtitle(paste0(session, "_", animal)) +
  theme_bw()

pdf(
  file = paste0(
    analysis_output_dir, "/", session, "_", animal, "_location_and_si.pdf"
  ),
  width = 11, height = 8.5, bg = "white", pagecentre = TRUE, colormodel = "srgb"
)
## Now add the plot to the PDF simply by calling plot()
plot(loc_and_si)
## To declare an end to this PDF writing session, use `dev.off()`
dev.off()


#### __depth vs vector sum and si by speed ####
loc_and_vecsum <-
  cluster_summaries %>%
  drop_na(sh) %>%
  ggplot(aes(y = spikeDepths_updated, x = (vector_sum+180) %% 360, fill = si)) +
  geom_point(aes(size = si), pch = 21) +
  scale_fill_viridis_c(option = "B") +
  scale_x_continuous(breaks = c(90, 180, 270), labels = c(270, 0, 90)) +
  #coord_fixed() +
  #facet_wrap(~Speed, nrow = 1) +
  facet_grid(rows = vars(sh), cols = vars(Speed)) +
  ylab("probe depth") +
  xlab("vector sum preferred direction") +
  ggtitle(paste0(session, "_", animal)) +
  theme_bw()

pdf(
  file = paste0(
    analysis_output_dir, "/", session, "_", animal, "_location_and_vecsum.pdf"
  ),
  width = 11, height = 8.5, bg = "white", pagecentre = TRUE, colormodel = "srgb"
)
## Now add the plot to the PDF simply by calling plot()
plot(loc_and_vecsum)
## To declare an end to this PDF writing session, use `dev.off()`
dev.off()


#### __depth vs vector sum and si by speed ####
depth_and_msr <-
  cluster_summaries %>%
  drop_na(sh) %>%
  #dplyr::filter(Speed == 32) %>%
  ggplot(aes(y = spikeDepths_updated, x = mean_spike_rate, 
             fill = si
  )) +
  geom_point(aes(size = si), pch = 21) +
  scale_fill_viridis_c(option = "B") +
  #coord_fixed() +
  facet_grid(rows = vars(sh), cols = vars(Speed)) +
  ylab("probe depth") +
  ggtitle(paste0(session, "_", animal)) +
  theme_bw()

pdf(
  file = paste0(
    analysis_output_dir, "/", session, "_", animal, "_depth_and_msr.pdf"
  ),
  width = 11, height = 8.5, bg = "white", pagecentre = TRUE, colormodel = "srgb"
)
## Now add the plot to the PDF simply by calling plot()
plot(depth_and_msr)
## To declare an end to this PDF writing session, use `dev.off()`
dev.off()


