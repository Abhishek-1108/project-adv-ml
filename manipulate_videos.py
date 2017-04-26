from glob import glob
import os

import cv2
import numpy as np


def make_dir_if_not_exist(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def download_from_file(filepath, output_dir, start=1, end=100):
    with open(filepath) as infile:
        lines = infile.readlines()

    for line in lines[start:end]:
        if line.startswith('#'):
            continue
        
        split_line = line.split(',')
        vid = split_line[0]


def seconds_to_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    representation = "%02d:%02d:%02d" % (h, m, s)
    return representation


def cut_videos(vid_segments_file, downloads_dir, destination_dir):
    segments = {}  # ytid: (start, end)
    with open(vid_segments_file) as infile:
        lines = [l for l in infile]

    # parse file
    for line in lines:
        if line.startswith('#'):
            continue

        split_line = line.split(',')
        ytid, start, end = split_line[0], split_line[1], split_line[2]
        segments[ytid] = (start, end)

    # use downloads to create frames
    for f in glob(downloads_dir + '/*.mp4'):
        dest_path = f.replace(downloads_dir, destination_dir)

        ytid = os.path.splitext(os.path.basename(f))[0]
        dest_path = os.path.join(dest_path, ytid)
        make_dir_if_not_exist(dest_path)

        start_time = int(float(segments[ytid][0].strip()))
        end_time = int(float(segments[ytid][1].strip()))

        full_filepath = os.path.join(downloads_dir, f)
        write_frames_to_dir(full_filepath, dest_path, start_time, end_time)


def write_frames_to_dir(vid_file, dest_dir, start_time, end_time):
    # minimum and maximum fps was ~6 and ~30 respectively 
    # for the first set of filtered videos
    # lets average the frames after every 0.5 seconds (approx.)
    vid = cv2.VideoCapture(vid_file)
    fps = round(vid.get(cv2.CAP_PROP_FPS), ndigits=0)
    print('fps', fps)
    window_size = int(fps / 2)
    print('window_size', window_size)

    start_frame_count = fps * start_time
    stop_frame_count = fps * end_time

    frame_number = -1
    averaged_frame_count = -1
    window_count = 0
    window = []
    while True:
        read_success, frame = vid.read()
        if read_success:
            frame_number += 1

            # video segment of interest starts here
            if start_frame_count < frame_number:
                if frame_number < stop_frame_count:
                    window_count += 1
                    window.append(frame)
                    # if our window is full, average it and flush out
                    if window_count == window_size:
                        window_average = np.average(np.asarray(window), axis=0)
                        averaged_frame_count += 1
                        dest_path = '{}/{}.jpg'.format(dest_dir, averaged_frame_count)
                        cv2.imwrite(dest_path, window_average)
                        window = []
                        window_count = 0
                else:
                    # video segment of interest has ended
                    # average and flush out any remaining part of the window
                    if len(window) > 0:
                        window_average = np.average(np.asarray(window), axis=0)
                        averaged_frame_count += 1
                        dest_path = '{}/{}.jpg'.format(dest_dir, averaged_frame_count)
                        cv2.imwrite(dest_path, window_average)
        else:
            return


def create_frames(video_src_dir, dest_frames_dir):
    i = 0
    for f in glob(video_src_dir + '/*.mp4'):
        ytid = os.path.splitext(os.path.basename(f))[0]
        this_dest_dir = os.path.join(dest_frames_dir, ytid)
        make_dir_if_not_exist(this_dest_dir)
        write_frames_to_dir(vid_file=f, dest_dir=this_dest_dir)
        i += 1
        if i % 10 == 0:
            print('frames for {} files created'.format(i))

    print('frames were created for a total of {} files'.format(i))


def main():
    vid_segments_info_file = os.getenv('filtered_path')

    downloads_dir = os.getenv('downloads_dir')
    # segments_dir = os.getenv('segments_dir')
    # make_dir_if_not_exist(segments_dir)
    frames_dir = os.getenv('frames_dir')
    make_dir_if_not_exist(frames_dir)
    cut_videos(
        vid_segments_file=vid_segments_info_file,
        downloads_dir=downloads_dir,
        destination_dir=frames_dir
    )

    # create_frames(
    #     video_src_dir=downloads_dir,
    #     dest_frames_dir=frames_dir
    # )


if __name__ == '__main__':
    main()
