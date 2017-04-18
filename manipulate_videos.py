from glob import glob
import os

import cv2


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


def cut_videos(vid_segments_file, downloads_dir, destination_dir):
    segments = {}  # ytid: (start, end)
    with open(vid_segments_file) as infile:
        lines = [l for l in infile]

    for line in lines:
        if line.startswith('#'):
            continue

        split_line = line.split(',')
        ytid, start, end = split_line[0], split_line[1], split_line[2]
        segments[ytid] = (start, end)

    for f in glob(downloads_dir + '/*.mp4'):
        dest_path = f.replace(downloads_dir, destination_dir)
        ytid = os.path.splitext(os.path.basename(f))[0]
        start_time = segments[ytid][0]

        if os.path.exists(dest_path):
            print('skipped {}. segment exists.'.format(f))
        else:
            command = 'ffmpeg -ss {start} -i {input} -t 10 -c copy {output}'.format(
                start=start_time,
                input=f,
                output=dest_path
            )
            print(command)
            os.system(command)


def write_frames_to_dir(vid_file, dest_dir):
    vid = cv2.VideoCapture(vid_file)
    i = -1
    while True:
        read_success, frame = vid.read()
        if read_success:
            i += 1
            dest_path = '{}/{}.jpg'.format(dest_dir, i)
            cv2.imwrite(dest_path, frame)
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
    segments_dir = os.getenv('segments_dir')
    make_dir_if_not_exist(segments_dir)
    cut_videos(
        vid_segments_file=vid_segments_info_file,
        downloads_dir=downloads_dir,
        destination_dir=segments_dir    
    )

    frames_dir = os.getenv('frames_dir')
    create_frames(
        video_src_dir=segments_dir,
        dest_frames_dir=frames_dir
    )


if __name__ == '__main__':
    main()
