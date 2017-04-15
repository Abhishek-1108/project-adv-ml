from glob import glob
import os


def make_dir_if_not_exist(dirpath):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


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
        split_line = line.split(',')
        ytid, start, end = split_line[0], split_line[1], split_line[2]
        segments[ytid] = (start, end)

    for f in glob(downloads_dir + '/*.mp4'):
        dest_path = f.replace(downloads_dir, segments_dir)
        print(dest_path)


if __name__ == '__main__':
    input_file = '/proj/balanced_train_segments.csv'
    output_dir = '/proj/vids'
    make_dir_if_not_exist(output_dir)
    download_from_file(input_file, output_dir)

    downloads_dir = '/exp/downloads'
    segments_dir = '/exp/segments'
    make_dir_if_not_exist(segments_dir)
    cut_videos(
        vid_segments_file=input_file,
        downloads_dir=downloads_dir,
        segments_dir=segments_dir    
    )
