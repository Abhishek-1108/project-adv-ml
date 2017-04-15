from glob import glob
import os


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
        

def main():
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
        destination_dir=segments_dir    
    )

if __name__ == '__main__':
    main()
