# project-adv-ml
adv. ml project

## download files
- `mkdir downloads`
- `cut -f 1 -d ',' filtered.20.overlap.csv > downloads/vids_to_download.txt`
- `youtube-dl -f mp4 -a vids_to_download.txt --id --ignore-errors &> download.log &`

## pre-requisites
- `https://github.com/jayrambhia/Install-OpenCV`
- if `import cv2` fails, run `sudo ln /dev/null /dev/raw1394`
