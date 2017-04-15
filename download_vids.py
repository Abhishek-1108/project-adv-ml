def download_from_file(filepath, start=1, end=100):
    with open(filepath) as infile:
        lines = infile.readlines()

    for line in lines[start:end]:
        print(line)

if __name__ == '__main___':
    download_from_file('/proj/balanced_train_segments.csv')
