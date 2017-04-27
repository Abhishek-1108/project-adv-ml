from collections import defaultdict
import pickle
from filter_ontology import parse_ontology


def get_labels_for_ids(labels_file, whitelist):
    with open(labels_file) as infile:
        lines = infile.readlines()

    label_map = {}
    for line in lines:
        if line.startswith('#'):
            continue

        split_line = line.split(', ')
        ytid = split_line[0]
        labels = split_line[3].replace('"', '').strip().split(',')
        labels = [lb for lb in labels if lb in whitelist]
        label_map[ytid] = labels[0]

    return label_map


def main():
    # id_file_path = '/proj/pls_audioset_experiments/' + 'ids.pkl'
    labels_path = '/exp/filtered.20.overlap.csv'
    ontology_file = '/proj/ontology.json'
    ontology, _, _ = parse_ontology(ontology_file)
    reverse_ontology = {}
    for k, v in ontology.items():
        reverse_ontology[v] = k

    filter_list = [u'Blender', u'Jingle bell', u'Chopping (food)', u'Vacuum cleaner',
                   u'Arrow', u'Whip', u'Coin (dropping)', u'Slap, smack',
                   u'Gong', u'Chop', u'Crushing', u'Whistling', u'Singing bowl',
                   u'Change ringing (campanology)', u'Car passing by', u'Tuning fork',
                   u'Whistle', u'Whoosh, swoosh, swish', u'Hammer', u'Stomach rumble']
    whitelist = set([])
    for label in filter_list:
        whitelist.add(reverse_ontology[label])

    print('Labels in whitelist: ({}) {}'.format(len(whitelist), whitelist))

    label_map = get_labels_for_ids(labels_path, whitelist)
    print('Label map length: {}'.format(len(label_map.items())))

    with open('/proj/pls_audioset_experiments/label_map.pkl', 'wb') as outfile:
        pickle.dump(label_map, outfile)


if __name__ == '__main__':
    main()
