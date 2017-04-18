from collections import defaultdict

import ipdb
import json
import os


def parse_ontology(ontology_path):
    with open(ontology_path) as o:
        full_definition = json.load(o)
    # only interested in {id: name} map
    ontology = {}
    children = {}
    parent = {}
    for item in full_definition:
        ontology[item['id']] = item['name']
        children[item['id']] = item.get('child_ids', None)
        for ch in item.get('child_ids', []):
            parent[ch] = item['id']

    return ontology, children, parent


def roll_up_to_parent(labels, parent_map):
    result = []
    for label in labels:
        while parent_map.get(label, None):
            label = parent_map[label]
        
        result.append(label)

    return result


def analyse_file(filepath, ontology, children, parent):
    with open(filepath) as infile:
        lines = infile.readlines()

    label_counts = defaultdict(int)
    label_overlaps = defaultdict(lambda :defaultdict(int))
    for line in lines:
        if line.startswith('#'):
            continue
        
        split_line = line.split(', ')
        labels = split_line[3].replace('"', '').strip().split(',')
        # label_ids = roll_up_to_parent(label_ids, parent)

        for label in labels:
            label_counts[label] += 1

        for x in labels:
            for y in labels:
                if x == y:
                    continue
                label_overlaps[x][y] += 1

    # labels occurring the most
    print('labels occuring the most')
    sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    print([ontology[t[0]] for t in sorted_labels[:20]])
    # labels least occuring with others
    print('labels with least overlap')
    sorted_overlaps = sorted(label_overlaps.items(), key=lambda x: sum(x[1].values()))
    print([ontology[t[0]] for t in sorted_overlaps[:20]])

    print('')
    print('possible selection, with counts')
    top_labels = [i[0] for i in sorted_overlaps[:20]]
    print('ids:')
    print(top_labels)
    print('names:')
    print([ontology[t] for t in top_labels])
    print([label_counts[x] for x in top_labels])
    
    filtered_lines = []
    for line in lines:
        if any(label in line for label in top_labels):
            filtered_lines.append(line)

    ipdb.set_trace()
    return filtered_lines

def main():
    ontology_path = os.getenv('ontology_path')
    train_segments_path = os.getenv('train_segments_path')
    output_path = os.getenv('filtered_path')
    ontology, children, parent = parse_ontology(ontology_path)
    filtered_lines = analyse_file(train_segments_path, ontology, children, parent)
    with open(output_path, 'w') as outfile:
        outfile.writelines(filtered_lines)


if __name__ == '__main__':
    main()