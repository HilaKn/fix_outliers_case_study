import gensim
from collections import defaultdict
import argparse
import numpy as np
from os import listdir
from os.path import isfile, join
from nltk.stem import WordNetLemmatizer
import os
import operator
import sys

OUTLIER_LABEL = -1
MIN_COUNT = 2

def get_matching_nouns_from_patterns():
    noun_to_nouns = defaultdict(lambda: defaultdict(int))
    pattern_files = [f for f in listdir(args.patterns_folder) if isfile(join(args.patterns_folder, f))]
    lemmatizer = WordNetLemmatizer()
    for file in pattern_files:
        file_path = os.path.join(args.patterns_folder, file)
        with open(file_path) as f:
            for row in f:
                data = row.rstrip().split('\t')
                try:
                    noun_1 = lemmatizer.lemmatize(data[0])
                    noun_2 = lemmatizer.lemmatize(data[1])
                except :
                    print "Unexpected error:", sys.exc_info()[0]
                    print "skipping nouns"
                    continue
                count = int(data[4])
                noun_to_nouns[noun_1][noun_2] += count
                noun_to_nouns[noun_2][noun_1] += count

    final_dict = defaultdict(lambda: defaultdict(int))
    for noun, noun_dic in noun_to_nouns.iteritems():
        for context_noun, count in noun_dic.iteritems():
            if count > MIN_COUNT:
                final_dict[noun][context_noun] = count
    return final_dict

def get_label_to_nouns():
    label_to_nouns = defaultdict(list)
    with open(args.adj_analysis_file_with_outliers) as f:
        for row in f:
            row_data = row.rstrip('\n').split('\t')
            if row_data[0] != 'label':
                label_to_nouns[int(row_data[0])].append(row_data[1])

    return label_to_nouns

def generate_avg_vec_per_cluster(label_to_nouns):
    label_to_avg_vec = {}
    for label,nouns in label_to_nouns.iteritems():
        if label == OUTLIER_LABEL:
            continue
        nouns_matrix = np.array([we_model.word_vec(noun) for noun in nouns]).squeeze()
        nouns_avg_vector = np.average(nouns_matrix, axis=0)
        label_to_avg_vec[label] = nouns_avg_vector

    sorted_by_label = sorted(label_to_avg_vec.items(), key=operator.itemgetter(0))
    sorted_by_label_matrix = np.array([item[1] for item in sorted_by_label])

    return sorted_by_label_matrix

def get_outliers_matrix(outliers_list):
    #return list of outlier word vectors

    outliers_matrix = np.array([we_model.word_vec(word) for word in outliers_list]).squeeze()
    return outliers_matrix

def calc_pattern_matrix(noun_to_nouns_list, outliers, outliers_matrix):

    outliers_vecs = []
    for idx,outlier in enumerate(outliers):
        if outlier == "restaurant":
            print "here"
        nouns_to_count = noun_to_nouns_list[outlier] #list of tuples [(noun_1,count),(noun_2,count)..]
        noun_count_list = [(noun,count) for noun,count in nouns_to_count.iteritems() if noun in we_model.vocab]
        outlier_vec = outliers_matrix[idx]#np.zeros(300)
        if len(noun_count_list) >= 1:
            nouns = [item[0] for item in noun_count_list]
            nouns_matrix = np.array([we_model.word_vec(noun) for noun in nouns ]).squeeze()
            nouns_weights = [item[1] for item in noun_count_list]#weights are the co-occurrence counts
            if len(noun_count_list) == 1:
                outlier_vec = nouns_matrix
            else:
                outlier_vec = np.average(nouns_matrix, axis=0,weights=nouns_weights)
        outliers_vecs.append(outlier_vec)

    outliers_matrix = np.array(outliers_vecs).squeeze()
    return outliers_matrix


def calc_sim(outliers_matrix, label_to_avg_matrix):
    #claculate similarity between every outlier vector to all the clusters
    sim_matrix = np.dot(outliers_matrix, label_to_avg_matrix.T)
    #results dimension: (num_of_labels*num_of_outliers)
    return sim_matrix

def cluster(sim_matrix, outliers):
    label_to_outliers = defaultdict(list)
    clustering = np.argmax(sim_matrix, axis=1)
    labels = clustering.tolist()
    for idx,label in enumerate(labels):
        clustered_outlier = outliers[idx]
        label_to_outliers[label].append(clustered_outlier)
    return label_to_outliers

def generate_output(label_to_nouns, label_to_outliers, out_file):
    file_path = os.path.join(args.output_folder, out_file)
    with open(file_path, 'a') as f:
        for label,nouns in label_to_nouns.iteritems():
            if label == OUTLIER_LABEL:
                continue
            f.write("label\t{}\n".format(label))
            f.write("\n".join([(str(label) + "\t" + word) for word in nouns]))
            f.write("\n")
            f.write("\n".join([(str(label) + "\t" + word) for word in label_to_outliers[label]]))
            f.write("\n")


def run():

    noun_to_nouns_list = get_matching_nouns_from_patterns()
    label_to_nouns = get_label_to_nouns()
    label_to_avg_vec_matrix = generate_avg_vec_per_cluster(label_to_nouns)
    outliers_list = label_to_nouns[OUTLIER_LABEL]
    outliers_matrix = get_outliers_matrix(outliers_list)
    outliers_pattern_matrix = calc_pattern_matrix(noun_to_nouns_list,outliers_list, outliers_matrix)

    similarity_matrix = calc_sim(outliers_matrix, label_to_avg_vec_matrix)
    patterns_similarity_matrix = calc_sim(outliers_pattern_matrix, label_to_avg_vec_matrix)

    #cluster based on similarity_matrix
    label_to_outliers = cluster(similarity_matrix, outliers_list)
    generate_output(label_to_nouns, label_to_outliers, "sim")
    #cluster based on  patterns_similarity_matrix
    label_to_outliers = cluster(patterns_similarity_matrix, outliers_list)
    generate_output(label_to_nouns, label_to_outliers, "patterns")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generate adjectives with multi senses list.')

    parser.add_argument('we_file',help='pre-trained word embedding')
    parser.add_argument('adj_analysis_file_with_outliers',help='adj analysis file (with the labels and outliers)')
    # parser.add_argument('adj_clusters_file_with_outliers',help='adjective clusters file')
    parser.add_argument('patterns_folder',help='folder containing similar nouns patterns')
    parser.add_argument('output_folder', help ='output file for clustered nouns (including outliers')



    args = parser.parse_args()
    we_model = gensim.models.KeyedVectors.load(args.we_file, mmap='r') .wv # mmap the large matrix as read-only
    we_model.syn0norm = we_model.syn0

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    run()

    print "DONE"
