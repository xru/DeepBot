# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/8/30

'''
Cornell Movie--Dialogs Corpus
'''

import os
import re

import nltk
import pandas as pd
import numpy as np

FILES_NAME = ["movie_characters_metadata.txt",
              "movie_conversations.txt",
              "movie_lines.txt",
              "movie_titles_metadata.txt", ]

FILE_CODING = ["iso-8859-1",
               "us-ascii",
               "iso-8859-1",
               "iso-8859-1", ]

HEADERS = [["characterID", "characterName", "movieID", "movieTitle", "gender", "position"],
           ["Character1ID", "Character2ID", "movieID", "utterancesIDs"],
           ["lineID", "characterID", "movieID", "CharacterName", "utteranceText"],
           ["movieID", "movieTitle", "movieYear", "imbdRating", "imbdVotes", "genres"], ]

CODING_MAPS = dict(zip(FILES_NAME, FILE_CODING))
HEADERS_MAPS = dict(zip(FILES_NAME, HEADERS))


def _rawToCsv(file_dir, file_name, save=False):
    path = os.path.join(file_dir, file_name)
    assert os.path.exists(path) and os.path.isfile(path)
    data = []
    with open(os.path.join(file_dir, file_name), 'r', encoding=CODING_MAPS[file_name]) as f:
        for line in f:
            data.append([field.strip() for field in line.strip().split("+++$+++")])
    df = pd.DataFrame(data, columns=HEADERS_MAPS[file_name])
    if save:
        file_name = os.path.splitext(file_name)[0]
        csv_path = os.path.join(file_dir, file_name + ".csv")
        df.to_csv(csv_path, index=False, encoding="utf8")
        print("Save %s to %s" % (file_name, csv_path))
    return df


def _text_to_csv(fname):
    return os.path.splitext(fname)[0] + '.csv'


def utterance_generator(file_dir):
    csv_files = map(_text_to_csv, FILES_NAME)
    characters_df, conversation_df, lines_df, titles_df = \
        (pd.read_csv(os.path.join(file_dir, fn)) for fn in csv_files)

    ID2utterance = dict(zip(lines_df["lineID"], lines_df["utteranceText"]))
    #  直接用字典比在dataframe中查询快速
    pattern = re.compile(r"[^a-zA-Z1-9\.!?,' ]")

    for utteranceIDs in conversation_df['utterancesIDs']:
        lineIDs = re.findall("'(.*?)'", utteranceIDs)
        context = ID2utterance[lineIDs[0]]
        for lineID in lineIDs[1:]:
            utterance = ID2utterance[lineID]

            try:
                context = pattern.sub("", context)
                utterance = pattern.sub("", utterance)
            except Exception as e:
                # print("Wrong preprocess...%s" % e)
                continue

            yield context, utterance
            # print(len(context) - len(utterance))
            context = utterance


def take_some_analysis(file_dir):
    context_length = []
    utterance_length = []

    dist = nltk.FreqDist()

    for c, u in utterance_generator(file_dir):
        c_tokens = nltk.word_tokenize(c)
        u_tokens = nltk.word_tokenize(u)
        #  记录长度
        context_length.append(len(c_tokens))
        utterance_length.append(len(u_tokens))

        dist.update(c_tokens + u_tokens)

    cl_array = np.array(context_length)
    ul_array = np.array(utterance_length)

    print("most length of context is %d" % cl_array.max())
    print("most length of utterance is %d" % ul_array.max())
    print("mean length of context is %f" % cl_array.mean())
    print("mean length of utterance is %f" % ul_array.mean())

    sub_abs = np.abs(cl_array - ul_array)
    print("max,min,mean of abs(context_length -utterance_length) is %f,%f,%f" % (
        np.max(sub_abs), np.min(sub_abs), np.mean(sub_abs)))

    print("most common words :")
    print(dist.most_common(10))


if __name__ == '__main__':
    file_dir = "/data/cmd_corpus"
    # for fn in FILES_NAME:
    #     _rawToCsv(file_dir, fn, True)
    # take_some_analysis(file_dir)
    for i,j in utterance_generator(file_dir):
        print(i)
        print(j)
        print('\n')