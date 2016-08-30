# -*- coding: utf-8 -*-
# Project : DeepBot
# Created by igor on 16/8/30

'''
Cornell Movie--Dialogs Corpus
'''

import os
import re

import pandas as pd

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

    ID2movie = dict(zip(titles_df["movieID"], titles_df["movieTitle"]))
    for utteranceIDs in conversation_df['utterancesIDs']:
        lineIDs = re.findall("'(.*?)'", utteranceIDs)
        _, _, mID, character1, context = lines_df[lines_df['lineID'] == lineIDs[0]].values[0]
        for lineID in lineIDs[1:]:
            _, _, _, character2, utterance = lines_df[lines_df['lineID'] == lineID].values[0]
            yield character1, character2, ID2movie[mID], context, utterance
            character1 = character2
            context = utterance


if __name__ == '__main__':
    file_dir = "/data/cmd_corpus"
    # for fn in FILES_NAME:
    #     _rawToCsv(file_dir, fn, True)
    # for i in utterance_generator(file_dir):
    #     print(i)
