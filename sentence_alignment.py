#!/usr/bin/env python3
# sentence_alignment.py

# Author: Sofia Carpentieri

# Department of Computational Linguistics
# University of Zurich




import os
from tqdm import tqdm
import csv
import pandas as pd
import re

from sentence_transformers import SentenceTransformer

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = SentenceTransformer('sentence-transformers/stsb-xlm-r-multilingual')



def adjust_sentence_segments(filename:str):

    corrected_sentences = list()

    with open(filename, 'r', encoding='utf-8') as infile:

        text = infile.read()
        subtitle_items = text.split('\n\n')
        subtitles = [i.split('\n') for i in subtitle_items if len(i) > 1]

        segmented = list()
        segmented_time = list()

        current = list()
        current_time = list()

        for i, sub in enumerate(subtitles):

            current_time.append(sub[1].split(" --> ")[0])

            if len(sub[2:]) > 1:
                text = (" ".join(sub[2:]))

            elif len(sub[2:]) == 1:
                text = sub[2]

            if len(text) > 1:
                if text.strip().endswith(('.', '!', '?', '*')):
                    current.append(text)
                    segmented.append(current)

                    segmented_time.append([current_time, sub[1].split(" --> ")[1]])

                    current = list()
                    current_time = list()

                elif not text.strip().endswith(('.', '!', '?', '*')):
                    current.append(text)

        timestamps = list()

        for stamps in segmented_time:
            start = stamps[0][0]
            end = stamps[1]
            timestamps.append([start, end])


        for i, e in enumerate(zip(timestamps, segmented)):

            formatted_for_pipeline = (f'{i}', f'{e[0][0]} --> {e[0][1]}', f"{' '.join(e[1][0:])}")
            corrected_sentences.append(formatted_for_pipeline)

    return corrected_sentences



def get_sentences(filename:str):

    with open(filename, 'r', encoding='utf-8') as file:
        lines = list()
        for line in file:
            if not line == '\n':
                lines.append(line.strip())
        lines = [(lines[i], lines[i+1], lines[i+2]) for i, line in enumerate(lines) if i%3==0]
    
    return lines



def similarities(a:list, b:list, threshold=0.6):
    
    similarities = list()

    for embedding in a:
        simil = list()
        for e in b:
            sim = cosine_similarity(embedding, e)[0][0]
            if sim >= threshold:     # old threshold: >=0.85
                simil.append(sim)

            else:
                simil.append(0)
        similarities.append(simil)

    return similarities



def find_alignments(similarities:list, sentences1:list, sentences2:list, lang1:str, lang2:str):

    aligned = list()
    aligned_count = 0

    for i, s in enumerate(similarities):
        if sum(s) == 0:
            continue
        elif sum(s) > 0:
            highest = sorted([(a, b) for a, b in enumerate(s)], key=lambda x: x[1], reverse=True)[0]
            aligned.append({lang1: sentences1[i], lang2: sentences2[highest[0]]})
            aligned_count += 1

    return aligned



def clean_filenames(alignment:str):

    mitenand = ['DSGS mitenand manual', 'LIS insieme manual', 'LSF ensemble manual']
    helveticus = ['DSGS helveticus', 'LIS helveticus', 'LSF helveticus']

    if alignment == 'mitenand':

        count_all = 0

        df = pd.read_csv('alignment_mitenand_ensemble_insieme_manual.csv')
        df.fillna('', inplace=True)
        trios_df = df[mitenand]

        cleaned_filenames = list()

        for j, row in enumerate(trios_df.iterrows()):

            count_all += 1

            filenames = [row[1]['DSGS mitenand manual'], row[1]['LIS insieme manual'], row[1]['LSF ensemble manual']]

            if filenames.count('') < 2:
                cleaned = list()
                for i, name in enumerate(filenames):
                    if name == '':
                        cleaned.append(name)
                    elif not name.endswith('.srt'):
                        short = name.split('.')
                        if short[0].endswith("'"):
                            new = short[0][1:-1] + ".srt"
                            cleaned.append(new.strip())
                        elif short[0].endswith('"'):
                            new = short[0][1:-1] + '.srt'
                            cleaned.append(new.strip())
                        else:
                            new = short[0] + '.srt'
                            cleaned.append(new.strip())

                cleaned_filenames.append(cleaned)

    elif alignment == 'helveticus':

        count_all = 0

        df = pd.read_csv('alignment_helveticus.csv')
        df.fillna('', inplace=True)
        trios_df = df[helveticus]

        cleaned_filenames = list()

        for j, row in enumerate(trios_df.iterrows()):

            count_all += 1

            filenames = [row[1]['DSGS helveticus'], row[1]['LIS helveticus'], row[1]['LSF helveticus']]

            if filenames.count('') < 2:
                cleaned = list()
                for i, name in enumerate(filenames):
                    if name == '':
                        cleaned.append(name)
                    elif not name.endswith('.srt'):
                        short = name.split('.')
                        if short[0].endswith("'"):
                            new = short[0][:-1] + ".srt'"
                            cleaned.append(new.strip())
                        elif short[0].endswith('"'):
                            new = short[0][:-1] + '.srt"'
                            cleaned.append(new.strip())
                        else:
                            new = short[0] + '.srt'
                            cleaned.append(new.strip())

                cleaned_filenames.append(cleaned)



    return cleaned_filenames



def align_sentences_three(sentences: dict):

    de = [model.encode([s]) for i, t, s in sentences['de']]
    it = [model.encode([s]) for i, t, s in sentences['it']]
    fr = [model.encode([s]) for i, t, s in sentences['fr']]

    
    sims_de_it = similarities(de, it)
    de_it = find_alignments(sims_de_it, sentences['de'], sentences['it'], 'de', 'it')

    sims_de_fr = similarities(de, fr)
    de_fr = find_alignments(sims_de_fr, sentences['de'], sentences['fr'], 'de', 'fr')

    sims_it_fr = similarities(it, fr)
    it_fr = find_alignments(sims_it_fr, sentences['it'], sentences['fr'], 'it', 'fr')

    aligned = de_it

    # trios aligned (de-it-fr)
    for item in aligned:
        for dictionary in de_fr:
            try:
                if item['de'][0] == dictionary['de'][0]:
                    item.update({'fr': dictionary['fr']})
            except: continue

    de_ids = [item['de'][0] for item in aligned]

    # add de-fr
    for dictionary in de_fr:
        if dictionary['de'][0] not in de_ids:
            aligned.append(dictionary)

    it_ids = [item['it'][0] for item in aligned if 'it' in item]
    fr_ids = [item['fr'][0] for item in aligned if 'fr' in item]


    # add it-fr
    for dictionary in it_fr:
        if dictionary['it'][0] not in it_ids and dictionary['fr'][0] not in fr_ids:
            aligned.append(dictionary)


    return {i: item for i, item in enumerate(aligned)}



def align_sentences_two(sentences: dict):

    if sentences['fr'] == '':
        de = [model.encode([s]) for i, t, s in sentences['de']]
        it = [model.encode([s]) for i, t, s in sentences['it']]
        sims_de_it = similarities(de, it)
        de_it = find_alignments(sims_de_it, sentences['de'], sentences['it'], 'de', 'it')

        d =  {i: item for i, item in enumerate(de_it)}

    elif sentences['de'] == '':
        it = [model.encode([s]) for i, t, s in sentences['it']]
        fr = [model.encode([s]) for i, t, s in sentences['fr']]
        sims_it_fr = similarities(it, fr)
        it_fr = find_alignments(sims_it_fr, sentences['it'], sentences['fr'], 'it', 'fr')

        d = {i: item for i, item in enumerate(it_fr)}

    elif sentences['it'] == '':
        de = [model.encode([s]) for i, t, s in sentences['de']]
        fr = [model.encode([s]) for i, t, s in sentences['fr']]
        sims_de_fr = similarities(de, fr)
        de_fr = find_alignments(sims_de_fr, sentences['de'], sentences['fr'], 'de', 'fr')

        d =  {i: item for i, item in enumerate(de_fr)}

    return d



def format_alignment(aligned:dict, outfile:str):

    if not outfile.endswith('.tsv'):
            name = outfile.split('.')
            outfile = name[0] + '.tsv'

    with open(outfile, 'w', encoding='utf-8') as out:
        out.write(f'id\taligned\tde\tit\tfr\n')

        for i, dictionary in aligned.items():
            if len(dictionary) == 3:
                line = f'{i}\t3\t{dictionary["de"]}\t{dictionary["it"]}\t{dictionary["fr"]}\n'
                out.write(line)
            elif len(dictionary) == 2:
                if 'de' in dictionary and 'it' in dictionary:
                    line = f'{i}\t2\t{dictionary["de"]}\t{dictionary["it"]}\t\n'
                elif 'de' in dictionary and 'fr' in dictionary:
                    line = f'{i}\t2\t{dictionary["de"]}\t\t{dictionary["fr"]}\n'
                elif 'it' in dictionary and 'fr' in dictionary:
                    line = f'{i}\t2\t\t{dictionary["it"]}\t{dictionary["fr"]}\n'
                out.write(line)



def align_files_mitenand(file_de:str, file_it:str, file_fr:str, outpath:str):

    names = [file_de, file_it, file_fr]

    if '' in names:
        if names[0] == '':
            it = get_sentences(file_it)
            fr = get_sentences(file_fr)
            trio = {'de': '', 'it': it, 'fr': fr}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

        elif names[1] == '': 
            de = get_sentences(file_de)
            fr = get_sentences(file_fr)
            trio = {'de': de, 'it': '', 'fr': fr}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

        elif names[2] == '':
            de = get_sentences(file_de)
            it = get_sentences(file_it)
            trio = {'de': de, 'it': it, 'fr': ''}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

    else:

        de = get_sentences(file_de)
        it = get_sentences(file_it)
        fr = get_sentences(file_fr)
      
        trio = {'de': de, 'it': it, 'fr': fr}

        aligned = align_sentences_three(trio)
        format_alignment(aligned, outpath)



def align_files_helveticus(file_de:str, file_it:str, file_fr:str, outpath:str):

    names = [file_de, file_it, file_fr]

    if '' in names:
        if names[0] == '':
            it = adjust_sentence_segments(file_it)
            fr = get_sentences(file_fr)
            trio = {'de': '', 'it': it, 'fr': fr}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

        elif names[1] == '': 
            de = get_sentences(file_de)
            fr = get_sentences(file_fr)
            trio = {'de': de, 'it': '', 'fr': fr}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

        elif names[2] == '':
            de = get_sentences(file_de)
            it = adjust_sentence_segments(file_it)
            trio = {'de': de, 'it': it, 'fr': ''}
            aligned = align_sentences_two(trio)
            format_alignment(aligned, outpath)

    else:

        de = get_sentences(file_de)
        it = adjust_sentence_segments(file_it)
        fr = get_sentences(file_fr)
      
        trio = {'de': de, 'it': it, 'fr': fr}

        aligned = align_sentences_three(trio)
        format_alignment(aligned, outpath)



def run(server_shortname:str):

    outdir = 'aligned'

    path = '/net/cephfs/shares/iict-sp2.ebling.cl.uzh/Deliverable' # '{langugage}/{program}' + '/subtitles_segmented/' + '{filename}.srt'

    cleaned_mitenand = clean_filenames('mitenand')

    metafile_mitenand = open(f'{server_shortname}/{outdir}/meta_mitenand.tsv', 'w+', encoding='utf-8')
    metafile_mitenand.write(f'filename_target\taligned_languages\taligned_languages_names\tfile_lines\tde_source\tit_source\tfr_source\n')

    for j, trio in tqdm(enumerate(cleaned_mitenand), desc='Aligning files for mitenand'):

        if '' not in trio:

            try: 
                de = f'{path}/DSGS/mitenand/subtitles_segmented/{trio[0]}'
                it = f'{path}/LIS-CH/insieme/subtitles_segmented/{trio[1]}'
                fr = f'{path}/LSF-CH/ensemble/subtitles_segmented/{trio[2]}'

                f = f'{server_shortname}/{outdir}/mitenand_{j}.tsv'

                align_files_mitenand(de, it, fr, f) 

                metafile_mitenand.write(f'mitenand_{j}.tsv\t3\tde-it-fr\t{len(open(f).readlines())-1}\t{trio[0]}\t{trio[1]}\t{trio[2]}\n')

            except FileNotFoundError:

                print('\n' + '*'*10)
                print(f'FileNotFoundError: Please check the filenames for the following trio:')
                print(f'DE: {trio[0]}')
                print(f'IT: {trio[1]}')
                print(f'FR: {trio[2]}')
                print('*'*10)

        elif '' in trio:

            if trio[0] == '':
                try: 
                    it = f'{path}/LIS-CH/insieme/subtitles_segmented/{trio[1]}'
                    fr = f'{path}/LSF-CH/ensemble/subtitles_segmented/{trio[2]}'

                    f = f'{server_shortname}/{outdir}/mitenand_{j}.tsv'

                    align_files_mitenand('', it, fr, f)

                    metafile_mitenand.write(f'mitenand_{j}.tsv\t2\tit-fr\t{len(open(f).readlines())-1}\t\t{trio[1]}\t{trio[2]}\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()


            elif trio[1] == '':
                try: 
                    de = f'{path}/DSGS/mitenand/subtitles_segmented/{trio[0]}'
                    fr = f'{path}/LSF-CH/ensemble/subtitles_segmented/{trio[2]}'

                    f = f'{server_shortname}/{outdir}/mitenand_{j}.tsv'

                    align_files_mitenand(de, '', fr, f)

                    metafile_mitenand.write(f'mitenand_{j}.tsv\t2\tde-fr\t{len(open(f).readlines())-1}\t{trio[0]}\t\t{trio[2]}\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()

            elif trio[2] == '':
                try: 
                    de = f'{path}/DSGS/mitenand/subtitles_segmented/{trio[0]}'
                    it = f'{path}/LIS-CH/insieme/subtitles_segmented/{trio[1]}'

                    f = f'{server_shortname}/{outdir}/mitenand_{j}.tsv'

                    align_files_mitenand(de, it, '', f)

                    metafile_mitenand.write(f'mitenand_{j}.tsv\t2\tde-it\t{len(open(f).readlines())-1}\t{trio[0]}\t{trio[1]}\t\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()

    metafile_mitenand.close()


    cleaned_helveticus =  clean_filenames('helveticus')

    metafile_helveticus = open(f'{server_shortname}/{outdir}/meta_helveticus.tsv', 'w+', encoding='utf-8')
    metafile_helveticus.write(f'filename_target\taligned_languages\taligned_languages_names\tfile_lines\tde_source\tit_source\tfr_source\n')

    for j, trio in tqdm(enumerate(cleaned_helveticus), desc='Aligning files for helveticus'):

        if '' not in trio:

            try: 
                de = f'{path}/DSGS/helveticus/subtitles_segmented/{trio[0]}'
                it = f'{path}/LIS-CH/helveticus/subtitles_segmented/{trio[1]}'
                fr = f'{path}/LSF-CH/helveticus/subtitles_segmented/{trio[2]}'

                f = f'{server_shortname}/{outdir}/helveticus_{j}.tsv'

                align_files_helveticus(de, it, fr, f) 

                metafile_helveticus.write(f'helveticus_{j}.tsv\t3\tde-it-fr\t{len(open(f).readlines())-1}\t{trio[0]}\t{trio[1]}\t{trio[2]}\n')

            except FileNotFoundError:

                print('\n' + '*'*10)
                print(f'FileNotFoundError: Please check the filenames for the following trio:')
                print(f'DE: {trio[0]}')
                print(f'IT: {trio[1]}')
                print(f'FR: {trio[2]}')
                print('*'*10)

        elif '' in trio:

            if trio[0] == '':
                try: 
                    it = f'{path}/LIS-CH/helveticus/subtitles_segmented/{trio[1]}'
                    fr = f'{path}/LSF-CH/helveticus/subtitles_segmented/{trio[2]}'

                    f = f'{server_shortname}/{outdir}/helveticus_{j}.tsv'

                    align_files_helveticus('', it, fr, f)

                    metafile_helveticus.write(f'helveticus_{j}.tsv\t2\tit-fr\t{len(open(f).readlines())-1}\t\t{trio[1]}\t{trio[2]}\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()


            elif trio[1] == '':
                try: 
                    de = f'{path}/DSGS/helveticus/subtitles_segmented/{trio[0]}'
                    fr = f'{path}/LSF-CH/helveticus/subtitles_segmented/{trio[2]}'

                    f = f'{server_shortname}/{outdir}/helveticus_{j}.tsv'

                    align_files_helveticus(de, '', fr, f)

                    metafile_helveticus.write(f'helveticus_{j}.tsv\t2\tde-fr\t{len(open(f).readlines())-1}\t{trio[0]}\t\t{trio[2]}\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()

            elif trio[2] == '':
                try: 
                    de = f'{path}/DSGS/helveticus/subtitles_segmented/{trio[0]}'
                    it = f'{path}/LIS-CH/helveticus/subtitles_segmented/{trio[1]}'

                    f = f'{server_shortname}/{outdir}/helveticus_{j}.tsv'

                    align_files_helveticus(de, it, '', f)

                    metafile_helveticus.write(f'helveticus_{j}.tsv\t2\tde-it\t{len(open(f).readlines())-1}\t{trio[0]}\t{trio[1]}\t\n')

                except FileNotFoundError:
                    print('whoops no file found')
                    print(trio)
                    print()

    metafile_helveticus.close()






