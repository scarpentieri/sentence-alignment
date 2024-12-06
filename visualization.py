#!/usr/bin/env python3
# visualization.py

# Author: Sofia Carpentieri

# Department of Computational Linguistics
# University of Zurich



import os
import html
from tqdm import tqdm
import csv
import pandas as pd
import numpy as np
import re
import numpy as numpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import spacy
nlp_de = spacy.load("de_core_news_sm")
nlp_it = spacy.load("it_core_news_sm")
nlp_fr = spacy.load("fr_core_news_sm")


path_deliv = '/net/cephfs/shares/iict-sp2.ebling.cl.uzh/Deliverable'
paths = {'mitenand': 
			{'de': f'{path_deliv}/DSGS/mitenand/subtitles_segmented/', 
			'it': f'{path_deliv}/LIS-CH/insieme/subtitles_segmented/', 
			'fr': f'{path_deliv}/LSF-CH/ensemble/subtitles_segmented/'}, 
		'helveticus': 
			{'de': f'{path_deliv}/DSGS/helveticus/subtitles_segmented/', 
			'it': f'{path_deliv}/LIS-CH/helveticus/subtitles_segmented/',
			'fr': f'{path_deliv}/LSF-CH/helveticus/subtitles_segmented/'}		
		}




def statistics_sentences(alignment_name:str, server_shortname:str):

	all_sentences = list()
	all_sentences_tokens = list()

	with open(f'{server_shortname}/aligned/meta_{alignment_name}.tsv', 'r', encoding='utf-8') as alignment:
		tsv = pd.read_csv(alignment, sep='\t', header=0)

		filecounter = sum(tsv['aligned_languages'])

		for row in tqdm(tsv.iterrows(), desc='Tokenizing using spaCy'):

			try:
				de_sentences = open(paths[alignment_name]['de']+row[1][4]).readlines()
				it_sentences = open(paths[alignment_name]['it']+row[1][5]).readlines()
				fr_sentences = open(paths[alignment_name]['fr']+row[1][6]).readlines()

				# &#x27;		apostrophe
				all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in de_sentences[2::4]])
				all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in it_sentences[2::4]])
				all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in fr_sentences[2::4]])

				all_sentences_tokens.append([[token.text for token in nlp_de(html.escape(i.strip()))] for i in de_sentences[2::4]])
				all_sentences_tokens.append([[token.text for token in nlp_it(html.escape(i.strip()))] for i in it_sentences[2::4]])
				all_sentences_tokens.append([[token.text for token in nlp_fr(html.escape(i.strip()))] for i in fr_sentences[2::4]])

			except TypeError:
				if row[1][4] == '' or pd.isna(row[1][4]):
					it_sentences = open(paths[alignment_name]['it']+row[1][5]).readlines()
					fr_sentences = open(paths[alignment_name]['fr']+row[1][6]).readlines()

					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in it_sentences[2::4]])
					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in fr_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_de(html.escape(i.strip()))] for i in it_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_it(html.escape(i.strip()))] for i in fr_sentences[2::4]])

				elif row[1][5] == '' or pd.isna(row[1][5]):
					de_sentences = open(paths[alignment_name]['de']+row[1][4]).readlines()
					fr_sentences = open(paths[alignment_name]['fr']+row[1][6]).readlines()

					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in de_sentences[2::4]])
					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in fr_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_de(html.escape(i.strip()))] for i in de_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_it(html.escape(i.strip()))] for i in fr_sentences[2::4]])

				elif row[1][6] == '' or pd.isna(row[1][6]):
					de_sentences = open(paths[alignment_name]['de']+row[1][4]).readlines()
					it_sentences = open(paths[alignment_name]['it']+row[1][5]).readlines()

					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in de_sentences[2::4]])
					all_sentences.append([html.escape(i.strip()).replace('&#x27;', "'") for i in it_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_de(html.escape(i.strip()))] for i in de_sentences[2::4]])
					all_sentences_tokens.append([[token.text for token in nlp_it(html.escape(i.strip()))] for i in it_sentences[2::4]])
					


	all_sentences_flattened = [i for file in all_sentences for i in file]
	all_sentences_tokens_flattened = [t for file in all_sentences_tokens for i in file for t in i]

	out = open(f'{server_shortname}/statistics_sentences_{alignment_name}.txt', 'w+', encoding='utf-8')

	out.write('----------\n')
	out.write(f'Statistics for {alignment_name}:\n\n')
	if alignment_name == 'mitenand':
		out.write(f'Average sentences per file: {round(len(all_sentences_flattened)/filecounter)} sentences\n')
	elif alignment_name == 'helveticus':
		out.write(f'Average sentences per file: {round(len(all_sentences_flattened)/filecounter)} sentences\n')
	out.write(f'Least sentences in a file: {len(min(all_sentences, key=len))} sentences\n')
	out.write(f'Most sentences in a file: {len(max(all_sentences, key=len))} sentences\n')
	out.write(f'Average tokens per sentence: {round(len(all_sentences_tokens_flattened)/len(all_sentences_flattened))} tokens\n')
	out.write(f'Shortest sentence overall: {min(all_sentences_flattened, key=len)}\n')
	out.write(f'Longest sentence overall: {max(all_sentences_flattened, key=len)}\n')
	out.write('----------')
	out.close()


	

def statistics_alignments(alignment_name:str, server_shortname:str):

	out = open(f'{server_shortname}/statistics_alignments_{alignment_name}.txt', 'w+', encoding='utf-8')

	with open(f'{server_shortname}/aligned/meta_{alignment_name}.tsv', 'r', encoding='utf-8') as alignment:

		sentences = 0
		total_aligned = 0
		triple_aligned = 0

		tsv = pd.read_csv(alignment, sep='\t', header=0)
		for row in tsv.iterrows():
			aligned_sentences = row[1][3]
			total_aligned += aligned_sentences
			if row[1][1] == 3:
				de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
				it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
				fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4
				triple = [line[1][1] for line in pd.read_csv(f'{server_shortname}/aligned/{row[1][0]}', sep='\t').iterrows() if line [1][1] == 3]
				sentences += de_sentences + it_sentences + fr_sentences
				triple_aligned += len(triple)

				out.write(f'{row[1][0]}\t{aligned_sentences}\t{de_sentences}\t{it_sentences}\t{fr_sentences}\ttriple_aligned:{len(triple)}\tpercentage_all_triple:{round(100 * float(len(triple))/float(aligned_sentences), 2)}%\n')
			elif row[1][1] == 2:
				if pd.isna(row[1][4]):
					it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
					fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4
					sentences += it_sentences + fr_sentences
					out.write(f'{row[1][0]}\t{aligned_sentences}\t--\t{it_sentences}\t{fr_sentences}\n')

				elif pd.isna(row[1][5]):
					de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
					fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4
					sentences += de_sentences + fr_sentences
					out.write(f'{row[1][0]}\t{aligned_sentences}\t{de_sentences}\t--\t{fr_sentences}\n')

				elif pd.isna(row[1][6]):
					de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
					it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
					sentences += de_sentences + it_sentences
					out.write(f'{row[1][0]}\t{aligned_sentences}\t{de_sentences}\t{it_sentences}\t--\n')


	out.write(f'Overall sentences: {sentences} // Total aligned: {total_aligned} // Triples aligned: {triple_aligned} ({round(100*float(triple_aligned)/float(sentences), 4)}% of all sentences)')

	out.close()


def plotting_alignments(alignment_name:str, server_shortname:str):


	with open(f'{server_shortname}/aligned/meta_{alignment_name}.tsv', 'r', encoding='utf-8') as alignment:

		tsv = pd.read_csv(alignment, sep='\t', header=0)

		for row in tqdm(tsv.iterrows(), desc=f'Creating plots for {alignment_name}'):

			if row[1][1] == 3:
				# three languages matrices
				de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
				it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
				fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4

				maximum = max(de_sentences, it_sentences, fr_sentences)

				matrix_de_it = np.zeros([maximum, maximum], dtype=int)
				matrix_de_fr = np.zeros([maximum, maximum], dtype=int)
				matrix_it_fr = np.zeros([maximum, maximum], dtype=int)

				for line in pd.read_csv(f'{server_shortname}/aligned/{row[1][0]}', sep='\t').iterrows():
					if line[1][1] == 3:
						de = int(line[1][2].split('\'')[1])
						it = int(line[1][3].split('\'')[1])
						fr = int(line[1][4].split('\'')[1])
						matrix_de_it[de-1][it-1] = 2
						matrix_de_fr[de-1][fr-1] = 2

					elif line[1][1] == 2:
						if pd.isna(line[1][2]):
							it = int(line[1][3].split('\'')[1])
							fr = int(line[1][4].split('\'')[1])
							matrix_it_fr[it-1][fr-1] = 1
						elif pd.isna(line[1][3]):
							de = int(line[1][2].split('\'')[1])
							fr = int(line[1][4].split('\'')[1])
							matrix_de_fr[de-1][fr-1] = 1
						elif pd.isna(line[1][4]):
							de = int(line[1][2].split('\'')[1])
							it = int(line[1][3].split('\'')[1])
							matrix_de_it[de-1][it-1] = 1

					fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
					fig.set_figwidth(15)
					fig.suptitle(f'{row[1][0]}')

					ax1.imshow(matrix_de_it, cmap='hot_r')
					ax1.set(xlabel='Sentence IDs (Italian)', ylabel='Sentence IDs (German)', title='Matrix DE-IT')

					ax2.imshow(matrix_de_fr, cmap='hot_r')
					ax2.set(xlabel='Sentence IDs (French)', ylabel='Sentence IDs (German)', title='Matrix DE-FR')

					ax3.imshow(matrix_it_fr, cmap='hot_r')
					ax3.set(xlabel='Sentence IDs (French)', ylabel='Sentence IDs (Italian)', title='Matrix IT-FR')

					plt.savefig(f'{server_shortname}/plots/{row[1][0][:-4]}_plot.png')
					# plt.clf()
					ax1.clear()
					ax2.clear()
					ax3.clear()
					plt.close(fig)


			elif row[1][1] == 2:
				# two languages matrix
				if pd.isna(row[1][4]):	# no german
					it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
					fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4

					maximum = max(it_sentences, fr_sentences)

					matrix = np.zeros([maximum, maximum], dtype=int)

					for line in pd.read_csv(f'{server_shortname}/aligned/{row[1][0]}', sep='\t').iterrows():
						it = int(line[1][3].split('\'')[1])
						fr = int(line[1][4].split('\'')[1])
						matrix[it-1][fr-1] = 1

					plt.imshow(matrix, cmap='Purples')
					plt.title(f'{row[1][0]}')
					plt.xlabel('Sentence IDs (French)')
					plt.ylabel('Sentence IDs (Italian)')
					plt.savefig(f'{server_shortname}/plots/{row[1][0][:-4]}_plot.png')
					plt.cla()

				elif pd.isna(row[1][5]):	# no italian
					de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
					fr_sentences = len(open(paths[alignment_name]['fr']+row[1][6]).readlines())//4
					
					maximum = max(de_sentences, fr_sentences)

					matrix = np.zeros([maximum, maximum], dtype=int)

					for line in pd.read_csv(f'{server_shortname}/aligned/{row[1][0]}', sep='\t').iterrows():
						de = int(line[1][2].split('\'')[1])
						fr = int(line[1][4].split('\'')[1])
						matrix[de-1][fr-1] = 1

					plt.imshow(matrix, cmap='Purples')
					plt.title(f'{row[1][0]}')
					plt.xlabel('Sentence IDs (French)')
					plt.ylabel('Sentence IDs (German)')
					plt.savefig(f'{server_shortname}/plots/{row[1][0][:-4]}_plot.png')
					plt.cla()

				elif pd.isna(row[1][6]):	# no french
					de_sentences = len(open(paths[alignment_name]['de']+row[1][4]).readlines())//4
					it_sentences = len(open(paths[alignment_name]['it']+row[1][5]).readlines())//4
					
					maximum = max(de_sentences, it_sentences)

					matrix = np.zeros([maximum, maximum], dtype=int)

					for line in pd.read_csv(f'{server_shortname}/aligned/{row[1][0]}', sep='\t').iterrows():
						de = int(line[1][2].split('\'')[1])
						it = int(line[1][3].split('\'')[1])
						matrix[de-1][it-1] = 1

					plt.imshow(matrix, cmap='Purples')
					plt.title(f'{row[1][0]}')
					plt.xlabel('Sentence IDs (Italian)')
					plt.ylabel('Sentence IDs (German)')
					plt.savefig(f'{server_shortname}/plots/{row[1][0][:-4]}_plot.png')
					plt.cla()


	






