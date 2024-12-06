

# Sentence alignment and visualization of subtitle files

Sofia Carpentieri\
Departement of Computational Linguistics\
University of Zurich

---

**!! NOTE**  Please note that the code only works, when it is run on the s3it Science Cluster server, because it accesses the subtitle files directly from the deliverable under ```/net/cephfs/shares/iict-sp2.ebling.cl.uzh/Deliverable```.

## Set-Up

Please run ```$ pip3 install -r requirements.txt``` first, to create the apt environment.

## ```run_analysis.py```

This script combines all code needed for the sentence alignment and visualization. To run the code, you can use the shell command:

```$ python3 run_analysis.py```

The program will ask you to enter the path to the directory where all data should be stored (e.g., ```/home/uzh-shortname/analysis```). Then, the analysis will be run. It contains two main modules, ```sentence_alignment.py``` and ```visualization.py```, which are explained in the following.

## ```sentence_alignment.py```

This script contains all code necessary for running the pipeline to align the sentences of both corpora, mitenand/ensemble/insieme and helveticus. 

The steps of the pipeline are the following:

1. Clean the filenames to avoid files not being found on the server.
2. Iterate over all aligned episodes bit by bit. Either the episodes are in a trio, if the episode is translated into all languages, or in a duo, if the episode is only available in two languages.
3. For each duo/trio of episodes, firstly, the sentences, sentence ids and timestamps are extracted from the subtitle files.
4. To find the alignments, all sentences in all episodes from the duo/trio are embedded using a ```SentenceTransformer``` model.
5. Using the embeddings, cosine similarities are cross-calculated between all sentences. The threshold set to 0.6 indicates that any sentence pair with a similarity lower than 0.6 is disregarded for the alignment.
6. Then, the alignments are put together between the languages of the episodes.
7. Lastly, after the result have been compiled in separate files, the overall meta-file is written containing information about the alignments.


## ```visualization.py```

The visualization of the data is done in three ways: gathering some statistics about the sentences in the subtitle files, gathering statistics about the alignments and plotting the alignments for a better visual understanding. The visualization is done separately for each corpus, as has to be specified when calling the functions.

#### statistics_sentences()

This function returns the average number of sentences per file, the least and most sentences in a file, the average number of tokens per file and the shortest and longest sentence in all files.

#### statistics_alignments()

This function returns the overall number of sentences for each alignment file, the total number of sentences aligned, the number of triples aligned (for episodes available in all three languages) and the calculated percentage of triples among all sentences.

#### plotting_alignments()

This function creates plots to visualize the sentence alignments. In each plot, all sentence ids are represented. If two sentences are aligned, they are highlighted in the matrix. A perfectly aligned file can be recognized by a perfectly straight line from the top left to the bottom right. 



## Final directory structure


After running ```run_analysis.py```, the output files are stored in the directory specified by the user. Additionally, new directories are created. The final directory structure should look like the following:

```
directory/
│   ...
│
└─── aligned/
│   │   helveticus_0.tsv
│   │   helveticus_1.tsv
│   │   ...
│   │   meta_helveticus.tsv
│   │   mitenand_0.tsv
│   │   mitenand_1.tsv
│   │   ...
│   │   meta_mitenand.tsv
│   
└─── plots/
    │   helveticus_0_plot.png
    │   helveticus_1_plot.png
    │   ...
    │   mitenand_0_plot.png
    │   mitenand_1_plot.png
    │   ...

```





