#!/usr/bin/env python3
# run_analysis.py

# Author: Sofia Carpentieri

# Department of Computational Linguistics
# University of Zurich


import sentence_alignment
import visualization
import os



def main():

	server_shortname = input('Please enter the path to your new, empty directory where all data should be stored: ')

	os.mkdir(f'{server_shortname}/aligned')
	os.mkdir(f'{server_shortname}/plots')
	
	sentence_alignment.run(server_shortname)

	visualization.statistics_sentences('mitenand', server_shortname)
	visualization.statistics_sentences('helveticus', server_shortname)

	visualization.statistics_alignments('mitenand', server_shortname)
	visualization.statistics_alignments('helveticus', server_shortname)

	visualization.plotting_alignments('mitenand', server_shortname)
	visualization.plotting_alignments('helveticus', server_shortname)



if __name__ == '__main__':
	main()