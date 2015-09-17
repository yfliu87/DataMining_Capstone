from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from gensim import models
from gensim import matutils

#www.open-open.com/lib/view/open1411355493171.html

basePath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2'
categoryPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/categories'
ldaPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lda'
lsiOutputPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lsi'
ldaOutputPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lda'

def getFiles():
	import os

	file_list = []
	count = 0
	for (dirpath, dirnames, filenames) in os.walk(categoryPath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			file_list.append(target_file)

			count += 1

			if count >= 50:
				break

	print "read in " + str(len(file_list)) + " files"
	return file_list

def readReview(fileList):
	reviews = []

	for f in fileList:
		reader = open(f, 'r')
		line = reader.readline()
		curReview = ''

		while line:
			curReview += line
			line = reader.readline()

		reviews.append(curReview)
		reader.close()

	print "read reviews from file list done"
	return reviews


def preprocess(reviews, low_freq_filter = True):
	import nltk
	from nltk.tokenize import word_tokenize

	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews] 
	print "review tokenize done"

	#remove stop words
	from nltk.corpus import stopwords
	english_stopwords = stopwords.words('english')
	print 'remove stop words done'

	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]

	#remove punctuations
	english_punctuations = [',','.',':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]
	print 'remove punctuations done'

	#stemming
	from nltk.stem.lancaster import LancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]
	print 'stemming done'

	#remove word whose frequency is 1
	if low_freq_filter:
		all_stems = sum(review_stemmed, [])
		stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
		final_review = [[stem for stem in text if stem not in stems_once] for text in review_stemmed]
	else:
		final_review = review_stemmed

	print 'remove low freq done'
	print "preprocess of reviews done"
	return final_review


def ldaProcessing(file_list):
	reviews = []

	for f in file_list:
		curReview = [line.strip() for line in file(f)]

		reviews.append(curReview)

	return reviews


def train_by_lsi(reviews):
	from gensim import corpora, models, similarities

	dictionary = corpora.Dictionary(reviews)
	corpus = [dictionary.doc2bow(text) for text in reviews] 
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)
	index = similarities.MatrixSimilarity(lsi[corpus])

	print "train by lsi done"
	return (index, dictionary, lsi)

def train_by_lda(reviews):
	from gensim import corpora, models, similarities

	dictionary = corpora.Dictionary(reviews)
	corpus = [dictionary.doc2bow(text) for text in reviews] 
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]

	lda = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=10)
	index = similarities.MatrixSimilarity(lda[corpus])

	print "train by lda done"
	return (index, dictionary, lda)

def getSimOfAllReviews(outputpath, fileList, index, dictionary, model):
	#go through all reviews and preprocess review
	#then try to get the similarity between current review and all others
	for f in fileList:
		curReview ='' 

		reader = open(f, 'r')
		line = reader.readline()

		while line:
			curReview += line
			line = reader.readline()

		reader.close()

		rev = [curReview]

		target_rev = preprocess(rev, low_freq_filter=True)
  
		ml_bow = dictionary.doc2bow(target_rev[0]) 
  
		sims = index[model[ml_bow]]

		output(outputpath, sims, f, fileList)


def output(outputpath, sim, currentFile, fileList):
	sort_sims = sorted(enumerate(sim), key=lambda item: -item[1])
	size = len(sort_sims)

	cur_cuisine = currentFile.split('/')[-1].split('.')[0]

	for i in range(size):
		outputsim = str(sort_sims[i][1])
		file_index = sort_sims[i][0]
		file_name = fileList[file_index]

		ref_cuisine = file_name.split('/')[-1].split('.')[0]

		if ref_cuisine == cur_cuisine:
			outputsim = str(1.0)

		outputfile = outputpath + '/' + cur_cuisine + '_' + ref_cuisine + '.txt'

		with open(outputfile, 'w') as writer:
			writer.write(outputsim)

	return fileSequence


def generateSimilarityMap(path, cluster_size, config):
	import plotly.plotly as py
	import plotly.graph_objs as go

	similarity_cluster = get_cuisine_similarity(path, cluster_size,config)
	data = []
	cuisine_list = []

	for curVals in similarity_cluster.values():
		intersect = [val for val in curVals if val in cuisine_list]

		for v in intersect:
			cuisine_list.remove(v)
			curVals.remove(v)

		cuisine_list += intersect
		cuisine_list += curVals

	for val in cuisine_list:
		curData = []
		for ref in cuisine_list:
			curData.append(read_similarity_data(val, ref, config))

		data.append(curData)


	raw_data = go.Data([go.Heatmap(z = data, x = cuisine_list, y = cuisine_list, colorscale = 'Viridis')])
	layout = go.Layout(title = 'cluster_' + config, xaxis = dict(ticks = ''), yaxis = dict(ticks = ''))
	fig = go.Figure(data = raw_data, layout = layout)
	url = py.plot(fig, filename = 'cluster_' + config, validate = True)
	

def read_similarity_data(cuisine1, cuisine2, config):

	if 'LDA' in config:
		path = ldaOutputPath
	if 'LSI' in config:
		path = lsiOutputPath

	file_name = path + '/' + cuisine1 + '_' + cuisine2 + '.txt'

	similarity = 0.0
	with open(file_name, 'r') as reader:
		similarity = float(reader.readline())

	return similarity


def get_cuisine_similarity(filepath, cluster_size, config):
	#cuisine_list = []
	cuisine_sim = {}

	import os
	for (dirpath, dirnames, filenames) in os.walk(filepath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			cuisines = filename.split('.')[0].split('_')

			refCuisine = cuisines[0]

			if refCuisine not in cuisine_sim:
				cuisine_sim[refCuisine] = []
				cuisine_sim[refCuisine].append(refCuisine)

			similarity = getSim(target_file)

			if similarity > 0.7:
				cuisine_sim[refCuisine].append(cuisines[1])

			cuisine_sim[refCuisine] = list(set(cuisine_sim[refCuisine]))

			#cuisine_sim[cuisines[0] + cuisines[1]] = getSim(target_file)
			#cuisine_list += [cuisines[0], cuisines[1]]

			#print 'similarity of ', cuisines[0], cuisines[1], 'is: ', cuisine_sim[cuisines[0] + cuisines[1]]

	clustered_sim = findCluster(cuisine_sim, filepath, cluster_size, config)

	print 'get cluster similarity done'
	#return cuisine_sim, list(set(cuisine_list))
	return clustered_sim


def findCluster(cuisine_sim, outputpath, cluster_size, config):
	to_be_deleted = []
	to_be_updated = {}

	previousSize = -1
	while True:
		if previousSize == len(cuisine_sim) or len(cuisine_sim) <= cluster_size:
			break

		previousSize = len(cuisine_sim)
		#print "cuisine sim: ", cuisine_sim

		for cuisine in cuisine_sim.keys():
			if cuisine in to_be_deleted:
				continue

			similar_cuisines = cuisine_sim[cuisine]

			for cui in similar_cuisines:
				if cui not in cuisine_sim or cui == cuisine:
					continue

				to_be_updated[cuisine] = cuisine_sim[cui]
				to_be_deleted.append(cui)

			to_be_deleted = list(set(to_be_deleted))	

		'''
		with open(basePath + '/' + config + '_' + str(cluster_size) + '_clusters.txt', 'a') as writer:
			writer.write("\nto be deleted: \n")
			writer.write(','.join(to_be_deleted))
			writer.write("\n\nto be updated: \n") 
			writer.write('\n'.join(to_be_updated))
		'''

		for item in to_be_updated.keys():
			for val in to_be_updated[item]:
				if val not in cuisine_sim[item]:
					cuisine_sim[item].append(val)

		for item in to_be_deleted:
			del cuisine_sim[item]

		to_be_deleted = []
		to_be_updated.clear()


	with open(basePath + '/' + config + '_' + str(cluster_size) + '_clusters.txt', 'a') as writer:
		for k,v in cuisine_sim.items():
			writer.write(k + ':')
			values = []
			for val in v:
				values.append(val)
				
			writer.write(','.join(values))
			writer.write('\n')

	return cuisine_sim


def getSim(file):
	reader = open(file, 'r')
	line = reader.readline()
	reader.close()
	return float(line)


def main():
	#fileList = getFiles()

	#reviews = readReview(fileList)

	#reviews_processed =	preprocess(reviews)

	#all reviews have been processed by lsi model
	#(lsi_index, lsi_dictionary, lsi) = train_by_lsi(reviews_processed)	
	#getSimOfAllReviews(lsiOutputPath, fileList, lsi_index, lsi_dictionary, lsi)
	#generateSimilarityMap(lsiOutputPath, 'LSI_TFIDF')
	#print "LSI TFIDF similarity done"

	#(lda_index, lda_dictionary, lda) = train_by_lda(reviews_processed)
	#getSimOfAllReviews(ldaOutputPath, fileList, lda_index, lda_dictionary, lda)
	#generateSimilarityMap(ldaOutputPath, 'LDA_TFIDF')
	#print "LDA TFIDF similarity done"


	generateSimilarityMap(lsiOutputPath, 3, 'LSI_TFIDF')
	#generateSimilarityMap(lsiOutputPath, 5, 'LSI_TFIDF')
	generateSimilarityMap(ldaOutputPath, 3, 'LDA_TFIDF')


if __name__ == '__main__':
	main()