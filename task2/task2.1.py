import json
from nltk.cluster.util import cosine_distance
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from gensim import models
from gensim import matutils


basePath = '/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/'
categoryPath = '/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/categories'
ldaPath = '/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lda'

def restaurantIDCategoryMapping(businessFile):
	reader = open(businessFile, 'r')
	reader.seek(0)
	idCategoryMap = {}
	line = reader.readline()
	while line:
		business_json = json.loads(line)
		businessID = business_json['business_id']
		businessCategory = business_json['categories']

		if 'Restaurants' not in businessCategory:
			line = reader.readline()
			continue

		if businessID not in idCategoryMap:
			idCategoryMap[businessID] = []

		for cat in businessCategory:
			if cat != 'Restaurants':
				cat = cat.replace('/','-')
				idCategoryMap[businessID].append(cat)

		line = reader.readline()

	reader.close()

	for busID in idCategoryMap.keys():
		idCategoryMap[busID] = list(set(idCategoryMap[busID]))

	print 'build restaurantID category map finished'
	return idCategoryMap


def reviewRestaurantIDMapping(busIDCategoryMap, reviewFile, cuisineReviewMap):
	busIDs = busIDCategoryMap.keys()

	reader = open(reviewFile, 'r')
	reader.seek(0)

	line = reader.readline()

	count = 0
	while line:
		count += 1
		review_json = json.loads(line)
		busID = review_json['business_id']

		if busID not in busIDs:
			line = reader.readline()
			continue	

		reviews = review_json['text']

		categories = busIDCategoryMap[busID]
		updateCategoryWithReviews(reviews, categories, cuisineReviewMap)

		line = reader.readline()

	print "total reviews: ", count
	reader.close()

	outputFileByCuisine(cuisineReviewMap)


def outputFileByCuisine(cuisineReviewMap):
	for cuisine in cuisineReviewMap.keys():
		targetFile = basePath + 'output/' + cuisine + '.txt'

		with open(targetFile, 'w') as writer:
				writer.write('\n\n'.join(cuisineReviewMap[cuisine]))


def updateCategoryWithReviews(reviews, categories, cuisineReviewMap):
	for review in reviews.split('\n\n'):
		for category in categories:
			if category not in cuisineReviewMap:
				cuisineReviewMap[category] = []

			cuisineReviewMap[category].append(review)

''''
def findSimilarity(fileFolder):
	import os
	import numpy as np

	list_of_files = [] 
	for (dirpath, dirnames, filenames) in os.walk(fileFolder):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			list_of_files.append(target_file)

	print "total file: ", len(list_of_files)

	simDoc = []

	for i in range(len(list_of_files)):
		ref = list_of_files[i]
		refDoc = train(ref)

		for j in range(i + 1, len(list_of_files) , 1):
			candidate = list_of_files[j]
			candidateDoc = train(candidate)

			ref = np.asarray(refDoc[0, :].todense()).reshape(-1)
			candidate = np.asarray(candidateDoc[0, :].todense()).reshape(-1)

			sim = getSimilarity(refDoc, candidateDoc)

			print 'similarity between ' + list_of_files[i] + "," + list_of_files[j] + ": " + str(sim)
			simDoc.append((i, j, sim))

	return simDoc, list_of_files
'''

def findSimilarity():
	file_list = []
	import os
	for (dirpath, dirnames, filenames) in os.walk(categoryPath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			file_list.append(target_file)

	return file_list

'''
def lsiProcessing(file_list):

	for i in range(0, len(file_list) , 1):
		for j in range(i + 1, len(file_list),  1):
			ref = file_list[i]
			candidate = file_list[j]

			ref_processed = preprocess(ref)
			candidate_processed = preprocess(candidate)

			texts = [ref_processed, candidate_processed]

			dictionary = corpora.Dictionary(texts)
			corpus = [dictionary.doc2bow(text) for text in texts]
			print "corpus: ", corpus

			#calculate training document
			tfidf = models.TfidfModel(corpus)
			corpus_tfidf = tfidf[corpus]
			print "tfidf corpus: ", corpus_tfidf

			lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics = 1)
			corpus_lsi = lsi[corpus_tfidf]

			for doc in corpus_lsi:
				print "lsi doc: ", doc

			lda_notfidf = models.LdaModel(corpus, id2word=dictionary, num_topics=2) 
			print "lda_NoTfidf topic: ", lda_notfidf.print_topics(2)

			lda_tfidf = models.LdaModel(corpus_tfidf, id2word=dictionary, num_topics=2)
			print "lda_Tfidf topic: ", lda_tfidf.print_topics(2)


def preprocess(target_file):
	import nltk

	reviews = [line.strip() for line in file(target_file)]

	review_lower = [[word for word in review.lower().split()] for review in reviews]

	#separate splitor
	print 'separate splitor'
	from nltk.tokenize import word_tokenize
	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews]

	#remove stop words
	print 'remove stop words'
	from nltk.corpus import stop_words
	english_stopwords = stopwords.words('english')

	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]

	#remove punctuations
	print 'remove punctuations'
	english_punctuations = [',','.',':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]
	print review_filtered[0]

	#stemming
	print 'stemming'
	from nltk.stem.lancaster import CancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]

	#remove word whose frequency is 1
	print 'remove Freq = 1'
	all_stems = sum(review_stemmed, [])
	stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
	final_review = [[stem for stem in text if stem not in stems_once] for text in review_stemmed]

	return final_review


def train(file):
	import re
	words = []

	reader = open(file, 'r')
	for line in reader:
		for word in line.strip().split(' '):
			words.append(re.sub(r'\W', '', word))

	reader.close()

	vectorizer = CountVectorizer(stop_words="english")
	transformer = TfidfTransformer()
	return transformer.fit_transform(vectorizer.fit_transform(words))
	#pipeline = Pipeline([("vect", CountVectorizer(min_df=0, stop_words="english")), ("tfidf", TfidfTransformer(use_idf=False))])
	#return pipeline.fit_transform(words)


def getSimilarity(ref, candidate):
	return cosine_distance(ref, candidate)
'''

def ldaProcessing(file_list):
	cuisine_map = [] 
	for i in range(0, len(file_list) , 1):
		for j in range(i + 1, len(file_list),  1):
			ref = file_list[i]
			candidate = file_list[j]

			reviews = [line.strip() for line in file(ref)]
			reviews += [line.strip() for line in file(candidate)]

			first_quisine = ref.split('/')[-1].split('.')[0]
			second_quisine = candidate.split('/')[-1].split('.')[0]

			if (first_quisine, second_quisine) in cuisine_map or (second_quisine, first_quisine) in cuisine_map:
				continue
			else:
				cuisine_map.append((first_quisine, second_quisine))

			#no tfidf
			lda(1, 5, reviews, 5, ldaPath, first_quisine, second_quisine, False)

			#tfidf
			lda(1, 5, reviews, 5, ldaPath, first_quisine, second_quisine, True)


def lda(K, numfeatures, texts, num_display_words, outputFolder, first_cuisine, second_cuisine, bIDF):
    K_clusters = K
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=numfeatures, min_df=2, stop_words='english', use_idf=bIDF)

    X = vectorizer.fit_transform(texts)
    
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)
    lda = models.ldamodel.LdaModel(corpus, num_topics=K_clusters, id2word=id2words)
        
    output_text = []
    for i, item in enumerate(lda.show_topics(num_topics=K_clusters, num_words=num_display_words, formatted=False)):
        output_text.append("Topic: " + str(i))
        for weight,term in item:
            output_text.append( term + " : " + str(weight) )

    if bIDF:
    	outputfile = outputFolder + '/idf/' + first_cuisine + '_' + second_cuisine + '.txt'
    else:
    	outputfile = outputFolder + '/noidf/' + first_cuisine + '_' + second_cuisine + '.txt'

    with open ( outputfile, 'w' ) as f:
        f.write('\n'.join(output_text))


def generateHeatmap(idfConfig):
	import plotly.plotly as py
	import plotly.graph_objs as go

	cuisines_sim, cuisine_list = get_cuisine_similarity(ldaPath + '/' + idfConfig)
	data = []

	for first_cuisine in cuisine_list:
		new_row = []
		for second_cuisine in cuisine_list:
			if (first_cuisine + second_cuisine) in cuisines_sim:
				new_row.append(float(cuisines_sim[first_cuisine + second_cuisine]))
			elif (second_cuisine + first_cuisine) in cuisines_sim:
				new_row.append(float(cuisines_sim[second_cuisine + first_cuisine]))
			elif first_cuisine == second_cuisine:
				new_row.append(1.0)
			else:
				new_row.append(0.0)	

		data.append(new_row)


	raw_data = go.Data([go.Heatmap(z = data, x = cuisine_list, y = cuisine_list, colorscale = 'Viridis')])
	layout = go.Layout(title = 'cuisine_similarity_' + idfConfig, xaxis = dict(ticks = ''), yaxis = dict(ticks = ''))
	fig = go.Figure(data = raw_data, layout = layout)
	url = py.plot(fig, filename = 'cuisine_similarity_' + idfConfig, validate = True)


def get_cuisine_similarity(filepath):
	cuisine_list = []
	cuisine_sim = {}

	import os
	for (dirpath, dirnames, filenames) in os.walk(filepath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			cuisines = filename.split('.')[0].split('_')

			cuisine_sim[cuisines[0] + cuisines[1]] = getMaxSim(target_file)
			cuisine_list += [cuisines[0], cuisines[1]]

			print 'similarity of ', cuisines[0], cuisines[1], 'is: ', cuisine_sim[cuisines[0] + cuisines[1]]


	print 'get cuisine similarity done'
	return cuisine_sim, list(set(cuisine_list))


def getMaxSim(file):
	reader = open(file, 'r')
	line = reader.readline()

	maxScore = 0.0
	while line:
		score = line.split(':')[1]

		if score > maxScore:
			maxScore = score

		line = reader.readline()

	reader.close()
	return maxScore


def main():
	'''
	restaurantFile = basePath + 'yelp_academic_dataset_business.json'
	reviewFile = basePath + 'yelp_academic_dataset_review.json'
	cuisineCategoryMap = {}

	idCategoryMap = restaurantIDCategoryMapping(restaurantFile)

	reviewRestaurantIDMapping(idCategoryMap, reviewFile, cuisineCategoryMap)
	'''
	fileList = findSimilarity()

	#lsiProcessing(fileList)
	#ldaProcessing(fileList)
	generateHeatmap('idf')
	generateHeatmap('noidf')

if __name__ == '__main__':
	main()