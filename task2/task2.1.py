from nltk.cluster.util import cosine_distance
from gensim import models


basePath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/'
categoryPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/categories'
cosineDistancePath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/cosine_distance'


def findSimilarity():
	import os
	import numpy as np
	from gensim import corpora

	list_of_files = [] 
	for (dirpath, dirnames, filenames) in os.walk(categoryPath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			list_of_files.append(target_file)

	for i in range(0, 50):
		ref = list_of_files[i]
		refDoc = preprocess(ref)
		refDict = buildDict(refDoc)

		for j in range(i + 1, 50, 1):
			candidate = list_of_files[j]
			candidateDoc = preprocess(candidate)
			candidateDict = buildDict(candidateDoc)

			combineDict = refDict.copy()
			combineDict.update(candidateDict)
			refWordList = getWordCountList(combineDict.keys(), refDict)
			candidateWordList = getWordCountList(combineDict.keys(), candidateDict)

			refArray = np.asarray(refWordList, dtype=int).reshape(-1)
			candidateArray = np.asarray(candidateWordList, dtype=int).reshape(-1)

			sim = cosine_distance(refArray, candidateArray)

			outputfile = getOutputFileName(ref, candidate)

			with open(outputfile, 'w') as writer:
				writer.write(str(np.asscalar(sim)))

	#print 'find similarity done'


def getOutputFileName(first, second):
	cuisine1 = first.split('/')[-1].split('.')[0]
	cuisine2 = second.split('/')[-1].split('.')[0]

	return cosineDistancePath + '/' + cuisine1 + '_' + cuisine2 + '.txt'


def preprocess(target_file):
	import nltk

	reviews = [line.strip() for line in file(target_file)]

	review_lower = [[word for word in review.lower().split()] for review in reviews]

	#separate splitor
	from nltk.tokenize import word_tokenize
	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews]

	#remove stop words
	from nltk.corpus import stopwords
	english_stopwords = stopwords.words('english')

	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]

	#remove punctuations
	english_punctuations = [',','.',':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]

	#stemming
	from nltk.stem.lancaster import LancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]

	#remove word whose frequency is 1
	all_stems = sum(review_stemmed, [])
	stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
	final_review = [[stem for stem in text if stem not in stems_once] for text in review_stemmed]

	return final_review


def getWordCountList(keys, dictionary):
	result = []

	for key in keys:
		if key not in dictionary:
			result.append(0)
		else:
			result.append(dictionary[key])

	return result


def buildDict(texts):
	result = {}

	for text in texts:
		for word in text:
			if word in result:
				result[word] += 1
			else:
				result[word] = 1

	return result


def sortByValue(dictionary):
	return sorted(dictionary.items(), key=lambda d:d[1], reverse=True)


def generateHeatmap(idfConfig):
	import plotly.plotly as py
	import plotly.graph_objs as go

	cuisines_sim, cuisine_list = get_cuisine_similarity(cosineDistancePath)
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

			cuisine_sim[cuisines[0] + cuisines[1]] = getSim(target_file)
			cuisine_list += [cuisines[0], cuisines[1]]

			#print 'similarity of ', cuisines[0], cuisines[1], 'is: ', cuisine_sim[cuisines[0] + cuisines[1]]


	print 'get cuisine similarity done'
	return cuisine_sim, list(set(cuisine_list))


def getSim(file):
	reader = open(file, 'r')
	line = reader.readline()
	reader.close()
	return float(line)


def main():
	findSimilarity()
	generateHeatmap('cosine_distance')

if __name__ == '__main__':
	main()