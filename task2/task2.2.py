from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from gensim import models
from gensim import matutils

#www.open-open.com/lib/view/open1411355493171.html

basePath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/'
categoryPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/categories'
ldaPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lda'
lsiOutputPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lsi'

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

	'''
	reviews = [line.strip() for line in file(target_file)]

	review_lower = [[word for word in review.lower().split()] for review in reviews]

	#separate splitor
	print 'separate splitor'
	from nltk.tokenize import word_tokenize
	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews]
	'''
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


def getSimOfAllReviews(outputpath, fileList, index, dictionary, lsi):
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

		target_rev = preprocess(rev, low_freq_filter=False)
  
		ml_bow = dictionary.doc2bow(target_rev[0]) 
  
		sims = index[lsi[ml_bow]]

		output(outputpath, sims, f, fileList)


def output(outputpath, sim, currentFile, fileList):
	sort_sims = sorted(enumerate(sim), key=lambda item: item[0])
	size = len(sort_sims)

	cur_cuisine = currentFile.split('/')[-1].split('.')[0]

	for i in range(size):
		print sort_sims[i][1]
		outputsim = str(sort_sims[i][1])
		file_index = sort_sims[i][0]
		file_name = fileList[file_index]

		ref_cuisine = file_name.split('/')[-1].split('.')[0]

		if ref_cuisine == cur_cuisine:
			outputsim = str(1.0)

		outputfile = outputpath + '/' + cur_cuisine + '_' + ref_cuisine + '.txt'

		with open(outputfile, 'w') as writer:
			writer.write(outputsim)

'''
def lda(K, numfeatures, texts, num_display_words, outputFolder, first_cuisine, second_cuisine, bIDF):
    K_clusters = K
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=numfeatures, min_df=2, stop_words='english', use_idf=bIDF)

    X = vectorizer.fit_transform(texts)
    
    id2words ={}
    for i,word in enumerate(vectorizer.get_feature_names()):
        id2words[i] = word

    corpus = matutils.Sparse2Corpus(X,  documents_columns=False)

    if bIDF:
    	tfidf = models.TfidfModel(corpus)
    	dataset = tfidf(corpus)
    else:
    	dataset = corpus

    lda = models.ldamodel.LdaModel(dataset, num_topics=K_clusters, id2word=id2words)
        
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
'''

def generateHeatmap(path, config):
	import plotly.plotly as py
	import plotly.graph_objs as go

	cuisines_sim, cuisine_list = get_cuisine_similarity(path)
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
	layout = go.Layout(title = 'similarity_' + config, xaxis = dict(ticks = ''), yaxis = dict(ticks = ''))
	fig = go.Figure(data = raw_data, layout = layout)
	url = py.plot(fig, filename = 'similarity_' + config, validate = True)


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

			print 'similarity of ', cuisines[0], cuisines[1], 'is: ', cuisine_sim[cuisines[0] + cuisines[1]]


	print 'get cuisine similarity done'
	return cuisine_sim, list(set(cuisine_list))


def getSim(file):
	reader = open(file, 'r')
	line = reader.readline()
	reader.close()
	return float(line)


def main():
	fileList = getFiles()

	reviews = readReview(fileList)

	reviews_processed =	preprocess(reviews)

	#all reviews have been processed by lsi model
	(index, dictionary, lsi) = train_by_lsi(reviews_processed)
	

	getSimOfAllReviews(lsiOutputPath, fileList, index, dictionary, lsi)

	generateHeatmap(lsiOutputPath, 'LSI_TFIDF')

	'''
	(index, dictionary, lda) = train_by_lda(reviews_processed)

	matchByFile(fileList, index, dictionary, lda)

	
	generateHeatmap('noidf')
	'''


if __name__ == '__main__':
	main()