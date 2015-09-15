from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline 
from gensim import models
from gensim import matutils

#www.open-open.com/lib/view/open1411355493171.html

basePath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/'
categoryPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/categories'
ldaPath = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/lda'

def getFiles():
	file_list = []
	import os
	for (dirpath, dirnames, filenames) in os.walk(categoryPath):
		for filename in filenames:
			target_file = os.path.join(dirpath, filename)

			file_list.append(target_file)

	return file_list

def readReview(fileList):
	reviews = []

	for f in fileList:
		reader = open(f, 'r')
		line = reader.readline()
		curReview = []

		while line:
			curReview.append(line)
			line = reader.readline()

		reviews.append(curReview)
		reader.close()

	return reviews


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


def ldaProcessing(file_list):
	reviews = []

	for f in file_list:
		curReview = [line.strip() for line in file(f)]

		reviews.append(curReview)

	return reviews

def preprocess(reviews):
	pass

def train_by_lda(reviews):
	pass




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


def match_by_file(fileList, index, dictionary, lda):
	pass

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
	fileList = getFiles()


	reviews = readReview(fileList)
	print reviews

	reviews_processed =	preprocess(reviews)
	(index, dictionary, lda) = train_by_lda(reviews_processed)

	matchByFile(fileList, index, dictionary, lda)

	generateHeatmap('idf')
	generateHeatmap('noidf')


if __name__ == '__main__':
	main()