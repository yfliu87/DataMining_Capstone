import os
import json

base_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3'
business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
business_review_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
manual_annotation_task_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3/manualAnnotationTask'


def get_target_business_type():
	result = []

	for (dirpath, dirnames, filenames) in os.walk(manual_annotation_task_path):
		for filename in filenames:
			result.append(filename.split('.')[0])

	return result


def build_business_type_id_map(business_type_list):
	result = {}

	reader = open(business_file_path, 'r')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		categories = json_line['categories']

		for cat in categories:
			if cat in business_type_list:
				business_id = json_line['business_id']

				if cat not in result:
					result[cat] = set() 

				result[cat].add(business_id)

		line = reader.readline()

	reader.close()
	return result


def included_in_target_business_type(current_id, business_category_id_map):
	for cat in business_category_id_map:
		if current_id in business_category_id_map[cat]:
			return True

	return False


def record_review(business_id_review_map, bus_id, review):
	if bus_id not in business_id_review_map:
		business_id_review_map[bus_id] = []

	business_id_review_map[bus_id].append(review)


def read_review_related_to_business_id(business_category_id_map):
	reader = open(business_review_file_path, 'r')
	business_id_review_map = {}

	line = reader.readline()

	while line:
		json_line = json.loads(line)
		bus_id = json_line['business_id'] 

		if included_in_target_business_type(bus_id, business_category_id_map):
			record_review(business_id_review_map, bus_id, json_line['text'])

		line = reader.readline()

	reader.close()
	return business_id_review_map 


def output_category_review_to_disk(business_category_id_map, business_id_review_map):
	import codecs

	for cat in business_category_id_map:
		reviews = []
		business_ids = business_category_id_map[cat]

		for bus_id in business_ids:
			if bus_id not in business_id_review_map:
				continue 

			reviews += business_id_review_map[bus_id]

		file_name = base_path + '/categoryReviewMap/' + cat + '_reviews.txt'

		with codecs.open(file_name, 'a', encoding='utf-8') as writer:
			writer.write(''.join(reviews))


def read_review(target_cuisine):
	reviews = []
	for (dirpath, dirnames, filenames) in os.walk(base_path + '/categoryReviewMap'):
		for filename in filenames:
			if filename.split('_')[0] == target_cuisine:
				target_file = os.path.join(dirpath, filename)
				reviews = read_file(target_file)
				break

	return reviews

def read_file(target_file):
	review = []
	reader = open(target_file, 'r')
	line = reader.readline()

	while line:
		if line != '\n':
			review.append(line)

		line = reader.readline()

	reader.close()
	return review

def preprocess(reviews):
	import nltk
	from nltk.tokenize import word_tokenize

	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews] 
	print "review tokenize done"

	#remove stop words
	from nltk.corpus import stopwords
	english_stopwords = stopwords.words('english')
	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]
	print 'remove stop words done'

	#remove punctuations
	english_punctuations = [',','.',':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]
	print 'remove punctuations done'

	#stemming
	from nltk.stem.lancaster import LancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]
	print 'stemming done'

	return review_stemmed


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


if __name__ == '__main__':
	target_business_type = get_target_business_type()
	business_category_id_map = build_business_type_id_map(target_business_type)
	business_id_review_map = read_review_related_to_business_id(business_category_id_map)
	output_category_review_to_disk(business_category_id_map, business_id_review_map)

	reviews = read_review('Chinese')
	reviews_processed =	preprocess(reviews)

	(lda_index, lda_dictionary, lda) = train_by_lda(reviews_processed)	
	#validate_dishes(validate_output_file_path, lda_index, lda_dictionary, lda)
	#getSimOfAllReviews(lsiOutputPath, fileList, lsi_index, lsi_dictionary, lsi)
	#generateSimilarityMap(lsiOutputPath, 'LSI_TFIDF')
