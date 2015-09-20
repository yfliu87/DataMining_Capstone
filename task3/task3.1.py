import os
import json

base_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3'
business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
business_review_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
business_tip_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_tip.json'
manual_annotation_task_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3/manualAnnotationTask'
validate_output_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3/validateOutput'

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


def record_tip(business_id_tip_map, bus_id, tip):
	if bus_id not in business_id_tip_map:
		business_id_tip_map[bus_id] = []

	business_id_tip_map[bus_id].append(tip)


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


def read_tip_related_to_business_id(business_category_id_map):
	reader = open(business_tip_file_path, 'r')
	business_id_tip_map = {}

	line = reader.readline()

	while line:
		json_line = json.loads(line)
		bus_id = json_line['business_id'] 

		if included_in_target_business_type(bus_id, business_category_id_map):
			record_tip(business_id_tip_map, bus_id, json_line['text'])

		line = reader.readline()

	reader.close()
	return business_id_tip_map 


def output_category_review_to_disk(business_category_id_map, business_id_review_map, business_id_tip_map):
	import codecs

	for cat in business_category_id_map:
		reviews = []
		business_ids = business_category_id_map[cat]

		for bus_id in business_ids:
			if bus_id in business_id_review_map: 
				reviews += business_id_review_map[bus_id]

			if bus_id in business_id_tip_map:
				reviews += business_id_tip_map[bus_id]


		file_name = base_path + '/categoryReviewMap/' + cat + '_tip_reviews.txt'

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
	#print "review tokenize done"

	#remove stop words
	from nltk.corpus import stopwords
	english_stopwords = stopwords.words('english')
	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]
	#print 'remove stop words done'

	#remove punctuations
	english_punctuations = [',','.',':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]
	#print 'remove punctuations done'

	#stemming
	from nltk.stem.lancaster import LancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]
	#print 'stemming done'

	return review_stemmed


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


def validate_dishes(target_cuisine, reviews, output_file_path, lsi_index, lsi_dictionary, lsi):
	for (dirpath, dirnames, filenames) in os.walk(base_path + '/manualAnnotationTask'):
		for filename in filenames:
			if filename.split('.')[0] == target_cuisine:
				target_file = os.path.join(dirpath, filename)
				output_file = output_file_path + '/' + target_cuisine + '.txt'
				check_dish(target_file, reviews, output_file, lsi_index, lsi_dictionary, lsi)
				break


def check_dish(target_file, reviews, output_file_path, lsi_index, lsi_dictionary, lsi):
	result = []
	reader = open(target_file,'r')
	line = reader.readline()

	while line:
		dish = line.split('\t')[0]
		target_dishes = dish.split()
		target_text = preprocess(target_dishes)

		ml_dish = target_text[0]
		ml_bow = lsi_dictionary.doc2bow(ml_dish)
		ml_lsi = lsi[ml_bow]
		sims = lsi_index[ml_lsi]
		sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

		found_match = False
		for item in sort_sims:
			review = reviews[item[0]]

			if dish in review:
				found_match = True
				break

		if found_match:
			result.append((dish, 1))
		else:
			result.append((dish, 0))

		line = reader.readline()

	reader.close()

	output_to_disk(output_file_path, result)


def output_to_disk(output_file_path, dish_appearance):
	writer = open(output_file_path, 'a')

	for item in dish_appearance:
		writer.write('\t'.join([item[0], str(item[1])]))
		writer.write('\n')

	writer.close()


if __name__ == '__main__':
	target_business_type = get_target_business_type()
	business_category_id_map = build_business_type_id_map(target_business_type)
	business_id_review_map = read_review_related_to_business_id(business_category_id_map)
	business_id_tip_map = read_tip_related_to_business_id(business_category_id_map)
	output_category_review_to_disk(business_category_id_map, business_id_review_map, business_id_tip_map)
	
	'''
	reviews = read_review('Chinese')
	reviews_processed =	preprocess(reviews)
	(lsi_index, lsi_dictionary, lsi) = train_by_lsi(reviews_processed)	

	validate_dishes('Chinese', reviews, validate_output_file_path, lsi_index, lsi_dictionary, lsi)
	'''
