'''
1. pre-process hygiene.dat file 
   use nltk to process each review including removing stop words, stemming, etc
2. build word bag from training and testing reviews
3. write word bag and processed reviews into disk file
3. read in disk file and convert to array representation according to word bag
4. train SVC model
5. predict new reviews
'''

import codecs
import numpy as np

review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/Hygiene/hygiene.dat'
processed_training_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/processed_training_rev.txt'
processed_test_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/processed_testing_rev.txt'
word_bank_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/word_bank.txt'
segPhrase_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/frequent_phrase.csv'
word_phrase_bank_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/word_phrase_bank.txt' 
random_bag_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/random_word_bag.txt'
sorted_bag_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/sorted_word_bag.txt'
hygiene_additional_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/Hygiene/hygiene.dat.additional'
hygiene_label_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/Hygiene/hygiene.dat.labels'
testing_label_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/testing_label_file.txt'

def read_reviews(review_file):
	training_review_map = {}
	test_review_map = {}

	reader = codecs.open(review_file, 'r', 'utf-8')
	line = reader.readline()
	counter = 1
	
	while line:
		reviews = line.split(' &#160')

		training_review_map[counter] = [rev.strip().lower() for rev in reviews]

		counter += 1

		if counter >= 547:
			break

		line = reader.readline()

	line = reader.readline()

	while line:
		reviews = line.split(' &#160')

		test_review_map[counter] = [rev.strip().lower() for rev in reviews]

		counter += 1
		line = reader.readline()

	reader.close()
	return training_review_map, test_review_map


def preprocess(review_map):
	for rev_id, reviews in review_map.items():
		review_map[rev_id] = process(reviews)

	return review_map

def process(reviews):
	#separate splitor
	from nltk.tokenize import word_tokenize
	review_tokenized = [[word.lower() for word in word_tokenize(review.decode('utf-8'))] for review in reviews]

	#remove stop words
	from nltk.corpus import stopwords
	english_stopwords = stopwords.words('english')

	review_filterd_stopwords = [[word for word in review if not word in english_stopwords] for review in review_tokenized]

	#remove punctuations
	english_punctuations = [',','.','...', ':',';','?','(',')','&','!','@','#','$','%']
	review_filtered = [[word for word in review if not word in english_punctuations] for review in review_filterd_stopwords]

	#stemming
	from nltk.stem.lancaster import LancasterStemmer
	st = LancasterStemmer()
	review_stemmed = [[st.stem(word) for word in review] for review in review_filtered]

	#remove word whose frequency is less than 5
	all_stems = sum(review_stemmed, [])
	stems_lt_three = set(stem for stem in set(all_stems) if all_stems.count(stem) < 5)
	final_review = [[stem for stem in text if stem not in stems_lt_three] for text in review_stemmed]

	return final_review


def build_word_bag(processed_training_review, processed_test_review):
	word_bag = {}

	for rev_id, review in processed_training_review.items():
		word_bag[rev_id] = {}
		for sub_rev in review:
			for rev in sub_rev:
				if rev not in word_bag[rev_id]:
					word_bag[rev_id][rev] = 1
				else:
					word_bag[rev_id][rev] += 1

	for rev_id, review in processed_test_review.items():
		word_bag[rev_id] = {}
		for sub_rev in review:
			for rev in sub_rev:
				if rev not in word_bag[rev_id]:
					word_bag[rev_id][rev] = 1
				else:
					word_bag[rev_id][rev] += 1

	return word_bag


def build_word_bank(word_bag):
	writer = codecs.open(random_bag_file, 'w')

	for rev_id, reviews in word_bag.items():
		for rev in reviews:
			writer.write(rev + '\t' + str(reviews[rev]) + '\n')

	writer.close()

	reader = codecs.open(random_bag_file, 'r')
	line = reader.readline()
	temp_bag = {}

	while line:
		word = line.split('\t')[0]
		count = int(line.split('\t')[1].split('\n')[0])

		if word not in temp_bag:
			temp_bag[word] = count
		else:
			temp_bag[word] += count

		line = reader.readline()

	reader.close()

	import collections
	sorted_bag = collections.OrderedDict(sorted(temp_bag.items(), key=lambda k:k[1], reverse=True))
	#sorted_bag = sorted(temp_bag.items(), lambda x,y : cmp(x[1], y[1]), reverse=True)

	return sorted_bag.keys()


def write_to_disk(review_map, output_file):
	writer = codecs.open(output_file, 'w', 'utf-8')

	for rev_id, review in review_map.items():
		message = ''
		for rev in review:
			message += ' '.join(rev) + ' '

		writer.write(str(rev_id) + '\t' + message[0:-1] + '\n')

	writer.close()


def write_word_bank(word_bank):
	writer = codecs.open(word_bank_file, 'w', 'utf-8')

	for word in word_bank:
		writer.write(word + '\n')

	writer.close()


def read_from_file(word_bank_file):
	word_bank = list()
	reader = codecs.open(word_bank_file, 'r', 'utf-8')
	line = reader.readline()

	while line:
		word_bank.append(line.split('\n')[0])
		line = reader.readline()

	reader.close()
	return word_bank


def read_phrase_file(segPhrase_file, word_bank):
	phrase_list = []
	reader = codecs.open(segPhrase_file, 'r')
	line = reader.readline()

	counter = 1
	while line:
		phrase = line.split(',')[0].replace('_', ' ')
		phrase_list.append(phrase)
		counter += 1

		if counter > 1000:
			break

		line = reader.readline()

	reader.close()
	return phrase_list + word_bank


def build_array_rep_from_file(processed_file, word_bank):
	array_rep = {} 
	reader = codecs.open(processed_file, 'r', 'utf-8')
	line = reader.readline()

	while line:
		rev_id = int(line[0:-1].split('\t')[0])
		review = line[0:-1].split('\t')[1]
		revs = review.split(' ')

		array_rep[rev_id] = []
		for word in word_bank:
			array_rep[rev_id].append(line.count(word))

		line = reader.readline()

	reader.close()
	return array_rep


def read_label(hygiene_label_file):
	training_label = {}

	reader = codecs.open(hygiene_label_file, 'r' ,'utf-8')
	line = reader.readline()
	counter = 1

	while line:
		training_label[counter] = int(line)
		line = reader.readline()
		counter += 1

		if counter >= 547:
			break

	reader.close()
	return training_label


def read_rev_count(hygiene_additional_file):
	training_rev_count = {}
	testing_rev_count = {}

	reader = codecs.open(hygiene_additional_file, 'r')
	line = reader.readline()

	counter = 1
	while line:
		training_rev_count[counter] = int(line.split(',')[-2])
		counter += 1

		if counter >= 547:
			break

		line = reader.readline()

	line = reader.readline()
	while line:
		testing_rev_count[counter] = int(line.split(',')[-2])

		counter += 1
		line = reader.readline()

	reader.close()
	return training_rev_count, testing_rev_count


def read_avg_rate(hygiene_additional_file):
	training_avg_rate_list = {}
	testing_avg_rate_list = {}
	reader = codecs.open(hygiene_additional_file, 'r')
	line = reader.readline()

	counter = 1
	while line:
		items = line.split(',')
		training_avg_rate_list[counter] = items[-1]
		counter += 1

		if counter >= 547:
			break

		line = reader.readline()

	line = reader.readline()
	while line:
		items = line.split(',')
		testing_avg_rate_list[counter] = items[-1]

		counter += 1
		line = reader.readline()

	reader.close()
	return training_avg_rate_list, testing_avg_rate_list


def train_SVC_model(training_review_array_rep, training_label, training_avg_rate):
	rep_list = []
	label_list = []

	for rev_id, array_rep in training_review_array_rep.items():
		array_rep.append(training_avg_rate[rev_id])
		rep_list.append(array_rep)
		label_list.append(training_label[rev_id])

	from sklearn.svm import SVC
	model = SVC()
	model.fit(np.array(rep_list), np.array(label_list))
	return model


def train_Bayes_model(training_review_array_rep, training_label):
	rep_list = []
	label_list = []

	for rev_id, array_rep in training_review_array_rep.items():
		rep_list.append(array_rep)
		label_list.append(training_label[rev_id])

	from sklearn.naive_bayes import GaussianNB
	model = GaussianNB()
	model.fit(np.array(rep_list), np.array(label_list))
	return model


def train_LDA_model(training_review_array_rep, training_label,training_avg_rate):
	rep_list = []
	label_list = []

	for rev_id, array_rep in training_review_array_rep.items():
		array_rep.insert(0, training_avg_rate[rev_id])
		rep_list.append(array_rep)
		label_list.append(training_label[rev_id])

	from sklearn.lda import LDA
	model = LDA()
	model.fit(np.array(rep_list), np.array(label_list))
	return model


def predict(model, processed_testing_review_array_representation, testing_avg_rate):
	testing_rep_list = []

	for rep_id, array_rep in processed_testing_review_array_representation.items():
		array_rep.append(testing_avg_rate[rep_id])
		testing_rep_list.append(array_rep)

	testing_label = model.predict(np.array(testing_rep_list))

	with codecs.open(testing_label_file, 'w') as writer:
		for label in testing_label:
			writer.write(str(label) + '\n')

	return testing_label

def write(word_phrase_bank):
	with codecs.open(word_phrase_bank_file, 'w') as writer:
		writer.write('\n'.join(word_phrase_bank))


if __name__ == '__main__':
	'''
	training_rev_map, test_rev_map = read_reviews(review_file)
	processed_training_review = preprocess(training_rev_map)
	write_to_disk(processed_training_review, processed_training_file)
	processed_test_review = preprocess(test_rev_map)
	write_to_disk(processed_test_review, processed_test_file)

	word_bag = build_word_bag(processed_training_review, processed_test_review)
	word_bank = build_word_bank(word_bag)
	write_word_bank(word_bank)
	
	word_bank = read_from_file(word_bank_file)
	word_phrase_bank = read_phrase_file(segPhrase_file, word_bank)
	write(word_phrase_bank)
	processed_training_review_array_representation = build_array_rep_from_file(processed_training_file, word_phrase_bank)
	processed_testing_review_array_representation = build_array_rep_from_file(processed_test_file, word_phrase_bank)
	'''

	training_label = read_label(hygiene_label_file)
	training_rev_count, testing_rev_count = read_rev_count(hygiene_additional_file)
	training_avg_rate, testing_avg_rate = read_avg_rate(hygiene_additional_file)

	svc_model = train_SVC_model(processed_training_review_array_representation, training_label, training_avg_rate)
	svc_test_label = predict(svc_model, processed_testing_review_array_representation, testing_avg_rate)

	'''
	bayes_model = train_Bayes_model(processed_training_review_array_representation, training_label)
	bayes_test_label = predict(bayes_model, processed_testing_review_array_representation)
	lda_model = train_LDA_model(processed_training_review_array_representation, training_label, training_avg_rate)
	lda_test_label = predict(lda_model, processed_testing_review_array_representation, testing_avg_rate)
	'''
