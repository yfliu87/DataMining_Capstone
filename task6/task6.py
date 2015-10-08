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

review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/Hygiene/hygiene.dat'
processed_training_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/processed_training_rev.txt'
processed_test_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/processed_testing_rev.txt'
word_bank_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/word_bank.txt'

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

	#remove word whose frequency is less than 3
	all_stems = sum(review_stemmed, [])
	stems_lt_three = set(stem for stem in set(all_stems) if all_stems.count(stem) <= 2)
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
	word_bank = set()

	for rev_id, reviews in word_bag.items():
		for rev in reviews:
			word_bank.add(rev)

	return word_bank


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


if __name__ == '__main__':
	training_rev_map, test_rev_map = read_reviews(review_file)
	processed_training_review = preprocess(training_rev_map)
	write_to_disk(processed_training_review, processed_training_file)

	processed_test_review = preprocess(test_rev_map)
	write_to_disk(processed_test_review, processed_test_file)

	word_bag = build_word_bag(processed_training_review, processed_test_review)
	word_bank = build_word_bank(word_bag)
	write_word_bank(word_bank)


