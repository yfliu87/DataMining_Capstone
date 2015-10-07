'''
1. pre-process hygiene.dat file, 
   use nltk to process each review including removing stop words, stemming, etc
2. build word bag from all reviews
3. convert each file to array representation
4. train SVC model
5. predict new reviews
'''

review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task6/Hygiene/hygiene.dat'
def read_reviews(review_file):
	review_map = {}

	import codecs
	reader = codecs.open(review_file, 'r', 'utf-8')

	line = reader.readline()

	counter = 1
	while line:
		reviews = line.split(' &#160')

		review_map[counter] = [rev.strip().lower() for rev in reviews]

		counter += 1
		line = reader.readline()

	return review_map


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


if __name__ == '__main__':
	rev_map = read_reviews(review_file)
	processed_review = preprocess(rev_map)