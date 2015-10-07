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

		review_map[counter] = reviews

		counter += 1
		line = reader.readline()

	return review_map


if __name__ == '__main__':
	rev_map = read_reviews(review_file)
	print rev_map[1]

