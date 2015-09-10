import json
from nltk.cluster.util import cosine_distance
import sklearn.feature_extraction.text import CountVectorizer
import sklearn.feature_extraction.text import TfidfTransformer 


basePath = '/home/yfliu/Documents/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task2/'

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


def findSimilarity(fileFolder):
	list_of_files = [] 
	for (dirpath, dirnames, filenames) in os.walk(fileFolder):
		for filename in filenames:
			target_file = os.sep.join(dirpath, filename)

			list_of_files.append(target_file)

	print "total file: ", len(list_of_files)

	simDoc = []

	for i in range(len(list_of_files)):
		ref = list_of_files[i]
		refDoc = train(ref)

		for j in range(i + 1, len(list_of_files) , 1):
			candidate = list_of_files[j]
			candidateDoc = train(candidate)

			sim = getSimilarity(refDoc, candidateDoc)

			print 'similarity between ' + list_of_files[i] + "," + list_of_files[j] + ": " + str(sim)
			simDoc.append((i, j, sim))

	return simDoc, list_of_files


def train(file):
	words = []

	reader = open(file, 'r')
	for line in reader:
		for word in line.strip().split(' '):
			words.append(word)

	reader.close()

	pipeline = Pipeline([("vect", CountVectorizer(min_df=1, stop_words="english")), ("tfidf", TfidfTransformer(use_idf=False))])
	return pipeline.fit_transform(words)


def getSimilarity(ref, candidate):
	return cosine_distance(ref, candidate)


def generateMatrix(simDoc, fileList):



def main():
	'''
	restaurantFile = basePath + 'yelp_academic_dataset_business.json'
	reviewFile = basePath + 'yelp_academic_dataset_review.json'
	cuisineCategoryMap = {}

	idCategoryMap = restaurantIDCategoryMapping(restaurantFile)

	reviewRestaurantIDMapping(idCategoryMap, reviewFile, cuisineCategoryMap)
	'''
	simDoc, fileList = findSimilarity(basePath + 'output')

	generateMatrix(simDoc, fileList)

if __name__ == '__main__':
	main()