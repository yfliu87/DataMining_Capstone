import json

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


def main():
	restaurantFile = basePath + 'yelp_academic_dataset_business.json'
	reviewFile = basePath + 'yelp_academic_dataset_review.json'
	cuisineCategoryMap = {}

	idCategoryMap = restaurantIDCategoryMapping(restaurantFile)
	reviewRestaurantIDMapping(idCategoryMap, reviewFile, cuisineCategoryMap)

	'''
	findSimilarity(cuisineCategoryMap)
	generate_graph()
	'''

if __name__ == '__main__':
	main()