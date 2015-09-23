business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
chinese_dish_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/student_dn_annotations.txt'
review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_10000_review.json'

def build_business_type_restaurant_id_map(target_type):
	import json

	result = {}
	result[target_type] = [] 

	reader = open(business_file_path, 'r')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		categories = json_line['categories']

		if target_type in categories:
			business_id = json_line['business_id']

			if business_id not in result[target_type]:
				result[target_type].append(business_id)

		line = reader.readline()

	reader.close()
	return result


def read_chinese_dish_from_file(dish_file):
	dishes = set() 

	reader = open(dish_file, 'r')
	line = reader.readline()

	while line:
		if line != '':
			dishes.add(line.replace('\n', ''))

		line = reader.readline()

	reader.close()

	return list(dishes)

def build_restaurant_id_review_map(chinese_restaurants):
	import json

	target_map = {}
	reader = open(review_file, 'r')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		bus_id = json_line['business_id'] 

		if bus_id not in chinese_restaurants['Chinese']:
			line = reader.readline()
			continue

		reviews = json_line['text']
		stars = json_line['stars']
		date = json_line['date']

		if bus_id not in target_map:
			target_map[bus_id] = {} 

		if stars not in target_map[bus_id]:
			target_map[bus_id][stars] = []

		target_map[bus_id][stars].append((date, reviews))

		line = reader.readline()

	reader.close()
	return target_map


if __name__ == '__main__':
	chinese_restaurants = build_business_type_restaurant_id_map("Chinese")
	dishes = read_chinese_dish_from_file(chinese_dish_file)
	restaurant_reviews = build_restaurant_id_review_map(chinese_restaurants)
