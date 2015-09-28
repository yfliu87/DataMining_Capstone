'''
{restaurants: {target_dish: [occurrence, avg.star]}}
'''
import codecs

business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'

def build_business_type_restaurant_id_map(target_type):
	import json

	result = {} 

	reader = codecs.open(business_file_path, 'r', 'utf-8')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		categories = json_line['categories']

		if target_type in categories:
			business_id = json_line['business_id']

			if business_id not in result:
				restaurant_name = json_line['name']
				address = json_line['full_address']
				result[business_id] = (restaurant_name, address.replace('\n',' '))

		line = reader.readline()

	reader.close()
	return result

'''
return map: {business_id: {star: [(date, review)]}}
'''
def build_restaurant_id_review_map(chinese_restaurants):
	import json

	target_map = {}
	reader = codecs.open(review_file, 'r', 'utf-8')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		bus_id = json_line['business_id'] 

		if bus_id not in chinese_restaurants:
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
	restaurant_reviews = build_restaurant_id_review_map(chinese_restaurants)
