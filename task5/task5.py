'''
{restaurants: {target_dish: [occurrence, avg.star]}}
'''
import codecs

business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
chinese_dish_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/student_dn_annotations.txt'
output_cuisine_occurrence_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task5/output/dumpling_restaurants_occurrence.txt'
output_cuisine_avgStar_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task5/output/dumpling_restaurants_avgStar.txt'
output_cuisine_address_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task5/output/dumpling_restaurants_address.txt'
'''
map: {business_id: (restaurant_name, address)}
'''
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

				if ',' in address:
					state = address.split(', ')[1]
					street = address.split(', ')[0].replace('\n', ' ')
				else:
					state = address.split('\n')[-1]
					street = ' '.join(address.split('\n')[0:-1])

				result[business_id] = (restaurant_name, state + ',' + street)

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


def read_chinese_dish_from_file(dish_file):
	dishes = set() 

	reader = codecs.open(dish_file, 'r', 'utf-8')
	line = reader.readline()

	while line:
		if line != '':
			dishes.add(line.replace('\n', ''))

		line = reader.readline()

	reader.close()

	return list(dishes)


def build_dish_star_map(dishes, restaurant_reviews):
	dish_star_map = {}

	for dish in dishes:
		dish_star_map[dish] = {}

	for dish in dishes:
		for bus_id in restaurant_reviews:
			stars = restaurant_reviews[bus_id]

			for star in stars:
				reviews = restaurant_reviews[bus_id][star]

				for review in reviews:
					date = review[0]
					content = review[1]

					if dish in content:
						if star not in dish_star_map[dish]:
							dish_star_map[dish][star] = []

						dish_star_map[dish][star].append((date))

	return dish_star_map


def get_specific_dish_occurrence_avgStar(target_cuisine, restaurant_reviews):
	business_id_occurrence_star_map = {}

	for business_id in restaurant_reviews.keys():
		stars = []
		for star in restaurant_reviews[business_id]:
			reviews = restaurant_reviews[business_id][star]

			for rev in reviews:
				if target_cuisine in rev[1]:
					stars.append(star)

		if len(stars) == 0:
			continue

		business_id_occurrence_star_map[business_id] = (len(stars), float('%.2f'%(float(sum(stars))/len(stars))))

	return business_id_occurrence_star_map

def get_restaurant_info(id_occurrence_star_map, chinese_restaurants):
	restaurants = {} 

	for business_id in id_occurrence_star_map:
		restaurants[business_id] = (chinese_restaurants[business_id], id_occurrence_star_map[business_id])

	return restaurants


def output_restaurant_info_by_occurrence(dumpling_restaurants):
	import collections

	sorted_map = collections.OrderedDict(sorted(dumpling_restaurants.items(), key=lambda k:k[1][1][0], reverse=True)) 

	writer = open(output_cuisine_occurrence_file, 'a')
	writer.write("restaurant" + '\t' + 'address' + '\t' + 'review occur' + '\t' + 'avg Star' + '\n')

	for restaurant in sorted_map.values():
		name = restaurant[0][0]
		address = restaurant[0][1]
		occurrence = restaurant[1][0]
		avgStar = restaurant[1][1]

		message = name + '\t' + address + '\t' + str(occurrence) + '\t' + str(avgStar) + '\n'

		writer.write(message)

	writer.close()


def output_restaurant_info_by_avgStar(dumpling_restaurants):
	import collections

	sorted_map = collections.OrderedDict(sorted(dumpling_restaurants.items(), key=lambda k:k[1][1][1], reverse=True)) 

	writer = open(output_cuisine_avgStar_file, 'a')
	writer.write("restaurant" + '\t' + 'address' + '\t' + 'review occur' + '\t' + 'avg Star' + '\n')

	for restaurant in sorted_map.values():
		name = restaurant[0][0]
		address = restaurant[0][1]
		occurrence = restaurant[1][0]
		avgStar = restaurant[1][1]

		message = name + '\t' + address + '\t' + str(occurrence) + '\t' + str(avgStar) + '\n'

		writer.write(message)

	writer.close()


def output_restaurant_info_by_address(dumpling_restaurants):
	import collections

	sorted_map = collections.OrderedDict(sorted(dumpling_restaurants.items(), key=lambda k:k[1][0][1], reverse=False)) 

	writer = open(output_cuisine_address_file, 'a')
	writer.write('address' + '\t' + 'restaurant' + '\t' + 'review occur' + '\t' + 'avg Star' + '\n')

	for restaurant in sorted_map.values():
		name = restaurant[0][0]
		address = restaurant[0][1]
		occurrence = restaurant[1][0]
		avgStar = restaurant[1][1]

		message = address + '\t' + name + '\t' + str(occurrence) + '\t' + str(avgStar) + '\n'

		writer.write(message)

	writer.close()


if __name__ == '__main__':
	chinese_restaurants = build_business_type_restaurant_id_map("Chinese")
	restaurant_reviews = build_restaurant_id_review_map(chinese_restaurants)
	#dishes = read_chinese_dish_from_file(chinese_dish_file)
	#dish_star_map = build_dish_star_map(dishes, restaurant_reviews)

	id_occurrence_star_map = get_specific_dish_occurrence_avgStar("dumpling", restaurant_reviews)
	dumpling_restaurants = get_restaurant_info(id_occurrence_star_map, chinese_restaurants)
	output_restaurant_info_by_occurrence(dumpling_restaurants)
	output_restaurant_info_by_avgStar(dumpling_restaurants)
	output_restaurant_info_by_address(dumpling_restaurants)
