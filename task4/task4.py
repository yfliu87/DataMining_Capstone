import codecs

business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
chinese_dish_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/student_dn_annotations.txt'
review_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
dish_statistic_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/dish_statistics.txt'
graph_data_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/dumpling_double_axis_data.tsv'
all_dish_occurance_file = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task4/all_dish_occurance_data.tsv'

def build_business_type_restaurant_id_map(target_type):
	import json

	result = {}
	result[target_type] = [] 

	reader = codecs.open(business_file_path, 'r', 'utf-8')
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

	reader = codecs.open(dish_file, 'r', 'utf-8')
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
	reader = codecs.open(review_file, 'r', 'utf-8')
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


def calculate(dish_star_map):
	dish_statistics = {}

	for dish in dish_star_map.keys():
		stars = dish_star_map[dish]

		average_star = 0.0
		occurance = 0
		total_star = 0

		for star in stars:
			star_num = int(star)
			current_occurance = len(dish_star_map[dish][star])
			total_star += star_num * current_occurance
			occurance += current_occurance

		if occurance == 0:
			continue

		average_star = float(total_star/occurance)
		dish_statistics[dish] = (occurance, average_star)


	return dish_statistics


def output_to_disk(dish_statistic_file, dish_statistics):
	writer = codecs.open(dish_statistic_file, 'a', 'utf-8')

	for dish in dish_statistics:
		occurance = dish_statistics[dish][0]
		average_star = dish_statistics[dish][1]

		writer.write(dish + '\t' + str(occurance) + '\t' + str(average_star) + '\n')

	writer.close()


def arrange_data_for_double_axis(dish_star_map):
	date_star_occurance_map = build_date_star_map(dish_star_map)
	date_star_average_occurance_map = calculate_average(date_star_occurance_map)


def calculate_average(date_star_occurance_map):
	import collections

	result = [] 
	sorted_map = collections.OrderedDict(sorted(date_star_occurance_map.items())) 

	for date in sorted_map.keys():
		occurance = len(sorted_map[date])
		total_star = sum(sorted_map[date])
		average_star = float(total_star/occurance)

		result.append(date + '\t' + str(occurance) + '\t' + str(average_star))

	writer = open(graph_data_file, 'a')
	writer.write('\t'.join(['date', 'occurance', 'average_star']))
	writer.write('\n')

	for item in result:
		writer.write(item + '\n')

	writer.close()


def build_date_star_map(dish_star_map):
	star_review_detail = dish_star_map['dumplings']

	date_star_occurance_map = {}

	for star in star_review_detail:
		for date in star_review_detail[star]:
			year_month = date.split('-')[0] + '-' + date.split('-')[1]

			if year_month not in date_star_occurance_map:
				date_star_occurance_map[year_month] = []

			date_star_occurance_map[year_month].append(star)

	return date_star_occurance_map


def arrange_data_for_all_dish_occurance(dish_star_map):
	import collections

	dish_occurance_map = {}

	for dish in dish_star_map.keys():
		occurance = 0

		for star in dish_star_map[dish]:
			occurance += len(dish_star_map[dish][star])

		dish_occurance_map[dish] = occurance 

	sorted_map = collections.OrderedDict(sorted(dish_occurance_map.items(), key=lambda d:d[1], reverse=True))

	writer = open(all_dish_occurance_file, 'a')
	writer.write('dish' + '\t' + 'occurance' + '\n')

	for dish, occurance in sorted_map.items():
		writer.write(dish.encode('utf-8') + '\t' + str(occurance) + '\n')

	writer.close()



if __name__ == '__main__':
	chinese_restaurants = build_business_type_restaurant_id_map("Chinese")
	dishes = read_chinese_dish_from_file(chinese_dish_file)
	restaurant_reviews = build_restaurant_id_review_map(chinese_restaurants)
	dish_star_map = build_dish_star_map(dishes, restaurant_reviews)
	#dish_statistics = calculate(dish_star_map)
	#output_to_disk(dish_statistic_file, dish_statistics)

	arrange_data_for_double_axis(dish_star_map)
	arrange_data_for_all_dish_occurance(dish_star_map)