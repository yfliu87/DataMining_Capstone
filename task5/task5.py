'''
{restaurants: {target_dish: [occurrence, avg.star]}}
'''

business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'

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