business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'

def build_business_type_restaurant_id_map(target_type):
	import json

	result = {}
	result[target_type] = set() 

	reader = open(business_file_path, 'r')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		categories = json_line['categories']

		if target_type in categories:
			business_id = json_line['business_id']
			result[target_type].add(business_id)

		line = reader.readline()

	reader.close()
	return result


if __name__ == '__main__':
	build_business_type_restaurant_id_map("Chinese")