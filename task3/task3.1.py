review_file_path = ''
business_file_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'
manual_annotation_task_path = '/home/yfliu/DataMining_Workspace/DataMining/CapstoneProject/yelp_dataset_challenge_academic_dataset/task3/manualAnnotationTask'

'''
1. build business_type, business_id map
2. traverse review file:
	traverse business id:
		collect all reviews belong to business_id

3. traverse tip file:
	traverse business id:
		collect all tips belong to business_id

'''

def get_target_business_type():
	import os
	result = []

	for (dirpath, dirnames, filenames) in os.walk(manual_annotation_task_path):
		for filename in filenames:
			result.append(filename.split('.')[0])

	return result


def build_business_type_id_map(business_type_list):
	import json
	result = {}

	reader = open(business_file_path, 'r')
	line = reader.readline()

	while line:
		json_line = json.loads(line)
		categories = json_line['categories']

		for cat in categories:
			if cat in business_type_list:
				business_id = json_line['business_id']

				if cat not in result:
					result[cat] = set() 

				result[cat].add(business_id)

		line = reader.readline()

	reader.close()

	return result

if __name__ == '__main__':
	target_business_type = get_target_business_type()
	business_type_id_map = build_business_type_id_map(target_business_type)

