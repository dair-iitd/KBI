import joblib
import numpy as np
import sys
from collections import Counter,defaultdict as ddict
import matplotlib.pyplot as plt



def create_plot(data, ax, bins, titleString):
	'''
		Plot the histogram for 'data'.
	'''
	ax.hist(data, bins)
	ax.set_title(titleString)
	#ax.savefig(fileName)


def get_frequency_table(train_data_entities, data):
	'''
		-train_data_entities = np array of (e1,e2) [training data]
		- returns the 
	'''
	entity_frequencies = ddict(int)
	entity_pair_frequencies = ddict(int)
	for (e1,e2) in train_data_entities:
		if (e1 != e2):
			entity_frequencies[e1]+=1
			entity_frequencies[e2]+=1
		else:
			entity_frequencies[e1]+=1

		entity_pair_frequencies[(e1,e2)]+=1

	freq_entities = [entity_frequencies[key] for key in entity_frequencies]
	freq_entity_pairs = [ entity_pair_frequencies[key] for key in entity_pair_frequencies]

	fig = plt.figure()
	ax1 = fig.add_subplot(2, 1, 1)
	ax2 = fig.add_subplot(2, 1, 2)

	create_plot(freq_entities, ax1, np.arange(1,50), "Histogram of entity frequencies for %s" %data)
	create_plot(freq_entity_pairs, ax2, np.arange(1,50), "Histogram of entity pair frequencies for %s" %data)
	plt.savefig("%s_distribution.png" %data)
	plt.close()

	# how many entities occured x number of times
	entity_freq_counter = Counter(freq_entities) 
	# how many entity pairs occured x number of times
	entity_pair_freq_counter = Counter(freq_entity_pairs)

	return entity_freq_counter, entity_pair_freq_counter

def get_oov_details(entities_in_train, entity_pairs_in_train, test_data):
	entity_oovs=0.0
	entity_pair_oovs=0.0

	for (e1,r,e2) in test_data:
		if (e1,e2) not in entity_pairs_in_train:
			entity_pair_oovs+=1

		if (e1 not in entities_in_train) or (e2 not in entities_in_train):
			entity_oovs+=1

	print("entity pair OOV rate: %5.4f" %(100.0*entity_pair_oovs/len(test_data)))
	print("entity OOV rate: %5.4f" %(100.0*entity_oovs/len(test_data)))


if __name__ == '__main__':
	data_path = sys.argv[1]
	dataset = data_path.split('/')[-1]

	train_data_entities = joblib.load("%s/train_entities.joblib" %data_path)
	test_data = joblib.load("%s/test.joblib" %data_path)

	entities_in_train = set([e for (e1,e2) in train_data_entities for e in (e1,e2)])
	entity_pairs_in_train = set([(e1,e2) for (e1,e2) in train_data_entities])

	entity_freq_counter, entity_pair_freq_counter = get_frequency_table(train_data_entities, dataset)

	print('='*50)
	get_oov_details(entities_in_train, entity_pairs_in_train, test_data)
	print('total entities in train_data: %d' %len(entities_in_train))
	print('total entity pairs in train_data: %d' %len(entity_pairs_in_train))
	print('total facts in train_data: %d' %train_data_entities.shape[0])
	print('entity singletons: %d' %(entity_freq_counter[1]))
	print('entity pair singletons: %d' %(entity_pair_freq_counter[1]))
	print('='*50)
