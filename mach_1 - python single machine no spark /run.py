from nltk.corpus import treebank as wsj
from memm import MEMM

import cProfile
import pstats
import random
import StringIO

def create_dataset():
	print 'Loading dataset'
	dataset = []
	tags = []
	sents = wsj.sents()

	for i,sentence in enumerate(wsj.tagged_sents()[:10]):
		prev = None
		prev_prev = None
		for j,word in enumerate(sentence):
			datapoint = {}
			datapoint['wn'] = sents[i]
			
			datapoint['index'] = j
			if(prev == None):
				datapoint['t_minus_one'] = '*'
			else:
				datapoint['t_minus_one'] = prev[1]
			if(prev_prev == None):
				datapoint['t_minus_two'] = '*'
			else:
				datapoint['t_minus_two'] = prev_prev[1]

			prev_prev = prev
			prev = word
			dataset.append(datapoint)
			tags.append(word[1])
	print 'Done'
	return dataset, tags

def f1(x,y):
	return round(random.random())

def f2(x,y):
	return round(random.random())

def f3(x,y):
	return round(random.random())

def f4(x,y):
	return round(random.random())

def f5(x,y):
	return round(random.random())

def f6(x,y):
	return round(random.random())

def f7(x,y):
	return round(random.random())

def f8(x,y):
	return round(random.random())

def f9(x,y):
	return round(random.random())

def f10(x,y):
	return round(random.random())



if __name__ == '__main__':
	data,tag = create_dataset()
	param = [0 for i in range(10)]
	print 'Profiling started'
	prof = cProfile.Profile()
	prof.enable()
	memm = MEMM(data, tag, param, [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10], 0)
	memm.train()
	prof.disable()
	s = StringIO.StringIO()
	sortby = 'cumulative'
	ps = pstats.Stats(prof, stream=s).sort_stats(sortby)
	ps.print_stats()
	print s.getvalue()
