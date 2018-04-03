import csv
import numpy as np
import tensorflow as tf

class Dataset:
	__LABEL_INDEX = 4
	TRAIN_EXAMPLES = 105
	TEST_EXAMPLES = 45
	BATCH_SIZE = 15
	FEATURES = 4
	CLASSES = 3

	def __init__(self):
		data = self.__read_file__()
		self.__set_np_examples_and_lables__(data)
		self.__normalize_data__()
		self.set_train_test_sets()
		self.__reset_last_index__()

	def __read_file__(self):
		data = []
		with open('iris.csv') as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				data.append(row)
		return data

	def __set_np_examples_and_lables__(self, data):
		exmpls = []
		lbls = []
		for row in data:
			exmpls.append(row[:self.__LABEL_INDEX])
			if row[self.__LABEL_INDEX] == "Iris-setosa":
				lbls.append(0)
			elif row[self.__LABEL_INDEX] == "Iris-versicolor":
				lbls.append(1)
			else:
				lbls.append(2)
		self.examples = np.array(exmpls, dtype=np.float32)
		self.labels = np.array(lbls, dtype=np.int32)

	def __get_min_max__(self, feature):
		ex_transp = self.examples.transpose()
		min, max = ex_transp[feature][0], ex_transp[feature][0]
		for i in range(len(ex_transp[feature])):
			if min > ex_transp[feature][i]:
				min = ex_transp[feature][i]
			if max < ex_transp[feature][i]:
				max = ex_transp[feature][i]
		return min, max

	def __normalize_data__(self):
		for i in range(self.FEATURES):
			min, max = self.__get_min_max__(i)
			for j in range(len(self.examples)):
				self.examples[j][i] = (self.examples[j][i] - min)/(max - min)

	def __shuffle_data__(self):
		p = np.random.permutation(len(self.examples))
		self.examples = self.examples[p]
		self.labels = self.labels[p]

	def set_train_test_sets(self):
		self.__shuffle_data__()
		self.train_examples = self.examples[0:self.TRAIN_EXAMPLES]
		self.train_labels = self.labels[0:self.TRAIN_EXAMPLES]
		self.test_examples = self.examples[self.TRAIN_EXAMPLES:]
		self.test_labels = self.labels[self.TRAIN_EXAMPLES:]

	def __reset_last_index__(self):
		self.__last_index = 0

	def get_train_set_length(self):
		return len(self.train_examples)

	def next_batch(self):
		if len(self.train_examples) - self.__last_index < self.BATCH_SIZE:
			aux = self.__last_index
			self.__reset_last_index__()
			return self.train_examples[aux:], self.train_labels[aux:]
		else:
			aux = self.__last_index
			self.__last_index = self.__last_index + self.BATCH_SIZE
			return self.train_examples[aux:self.__last_index], self.train_labels[aux:self.__last_index]


class Classifier:
	WEIGHTS = 0
	BIASES = 1

	def __init__(self, learning_rate=0.5, layers_num=1, 
		layers_shapes=[Dataset.FEATURES, Dataset.CLASSES]):
		self.learning_rate = learning_rate
		self.layers_num = layers_num
		self.layers_shapes = layers_shapes
		self.x = tf.placeholder(tf.float32, shape=[None, Dataset.FEATURES])
		self.y = tf.placeholder(tf.int32, shape=[None])
		self.__build_model__()

	def __build_layer__(self, layer_shapes):
		W = tf.Variable(tf.random_uniform([layer_shapes[0], layer_shapes[1]], dtype=tf.float32), 
			dtype=tf.float32)
		b = tf.Variable(tf.zeros([layer_shapes[1]], dtype=tf.float32), dtype=tf.float32)
		return [W, b]

	def __build_layers__(self):
		self.variables = []
		for i in range(self.layers_num):
			self.variables.append(self.__build_layer__(
				[self.layers_shapes[2*i], self.layers_shapes[2*i+1]]))

	def __build_net__(self):
		self.__build_layers__()
		self.ops = []
		for layer in range(self.layers_num):
			if layer == 0:
				self.ops.append((
					tf.matmul(self.x, self.variables[layer][self.WEIGHTS]) + 
					self.variables[layer][self.BIASES]))
			else:
				self.ops.append(tf.nn.sigmoid(
					tf.matmul(self.ops[layer-1], self.variables[layer][self.WEIGHTS]) +
					self.variables[layer][self.BIASES]))

	def __build_model__(self):
		self.__build_net__()
		self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, 
			logits=self.ops[self.layers_num-1]))
		self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


if __name__ == "__main__":
	EPOCHS = 100000

	dataset = Dataset()

	g = tf.Graph()
	with g.as_default():
		classifier = Classifier()

	with tf.Session(graph=g) as sess:
		sess.run(tf.global_variables_initializer())
		print("train acc:", sess.run(classifier.loss, feed_dict={
			classifier.x: dataset.train_examples,
			classifier.y: dataset.train_labels
			}))
		for e in range(EPOCHS):
			for _ in range(dataset.get_train_set_length()//dataset.BATCH_SIZE):
				examples_batch, labels_batch = dataset.next_batch()
				sess.run(classifier.train_op, feed_dict={
					classifier.x: examples_batch,
					classifier.y: labels_batch
					})
			if e == 0 or e == 9999 or e % 1000 == 0:
				print("epoch:", e, "train acc:", sess.run(classifier.loss, feed_dict={
					classifier.x: dataset.train_examples,
					classifier.y: dataset.train_labels
					}))
		print("test acc:", sess.run(classifier.loss, feed_dict={
			classifier.x: dataset.test_examples,
			classifier.y: dataset.test_labels
			}))
