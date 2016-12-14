import featureExtraction
import glob
import numpy
import scipy.io.wavfile
from sklearn.preprocessing import OneHotEncoder
from keras.regularizers import l2, activity_l2
import LPC
import utils
import Removesilence as rs
import os


def train(self, directory):
		train_data = self.featuresObj.load_data(directory)
		X_train = train_data[0]
		y_train = train_data[1]
		print X_train.shape

		X_train = self.Pca(X_train)
		y_train = self.encodeY(y_train)
		print y_train
		print X_train.shape
		print y_train.shape
		print 
		self.model.add(Dense(64, input_dim=X_train.shape[1] , init=activation_fn))
		self.model.add(Activation('tanh'))
		self.model.add(Dropout(0.5))

		self.model.add(Dense(self.featuresObj.num_speakers, init=activation_fn))
		self.model.add(Activation('softmax'))

		sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
		self.model.compile(loss='categorical_crossentropy',
					  optimizer=sgd,
					  metrics=['accuracy'])

		self.model.fit(X_train, y_train, nb_epoch=epochs, validation_split= 0.2, batch_size=32)
		self.mapping = self.featuresObj.mapping

	
	def test(self, testdirec, model, pca):
		# pca = self.train()
		num_speakers = filecount(testdirec)
		testdirec = testdirec + "/"
		files = glob.glob(testdirec + "*.txt")
		tot_positives = 0
		speaker_no = 0

		while (speaker_no < num_speakers):
			speaker_no += 1
			filename = testdirec + str(speaker_no) + ".txt"
			print filename
			print

			flag = self.testFile(filename, model, pca, speaker_no-1)
			if flag:
				tot_positives +=1

		print "total number of correct answers = " + str(tot_positives)
		print 
		print
		return tot_positives


	def testFile(self, filename, model, pca, phone_number, mapping):
		test_data = self.featuresObj.getFeaturesFromFile(filename)  ### frames by features(34)
		X_test = test_data
		X_test = pca.transform(X_test)
		modelNN = model.predict(X_test)

		sumRows = numpy.sum(modelNN, axis=0)
		sumRows /= modelNN.shape[0]
		print sumRows
		print
		index = numpy.argmax(sumRows)
		print sumRows[index]
		print mapping
		user_id = mapping[str(index+1)]
		print user_id
		print phone_number
		try:
			if int(phone_number)==int(user_id):
				if sumRows[index]<0.4:
					print ' true but less than 0.4' 
					print
				return True
			else:
				return False	
		except Exception, e:
			raise e
		


### counts number of directories in the given directory

def fcount(path):
	count = 0
	for f in os.listdir(path):
		child = os.path.join(path, f)
		if os.path.isdir(child):
			count +=1
	return count

def filecount(path):
	count = 0
	for f in os.listdir(path):
		child = os.path.join(path, f)
		if os.path.isfile(child):
			count +=1
	return count





class Features(object):
	"""docstring for Features"""
	num_speakers = 0
	mapping = {}
	def __init__(self, frame_size, frame_shift):
		self.frame_size = float(frame_size)
		self.frame_shift = float(frame_shift)
		# self.direc = direc
		# num_speakers = int(num_speakers)
	
	def getTrainingMatrix(self, direc):
		srno = 0
		flag = False
		fno =0
		for user_directory in os.listdir(direc):
			print
			print "username = " + str(user_directory)
			phone_number = user_directory.split("-")
			phone_number = phone_number[0]
			phone_number = int(phone_number)
			print phone_number
			print
			user_directory_path = os.path.join(direc, user_directory)

			if os.path.isdir(user_directory_path):
				Features.num_speakers += 1
				Features.mapping[Features.num_speakers] = phone_number
				srno += 1
				for file in os.listdir(user_directory_path):
					print "\nfile_name = " + str(file)
					fname = os.path.join(user_directory_path, file)
					if os.path.isfile(fname):
						fn = fname.split('/')
						fn = fn[-1]
						if fn[-4:]=='.txt':
							featuresT = self.getFeaturesFromFile(fname)
							if flag==False:
								c = featuresT
								flag = True
								y = numpy.ones(shape=(featuresT.shape[0],))
								y.fill(Features.num_speakers)
							else:
								c = numpy.concatenate((c, featuresT), axis = 0)
								y1 = numpy.ones(shape=(featuresT.shape[0],))
								y1.fill(Features.num_speakers)
								y = numpy.concatenate((y,y1), axis = 0)
						else:
							print "file is not an txt file"

		return (c, y)


	def getFeaturesFromFile(self, fname):
		fs, features = file.read(fname)
		stfeatures = featureExtraction.stFeatureExtraction(features, fs)
		featuresT = stfeatures.transpose()
		featuresT = numpy.concatenate((featuresT, lpc), axis = 1)
		return featuresT


	def load_data(self, directory):
		X, Y = self.getTrainingMatrix(directory)
		indices  = numpy.random.permutation(Y.shape[0])
		X = X[indices, :]
		Y = Y[indices]
		train_data_rows = int(Y.shape[0])

		train_x = X[0:train_data_rows+1, :]
		train_y = Y[0:train_data_rows+1]
		train_data = (train_x, train_y)
		return train_data





if __name__ == "__main__":
	num_args = len(sys.argv)

	if num_args < 3:
		""" directory name as well as test file name with it
				"""
		sys.exit(0)

	elif num_args >= 3:
		if sys.argv[1] == 'train':
			# direc = raw_input('Please Enter Input directory for training: ')
			direc = sys.argv[2]
			print direc
			print type(direc)
			speakers = int(fcount(direc))

			t = Train(frame_size=0.032, frame_shift=0.016)
			t.train(direc, 20)
			json_string = t.model.to_json()
			open('my_model_architecture.json', 'w').write(json_string)
			t.model.save_weights('my_model_weights.h5', overwrite = True)
			
			with open('my_dumped_pca.pkl', 'wb') as fid:
				cPickle.dump(t.pca, fid)

			fid.close()
			with open('dictionary.json', 'w') as f:
				json.dump(t.mapping, f)

			f.close()
			print "model is saved Finally"

		elif sys.argv[1] == 'test' :
			# testdirec = raw_input('Please enter Input directory or file for testing: ')
			testdirec = sys.argv[3]
			if os.path.isdir(testdirec):
				flag = True
			else:
				flag = False
			
			t = Train(frame_size=0.032, frame_shift=0.016)

			model = model_from_json(open('my_model_architecture.json').read())
			model.load_weights('my_model_weights.h5')

			with open('my_dumped_pca.pkl', 'rb') as fid:
				pca_loaded = cPickle.load(fid)

			fid.close()
			with open('dictionary.json', 'r') as f:
				try:
					mapping = json.load(f)
				except ValueError:
					mapping = {}

			print mapping
			f.close()

			sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
			model.compile(loss='categorical_crossentropy',
						  optimizer=sgd,
						  metrics=['accuracy'])
			print "model is read "
			if flag:
				tot_positives = t.test(testdirec, model, pca_loaded)
				print tot_positives
			else:
				# phone_number = raw_input('Please enter phone number for current user:')
				phone_number = sys.argv[2]
				true = t.testFile(testdirec, model, pca_loaded, phone_number, mapping)
				print true
				print
		else:
			print 'wrong argument entered: Please enter "train" or "test"'
	else:
		print "enter one argument whether to train or test"

	sys.exit(0)
