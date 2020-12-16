import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from imutils import face_utils
font = cv2.FONT_HERSHEY_SIMPLEX

import os


from keras.applications import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten, Subtract, Input, BatchNormalization, Conv2D, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.utils import plot_model

from sklearn.preprocessing import OneHotEncoder


# Getting the Files in the folder provided.
def get_files(path):
	return os.listdir(path)

''' 
path to the opencv haar cascade weights.


Change this to the haar cascade path on yout system.

'''
cascPath = "/Users/abdulrehman/opt/anaconda3/envs/Face-Detection/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml"


'''

The return bbx function takes input an image and applies haarcascade to find faces and returns the bounding boxes of the faces in the image.

'''
def return_bbx(image):
	faceCascade = cv2.CascadeClassifier(cascPath)
	faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
	return faces

'''

The path to the dataset with all the folders of each individual

Dataset structure 

DATASET FOLDER
	|
	|--- Subject 1
	|		| Subject 1 image 1
	|		|...
	|		| Subject 1 image m
	|
	|--- Subject 2
	|		| Subject 2 image 1
	|		|...
	|		| Subject 2 image m
	.
	.
	.
	|
	|--- Subject n
	|		| Subject n image 1
	|		|...
	|		| Subject n image m


'''

Dataset_path = '/Users/abdulrehman/Desktop/SML Project/FacesInTheWild/'

# read the csv with all the subject's names. We have used the Faces in the wild dataset which uses faces of celebrities
Celebs = pd.read_csv(Dataset_path+'lfw_allnames.csv')
Celebs = Celebs[Celebs['images']>10]

'''
cat_list -> Stores all the categories
X -> All the images
Y -> the name of the subject in the corresponding image in X
'''

print('Importing the data')
cat_list = []
X = []
Y = []
y_label = 0

for _, [name,__] in Celebs.iterrows():
	celeb_path = Dataset_path+'lfw-deepfunneled/'+name+'/'
	
	images_paths = get_files(celeb_path)
	temp = []
	for image_path in images_paths:
		image = cv2.imread(celeb_path+image_path,1)
		faces = return_bbx(image)
		if len(faces) == 1:
			if len(temp)>=10:
				break
			temp.append(len(X))
			(x,y,w,h) = faces[0]
			cropped = image[x:x+w, y:y+h]
			dim = (100, 100)
			resized = cv2.resize(cropped, dim, interpolation = cv2.INTER_AREA)
			image = np.array(resized).astype("float32")
			X.append(image)
			Y.append(y_label)
	y_label+=1
	cat_list.append(temp)

X_data = np.asarray(X)/255
Y_data = np.array(Y)
cat_list = np.asarray(cat_list)

print('Done Importing')

# print(X_data.shape, Y_data.shape, cat_list.shape)


print('Splitting the data')

a = Y_data
n_classes = len(set(a))
n_classes

train_split = 0.9

train_size = int(n_classes*train_split)
test_size = n_classes-train_size

train_files = train_size * 10

X_train = X_data[:train_files]
y_train = Y_data[:train_files]
cat_train = cat_list[:train_size]

#Validation Split
X_test = X_data[train_files:]
y_test = Y_data[train_files:]
cat_test = cat_list[train_size:]

# print('X&Y shape of training data :',X_train.shape, 'and', y_train.shape, cat_train.shape)
# print('X&Y shape of testing data :' , X_test.shape, 'and', y_test.shape, cat_test.shape)




'''
In this function we generate batches for training and testing the data.
This function imports data and creates data with 2 inputs for X and one output for each pair of outputs.

get_batch takes data, categories, datasize, and batch_size as input,
creates a numpy array Y with half the values 0 and half 1 and shuffles

for every output that is 1 the input X contains images of two different categories
for every outpyt that is 0 the input X contains images of the same categories
'''

def get_batch(data_x, data_cat, data_size, batch_size=64):
	
	#initializing the data for temporary use
	temp_x = data_x
	temp_cat_list = data_cat

	start=0
	end=data_size
	batch_x=[]

	# Initializing the Y output of size Batch_size, setting half the values to 0 and then shuffuling Y
	batch_y = np.zeros(batch_size)
	batch_y[int(batch_size/2):] = 1
	np.random.shuffle(batch_y)
	
	# Class list is a list of random Categories.
	# Batch X is a list of 2 numpy arrays of shape (batch_size, 100, 100, 3)
	class_list = np.random.randint(start, end, batch_size) 
	batch_x.append(np.zeros((batch_size, 100, 100, 3)))
	batch_x.append(np.zeros((batch_size, 100, 100, 3)))

	# Traversing through all the X and Y.
	# Assigning same different images of the same subject to X if Y = 0 and different images of different subjects is Y = 1
	for i in range(0, batch_size):

		'''
		First assign a random subjects image. as X[0] for ith position of the batch
		Class list is a list of random Categories, of size batch_size.
		- class_list[i] gets the random subject.
		- temp_cat_list[class_list[i]] gets the list of all the image positions from that randomply chosen category
		- np.random.choice on this list choses a ranom image from the list of all the image positions from the randomply chosen category
		- this randomly chosen image of a randomly chosen sunject is assigned to X[0] for the ith position of the batch.
		'''

		batch_x[0][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]  
		'''
		Now the y value is checked.
		if the Y valus is 0 we assign a random image of the same categoruy to X[1]
		if the Y value is 1 we assign a random image of a random category to X[1]
		'''
		if batch_y[i]==0:
			batch_x[1][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]

		else:
			temp_list = np.append(temp_cat_list[:class_list[i]], temp_cat_list[class_list[i]+1:])
			temp_list = np.random.choice(temp_list)
			batch_x[1][i] = temp_x[np.random.choice(temp_list)]            
			
	return(batch_x, batch_y)


'''
In this function we create a siamese network with 2 vgg16 networks working in parallel.
we use a convolutional model to find feature maps of images.
each convolutional neural netwirk takes an inage and outputs the feature maps of these images
Theses image feature maps are then compared using a subtraction layer.
The subtraction layer then finds the distance between the feature maps of the images
this disctance is then taken as input to the a Fully connected layer which predicts if the images belong to the same subject.
In this function we create the model.
'''


def get_model(input_shape):

	left_input = Input(input_shape)
	right_input = Input(input_shape)

	left = Sequential()
	left.add(left_input)
	left.add(Conv2D(64, (3,3), activation='relu'))
	left.add(MaxPooling2D(2,2))
	left.add(Conv2D(128, (3,3), activation='relu'))
	left.add(MaxPooling2D(2,2))
	left.add(Conv2D(128, (3,3), activation='relu'))
	left.add(MaxPooling2D(2,2))
	left.add(Conv2D(256, (3,3), activation='relu'))
	left.add(MaxPooling2D(2,2))
	left.add(Flatten())
	left.add(Dense(1028, activation='relu', kernel_regularizer=l2(1e-2)))

	right = Sequential()
	right.add(right_input)
	right.add(Conv2D(64, (3,3), activation='relu'))
	right.add(MaxPooling2D(2,2))
	right.add(Conv2D(128, (3,3), activation='relu'))
	right.add(MaxPooling2D(2,2))
	right.add(Conv2D(128, (3,3), activation='relu'))
	right.add(MaxPooling2D(2,2))
	right.add(Conv2D(256, (3,3), activation='relu'))
	right.add(MaxPooling2D(2,2))
	right.add(Flatten())
	right.add(Dense(1028, activation='relu', kernel_regularizer=l2(1e-2)))

	subtracted = Subtract()([left.output,right.output])
	subtracted = Dense(512, activation='sigmoid')(subtracted)
	subtracted = Dense(128, activation='sigmoid')(subtracted)
	out = Dense(2, activation='softmax')(subtracted)

	model = Model(inputs = [left.input, right.input], outputs = out)

	model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['accuracy'])

	return(model)



'''
For better results we can use a model trained on a large dataset and apply transfer learning.
We use a VGG16 model as the convolutional model to find the feature maps of the images.
This gives better results as it has been trained on millions of images and has learnt many relevant features.
In this function we create a siamese network with 2 vgg16 networks working in parallel.
each vgg16 model takes an inage and outputs the feature maps of these images
Theses image feature maps are then compared using a subtraction layer.
The subtraction layer then finds the distance between the feature maps of the images
this disctance is then taken as input to the a Fully connected layer which predicts if the images belong to the same subject.
In this function we create the model.
'''

def get_model_vgg(input_shape):

	vgg_left = VGG16(weights = 'imagenet',include_top = False, input_shape = input_shape)

	for layer in vgg_left.layers:
		layer.trainable = False
		layer._name = 'left_'+layer.name
		
	left = [layer.output for layer in vgg_left.layers][-5]

	left = Flatten()(left)
	# left = Dropout(0.5)(left)
	left = Dense(4096, kernel_regularizer=l2(1e-2))(left)
	left = BatchNormalization()(left)
	left = Activation('sigmoid')(left)


	vgg_right = VGG16(weights = 'imagenet',include_top = False, input_shape = input_shape)

	for layer in vgg_right.layers:
		layer.trainable = False
		layer._name = 'right_'+layer.name

	right = [layer.output for layer in vgg_right.layers][-5]

	right = Flatten()(right)
	# right = Dropout(0.5)(right)
	right = Dense(4096, kernel_regularizer=l2(1e-2))(right)
	right = BatchNormalization()(right)
	right = Activation('sigmoid')(right)


	subtracted = Subtract()([left,right])
	subtracted = Dense(1024, activation='sigmoid')(subtracted)
	subtracted = Dense(512, activation='sigmoid')(subtracted)
	out = Dense(2, activation='softmax')(subtracted)

	model = Model(inputs = [vgg_left.input,vgg_right.input], outputs = out)

	model.compile(loss="categorical_crossentropy", optimizer=Adam(0.001), metrics=['accuracy'])

	return(model)



'''
In the function one shot learning we use the trained model to test some validation data and compute the accuracy
'''

def one_shot_learning(model, n_way, n_val):
	
	#initializing the data for temporary use
	temp_x = X_test
	temp_cat_list = cat_test

	batch_x=[]
	x_0_choice=[]
	n_correct = 0
	
	# Class list is a list of random Categories from the test data of n_val length.
	class_list = np.random.randint(train_size+1, n_classes-1, n_val)

	for i in class_list:  
		# j = class_list[i] gets the random subject. 
		# J is a randomly chosen image from a randomly chosen subject from the test data.
		j = np.random.choice(cat_list[i])

		# temp is a list of 2 numpy arrays of shape (n_way, 100, 100, 3)
		temp=[]
		temp.append(np.zeros((n_way, 100, 100, 3)))
		temp.append(np.zeros((n_way, 100, 100, 3)))

		'''
		now in the for loop we are going to create n_way image pairs.
		the 1st image in all the pairs will be a random image j of the randomly chosen subject
		the second image for the first pair at position will belong to the the same subject as the subject of j
		the second image for all the other pairs will be from random subjects.
		'''
		for k in range(0, n_way):
			# Assigning the first pair of each image with the same image j of a random subject
			temp[0][k] = X_data[j]
			
			# Assigning the same subjects image as the second image of tbe first pair
			if k==0:
				temp[1][k] = X_data[np.random.choice(cat_list[i])]

			# Assigning the different subjects image as the second image of tbe all pairs except the first pair.
			else:
				temp_list = np.append(cat_list[:i], cat_list[i+1:])
				temp_list = np.random.choice(temp_list)
				temp[1][k] = X_data[np.random.choice(temp_list)]

		result = model.predict(temp)
		print(max(result.all()))
		exit(0)
	# 	result = result.flatten().tolist()
	# 	result_index = result.index(min(result))
	# 	if result_index == 0:
	# 		n_correct = n_correct + 1
	# print(n_correct, " correctly classified among ", n_val)
	# accuracy = (n_correct*100)/n_val
	# return accuracy



''' 
Now we will try to train the model and test its accuracy and performance.
Let us first initialize the model and then run it batch wise either with train_on_batch or fit
'''

print('initializing the siamese model')

enc = OneHotEncoder(sparse=False)

model = get_model((100,100,3))

# epochs = 20
# n_way = 20
# n_val = 64
# batch_size = 1280

print('Training')

# batch_x, batch_y = get_batch(X_train, cat_train, train_size, batch_size)
# batch_y = enc.fit_transform(batch_y.reshape(-1,1))
# history = model.fit(batch_x, batch_y,batch_size = 64,epochs = epochs)




# model = get_model_vgg((100,100,3))
epochs = 20
n_way = 10
n_val = 32
batch_size = 512

# print('Training')

loss_list=[]
accuracy_list=[]
for epoch in range(1,epochs):
	batch_x, batch_y = get_batch(X_train, cat_train, train_size, batch_size)
	batch_y = enc.fit_transform(batch_y.reshape(-1,1))
	loss = model.train_on_batch(batch_x, batch_y)
	loss_list.append((epoch,loss))
	print('Epoch:', epoch, ', Loss:',loss)
	if True:
		print("=============================================")
		accuracy = one_shot_learning(model, n_way, n_val)
		accuracy_list.append((epoch, accuracy))
		print('Accuracy as of', epoch, 'epochs:', accuracy)
		print("=============================================")
		if(accuracy>99):
			print("Achieved more than 90% Accuracy")	

model.save('siamese_vgg.h5')
