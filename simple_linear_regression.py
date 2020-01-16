# 2D simple linear regression algorithm, currently first attribute must be dependant variable and second the dependant
# Done mostly without the use of external modules, only pyplot and numpy for ploting and displaying the graph and random for splitting the data

import matplotlib.pyplot as plt
import numpy as np
import random


def importData(filename):
	"""
	Extracts and returns data from provided CSV file in a Python Dictionary

	:param filename: file name/path of csv data file
	:type filename: str
	:return dataset: dictionary with indexes of CSV column titles and assosiated data
	:rtype: dict

	"""
	with open(filename, 'r') as file:

		dataset = {}

		for index, row in enumerate(file):
			if index != 0:
				for index, d in enumerate(dataset):
					dataset[d].append(float(row.split()[index]))

			else:	
				for col in row.split():
					dataset[col] = []

	return dataset


def preprocessData(attributes, dependantVariable, dataset,
					split_at_index=False, split_index=0, 
					split_at_ratio=False, test_split_ratio=0.1, random_split=False):
	"""
	Data preprocessing, if split_at_index is selected, will simply return training and testing data 
	split at the given index. 
	If split_at_ratio is selected, will either split data at the given ratio if random_split is False,
	or split at the ratio but with randomly selected values from dataset at the given attribute if
	is set to True

	:param attributes: list of CSV column titles for extraction from dataset dictionary
	:type attributes: list
	:param dependantVariable: name of column in CSV file that will be used as dependant variable
	:type dependantVariable: str 
	:param dataset: dataset dictionary returned from importData method
	:type dataset: dict

	:param split_at_index: if set to True will split the dependant and independant variables at the
							provided index in split_index, default (False)
	:type split_at_index: boolean

	:param split_index: if split_at_index is True is the index  that will be used to split the data 
						into training and testing, default (0 or None)
	:type split_index: int

	:param split_at_ratio: if set to True will split data into training and testing based on ratios
							not a fixed index, default (False)
	:type split_at_ratio: boolean

	:param test_split_ratio: ratio used if split_at_ratio is set to True, default (0.1)
	:type test_split_ratio: int

	:param random_split: if True splits data randomly between training and testing, but still at
							the given ratio, default (False)
	:type random_split: boolean

	:return X_test: independant test variables 
	:rtype X_test: list
	:return y_test: dependant test variables
	:rtype y_test: list
	:return X_train: independant train variables
	:rtype X_train: list
	:return y_train: dependant train variables
	:rtype y_train: list 

	"""
	y = dataset.get(dependantVariable)
	for attr in attributes:
		if attr != dependantVariable:
			X = dataset.get(attr)

	if split_at_ratio and not split_at_index:
		# Code to split with a given ratio
		if random_split:
			# Split the data randomly but still at the same ratios
			rt = int(len(X) * test_split_ratio)  # ratio formula
			dataX = X[:]
			datay = y[:]
			X_test = [None] * rt
			y_test = [None] * rt
			
			for index, _ in enumerate(X_test):
				rand = random.randint(0, len(dataX)-1)   # returning an index
				X_test[index] = dataX[rand]
				y_test[index] = datay[rand]
				dataX.pop(rand)
				datay.pop(rand)

			rt = len(dataX) - rt

			X_train = [None] * rt
			y_train = [None] * rt

			for index, _ in enumerate(X_train):
				rand = random.randint(0, len(dataX)-1)
				X_train[index] = dataX[rand]
				y_train[index] = datay[rand]
				dataX.pop(rand)
				datay.pop(rand)		

		else:
			# Split the data into fixed ratios
			rt = int(len(X) * test_split_ratio)  # ratio formula
			X_test = X[:rt]
			y_test = y[:rt]
			X_train = X[rt:]
			y_train = y[rt:]

	elif split_at_index and not split_at_ratio:
		# Code to split at a given index
		X_test = X[:split_index]
		y_test = y[:split_index]
		X_train = X[split_index:]
		y_train = y[split_index:]

	elif split_at_index and split_at_ratio:
		raise ValueError("Please only select one split method - Either split_at_index or split_at_ratio!")

	else:
		raise ValueError("Please select a split method to perform - Either split_at_index or split_at_ratio!")

	return X_test, y_test, X_train, y_train


def findLine(X_train, y_train):
	"""
	Method for calculating line of best fit with the provided training data

	:param X_train: independnt variables list returned from preprocessingData method
	:type X_train: list
	:param y_train: dependant variable list returned from preprocessingData method
	:type t_train: list

	:return gradient: gradient for line of best fit in the provided training dataset
	:rtype gradient: int
	:return y_intercept: y-intercept for line of best fit in the provided training dataset
	:rtype y_intercept: int

	"""

	# Calculating mean averages of Dependatnt variables and Independant Variables
	_X = sum(X_train) / len(X_train)    # mean average of all X values
	_y = sum(y_train) / len(y_train)    # mean average of all y values

	# Finding the gradient of the line
	#  (xi−X¯¯¯)
	xi__X = [x - _X for x in X_train]

	#  (yi−Y¯¯¯)
	yi__y = [y - _y for y in y_train]

	# xi__X_yi__Y
	xi__X_yi__Y = sum([xi__X[i] * yi__y[i] for i in range(len(xi__X))])

	# xi__X2
	xi__X2 = sum([(X_train[i] - _X) ** 2 for i in range(len(xi__X))])

	# Gradient
	gradient = xi__X_yi__Y / xi__X2

	# Finding Y-Intecept
	y_intercept = _y - gradient * _X

	return gradient, y_intercept


def equation(x, gradient, y_intercept):
	"""
	Function for calculating y value at any point on line of best fit from a provided X value

	:param x: x axis value for passing into equation
	:type x: int
	:param gradient: gradient returned from findLine funtion
	:type gradient: int
	:param y_intercept: y-intercept returned from findLinefunction
	:type y_intercept: int

	:return y: y axis value on line of best fit from provided x axis value
	:rtype y: int

	"""
	return gradient * x + y_intercept				


def plotGraph(X_test, y_test, X_train, y_train, gradient, y_intercept, attributes,
				xlim=(-350,650), ylim=(0,500), linestart=-1000, lineend=1000, lineaxes=True, plotextrapoint=True, pointx=-273.15):
	"""
	Function for plotting graph using matplotlib.pyplot, using a basic scatterplot then overlaying
	with line of best fit, also able to plot a prediction point if plotextrapoint is set to  True
	and an X-value is provided in the pointx parameter

	:param X_test: independant test variables returned from processing data function
	:type: list
	:param y_test: dependant test variables returned from processing data function
	:type: list
	:param X_train: independant train variables returned from processing data function
	:type: list
	:param y_train: dependant train variables returned from processing data function
	:type: list 
	
	:param gradient: gradient returned from findLine funtion
	:type gradient: int
	:param y_intercept: y-intercept returned from findLinefunction
	:type y_intercept: int
	:param attributes: attributes to be extracted from CSV file, must be array of length 2, 
						first index is the independant variable, second is the dependant variable

	:param xlim: pyplots x axis limits, tuple with start value and end value, default (-350, 650)
	:type xlim: tuple
	:param xlim: pyplots y axis limits, tuple with start value and end value, default (0, 500)
	:type xlim: tuple
	
	:param linestart: x coordinated of where the line of best fit begins, default (-1000)
	:type linestart: int
	:param linestart: x coordinated of where the line of best fit ends, default (1000)
	:type linestart: int

	:param lineaxes: if True draws lines within the plot to represent x and y axes, default (True)
	:type lineaxes: boolean
	:param plotextrapoint: is True plots a point from a given x value in pointx for predictions, default (True)
	:type plotextrapoint: True
	:param pointx: x value to plot prediciton point at, default (-273.15)
	:type pointx: int

	"""

	plt.scatter(X_train, y_train)
	plt.scatter(X_test, y_test)

	plt.xlabel(attributes[0].upper()[0] + attributes[0][1:] + "    X axis - Independant Variable")   # X axis is our independant variable
	plt.ylabel(attributes[1].upper()[0] + attributes[1][1:] + "    Y axis - Dependant Variable")     # Y axis is out dependant variable

	plt.xlim(xlim[0], xlim[1])
	plt.ylim(ylim[0], ylim[1])

	x = np.linspace(linestart, lineend)
	y = gradient * x + y_intercept

	if lineaxes:
		plt.plot([0, 0], [-1000, 1000], color='k')   # y-axis
		plt.plot([-2500, 2500], [0, 0], color='k')   # x-axis

	if plotextrapoint:
		# Plot extra point
		plt.scatter(pointx, equation(pointx, gradient, y_intercept), color='m')

	# Plot line of best fit
	plt.plot(x, y, '-r', label='y=mx+c')

	plt.show()


def run(attributes, splitIndex, filename, dependantVariable):
	"""
	Function ot run to extract data from CSV file, preprocess the data, find line of best fit
	then display the information in a matplotlib.pyplot graph

	:param attributes: attributes to be extracted from CSV file, must be array of length 2, 
						first index is the independant variable, second is the dependant variable
	:type arrtibutes: list
	param splitIndex: index in the data that will be split into training and testing, lower half is 
						testing, upper is training
	:type splitIndex: int
	:param filename: file name/path of csv data file
	:type filename: str
	:param dependantVariable: name of column in CSV file that will be used as dependant variable
	:type dependantVariable: str

	"""
	dataset = importData(filename)
	# This is the line to run for physics demo
	X_test, y_test, X_train, y_train = preprocessData(attributes, dependantVariable, dataset, split_at_index=True, split_index=splitIndex)
	print("X: ", X_train)
	print("Y: ", y_train)
	gradient, y_intercept = findLine(X_train, y_train)
	plotGraph(X_test, y_test, X_train, y_train, gradient, y_intercept, attributes)


run(['temperature', 'collisions'], 13, 'pressure.csv', 'collisions')
	