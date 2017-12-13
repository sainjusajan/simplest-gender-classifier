from sklearn import tree

# For input we take height, weight and size of shoes.
X = [[199, 122, 23], [123,344, 322], [122,344,222], [455, 222,422], [233, 334, 222], [233,345,566], [122,344,455], [199, 122, 23], [123,344, 322], [122,344,222]]
# For dummy results
Y = ['male', 'female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'male']

# we make the classifier now
# first call a decisionTreeClassifier and make an instance of it i.e. clf
clf = tree.DecisionTreeClassifier()
# fit the classifier with the dummy data above
clf.fit(X, Y)

# generating the predicted value
prediction = clf.predict([[123,343,320]])
print(prediction)
