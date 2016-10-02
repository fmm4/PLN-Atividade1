from nltk.corpus import reuters
import nltk
import numpy as np
from operator import itemgetter
from math import log
from nltk.classify import SklearnClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

def extractTestTraining(conj,indice):
	trein = conj[:indice[0]]
	teste = conj[indice[0]:]
	return [trein,teste]


#Insert class and word, get chance of it being there
def getWordCount(classe, word,classesWords):
	if word in classesWords[classe]:	
		return classesWords[classe][word]+1
	else:
		return 1

#Sum of all words that exist in a class
def sumWords(classe,classesWords):
	sum = 0
	for word in classesWords[classe]:
		sum += classesWords[classe][word]+1
	return sum

#Sum of all documents that exist in all classes
def sumDocumentos(classes):
	sum = 0
	for doc in classes:
		sum += len(classes[doc])
	return sum

#
def getDocCount(classes,selectClass):
	return len(classes[selectClass])

# DEPRECATED: Usar word_features
# def getDocBoW(fileid):
# 	bagofwords = {}
# 	docWords = reuters.words(fileid)
# 	for word in docWords:
# 		if word in bagofwords:
# 			bagofwords[word] += 1
# 		else:
# 			bagofwords[word] = 1
# 	return bagofwords

def getProbWordInClass(word,selectClass,classesWords):
	countW = getWordCount(selectClass,word,classesWords)
	sumOfW = sumWords(selectClass,classesWords)
	return countW/sumOfW

def getProbOfClass(classes,selectClass):
	countC = getDocCount(classes,selectClass)
	sumOfC = sumDocumentos(classes)
	return countC/sumOfC

def getProbDictionary(classesDoc, classesWord):
	dicDeProb = {}
	for probClass in classesDoc:
		dicDeProb[probClass] = {}
		dicDeProb[probClass]['wordsProb'] = {}
		dicDeProb[probClass]['classProb'] = log(getProbOfClass(classesDoc,probClass)) 		
		for word in classesWord:
			dicDeProb[probClass]['wordsProb'][word] = log(getProbWordInClass(word,probClass,classesWord))
	return dicDeProb

def classBayes(word_features,probabilityDictionary,classesWords):
	documentsProbabilities = {}
	for testedClass in probabilityDictionary:
		prob = 0
		prob += probabilityDictionary[testedClass]['classProb']
		for word in word_features:
			if word in probabilityDictionary[testedClass]['wordsProb']:
				prob += probabilityDictionary[testedClass]['wordsProb'][word]*word_features[word]
			else:
				prob ++ log(getProbWordInClass(word,testedClass,classesWords))
		documentsProbabilities[testedClass] = prob
	return max(documentsProbabilities.items(), key=itemgetter(1))[0]


def word_features(doc,stemmer):
	words = {}
	for word in reuters.words(doc):
		if word not in stopwords.words('english'):
			stemmed_word = stemmer.stem(word)
			if word in words:
				words[word]+=1
			else:
				words[word]=1
	return words

def sumLine(i,confMatrix):
	sumOfLin = 0
	for k in confMatrix[i]:
		sumOfLin += confMatrix[i][k]
	return sumOfLin

def sumColumn(i,confMatrix):
	sumOfCol = 0
	for k in confMatrix:
		sumOfCol += confMatrix[k][i]
	return sumOfCol


def recall(i,confMatrix):
	cellValue = confMatrix[i][i]
	lineValue = sumLine(i,confMatrix)
	if lineValue == 0:
		lineValue = 1
	return cellValue/lineValue
	
def precision(i,confMatrix):
	cellValue = confMatrix[i][i]
	colValue = sumColumn(i,confMatrix)
	if colValue == 0:
		colValue = 1
	return cellValue/colValue

def sumRight(confMatrix):
	sumRight = 0
	for i in confMatrix:
		sumRight += confMatrix[i][i]
	return sumRight

def sumMatrix(confMatrix):
	sumTotal = 0
	for i in confMatrix:
		for k in confMatrix[i]:
			sumTotal+= confMatrix[i][k]
	return sumTotal

def f_measure(recall,precision):
	if(recall+precision == 0):
		return 0
	return (2*recall*precision)/(recall+precision)

def accuracy(confMatrix):	
	return sumRight(confMatrix)/sumMatrix(confMatrix)


def main():
	# documentWords = [word
	# 				for category in reuters.categories()
	# 				for fileid in reuters.fileids(category)
	# 				for word in reuters.words(fileid)]
	documents = [category
					for category in reuters.categories()
					for fileid in reuters.fileids(category)]
	k = []
	for i in reuters.categories():
		k.append([i,documents.count(i)])
	k = sorted(k,key=itemgetter(1))
	k = k[::-1]
	k = k[0:10]
	#DOcumentos validos

	stemmer = SnowballStemmer("english")
	treinamento = {}
	teste = {}
	docTotal = []
	nrDocumentos = [[2877,1087]
					,[1650, 179]
					,[538, 179]
					,[433, 149]
					,[389, 189]
					,[369, 119]
					,[347, 131]
					,[197, 89]
					,[212, 71]
					,[182, 56]]
	for i in range(0,10):
		docTotal = [fileid for fileid in reuters.fileids(k[i][0])]
		docsUsaveis = extractTestTraining(docTotal,nrDocumentos[i])
		treinamento[k[i][0]] = docsUsaveis[0]
		teste[k[i][0]] = docsUsaveis[1]

	wordDictionary = {}
	for classe in treinamento:
		wordDictionary[classe] = {}
		for doc in treinamento[classe]:
			for word in reuters.words(doc):
				if word not in stopwords.words('english'):
					treated_word = stemmer.stem(word)
					if word in wordDictionary[classe]:
						wordDictionary[classe][word]+=1
					else:
						wordDictionary[classe][word]=1


	probabilityDictionary = getProbDictionary(treinamento,wordDictionary)


	trainset = []
	labeled_train = ([(word_features(treatdoc,stemmer),i) for i in treinamento for treatdoc in treinamento[i]])	
	labeled_test =  ([(word_features(treatdoc,stemmer),i) for i in teste for treatdoc in teste[i]])

	classifier = nltk.NaiveBayesClassifier.train(labeled_train)
	SVCClassifier =   SklearnClassifier(SVC()).train(labeled_train)


	# print([classes for classes in wordDictionary])

	confMatrixBayesProg = {}
	confMatrixSVCProg = {}
	confMatrixBayesUser = {}
	for y in k:
		confMatrixBayesProg[y[0]] = {}
		confMatrixSVCProg[y[0]] = {}
		confMatrixBayesUser[y[0]] = {}
		for x in k:
			confMatrixBayesProg[y[0]][x[0]] = 0
			confMatrixSVCProg[y[0]][x[0]] = 0
			confMatrixBayesUser[y[0]][x[0]] = 0

	total = 0

	for label in labeled_test:
		total+=1




	progress = 0
	for i in labeled_test:
		print("Progress: %.2f [%d of %d]" % (progress/total,progress,total))
		progress+=1
		resultProg = classifier.classify(i[0])
		confMatrixBayesProg[i[1]][resultProg]+=1
		resultSVC = SVCClassifier.classify(i[0])
		confMatrixSVCProg[i[1]][resultSVC]+=1
		resultUser = classBayes(i[0],probabilityDictionary,wordDictionary)
		confMatrixBayesUser[i[1]][resultUser]+=1
		# print("%s:	%s [Bayes-NLTK] %s [SVC-scikit] %s [Our Bayes]" % (i[1],resultProg,resultSVC,resultUser))



	print("bayesconfmatrx")
	print(confMatrixBayesProg)

	print("ourconfmatrx")
	print(confMatrixBayesUser)

	print("svcconfmatrx")
	print(confMatrixSVCProg)

	getAccuracyProg = accuracy(confMatrixBayesProg)
	print("Bayes Programa")
	print("Acruacia")
	print(getAccuracyProg)
	for e in confMatrixBayesProg:
		print("Classificando classe %s:" % e)
		print("Recall")
		recallProg = recall(e,confMatrixBayesProg)
		print(recallProg)
		print("Precision")
		precisionProg = precision(e,confMatrixBayesProg)
		print(precisionProg)
		print("F-Measure")
		print(f_measure(recallProg,precisionProg))

	getAccuracyUser = accuracy(confMatrixBayesUser)
	print("Bayes Alunos")
	print("Acruacia")
	print(getAccuracyUser)
	for e in confMatrixBayesUser:
		print("Classificando classe %s:" % e)
		print("Recall")
		recallUser = recall(e,confMatrixBayesUser)
		print(recallUser)
		print("Precision")
		precisionUser = precision(e,confMatrixBayesUser)
		print(precisionUser)
		print("F-Measure")
		print(f_measure(recallUser,precisionUser))

	getAccuracySVC = accuracy(confMatrixSVCProg)
	print("SVC scikit")
	print("Acruacia")
	print(getAccuracySVC)
	for e in confMatrixSVCProg:
		print("Classificando classe %s:" % e)
		print("Recall")
		recallProg = recall(e,confMatrixSVCProg)
		print(recallProg)
		print("Precision")
		precisionProg = precision(e,confMatrixSVCProg)
		print(precisionProg)
		print("F-Measure")
		print(f_measure(recallProg,precisionProg))

	# for i in teste:
	# 	print("Classfying for %s:" % i)
	# 	results = []
	# 	for c in range(0,4):
	# 		result = classBayes(teste[i][c],treinamento,classificador)
	# 		results.append(result)
	# 	print("%d correct." % results.count(i))
	# 	print("%d total." % len(results))


if __name__ == "__main__":
    main()
