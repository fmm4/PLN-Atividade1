from nltk.corpus import reuters
import nltk
import numpy as np
from operator import itemgetter
from math import log

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

def getDocBoW(fileid):
	bagofwords = {}
	docWords = reuters.words(fileid)
	for word in docWords:
		if word in bagofwords:
			bagofwords[word] += 1
		else:
			bagofwords[word] = 1
	return bagofwords

def getProbWordInClass(word,selectClass,classesWords):
	countW = getWordCount(selectClass,word,classesWords)
	sumOfW = sumWords(selectClass,classesWords)
	return countW/sumOfW

def getProbOfClass(classes,selectClass):
	countC = getDocCount(classes,selectClass)
	sumOfC = sumDocumentos(classes)
	return countC/sumOfC

def classBayes(doc,classesDoc, classesWords):
	docBoW = getDocBoW(doc)
	documentsProbabilities = {}
	for testedClass in classesDoc:
		prob = 0
		prob += log(getProbOfClass(classesDoc,testedClass))
		for word in docBoW:
			prob += log(getProbWordInClass(word,testedClass,classesWords))
		documentsProbabilities[testedClass] = prob
	return max(documentsProbabilities.items(), key=itemgetter(1))[0]


def word_features(doc):
	words = {}
	for word in reuters.words(doc):
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
	return cellValue/lineValue
	
def precision(i,confMatrix):
	cellValue = confMatrix[i][i]
	colValue = sumColumn(i,confMatrix)
	return cellValue/colValue

def sumRight(confMatrix):
	sumRight = 0
	for i in confMatrix:
		sumRight = confMatrix[i][i]
	return sumRight

def sumMatrix(confMatrix):
	sumTotal = 0
	for i in confMatrix:
		for k in confMatrix[i]:
			sumTotal+= confMatrix[i][k]
	return sumTotal

def f_measure(recall,precision):
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

	classificador = {}

	# vocabulario = []
	# for i in docTotal:
	# 	for word in reuters.words(i):
	# 		vocabulario.append(word)

	# vocabularioList = list(set(vocabulario))


	# for classe in treinamento:
	# 	classificador[classe] = {}
	# 	for word in vocabularioList:
	# 		classificador[classe][word] = 0
	# 	for doc in treinamento[classe]:
	# 		words = reuters.words(doc)
	# 		for palavra in words:
	# 			classificador[classe][palavra] += 1



	trainset = []
	labeled_train = ([(word_features(treatdoc),i) for i in treinamento for treatdoc in treinamento[i]])	
	labeled_test =  ([(word_features(treatdoc),i) for i in teste for treatdoc in teste[i]])


	classifier = nltk.NaiveBayesClassifier.train(labeled_train)

	
	sumRecall = 0
	sumPrecision= 0

	confMatrix = {
	for y in k:
		confMatrix[y[0]] = {}
		for x in k:
			confMatrix[y[0]][x[0]] = 0

	for i in labeled_test:
		result = classifier.classify(i[0])
		confMatrix[i[1]][result]+=1




	getAccuracy = accuracy(confMatrix)
	print("Acruacia")
	print(getAccuracy)
	for e in confMatrix:
		print("Classificando classe %s:" % e)
		print("Recall")
		recallE = recall(e,confMatrix)
		print(recallE)
		print("Precision")
		precisionE = precision(e,confMatrix)
		print(precisionE)
		print("F-Measure")
		print(f_measure(recallE,precisionE))



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
