# Clustering
Clustering System of Short Text Descriptions
The objective of this pipeline, is to build a logic that can cluster short texts into similar root causes. All of these texts come from infrastructure and applications issues.

## Pre procesing
The cleaning up of the data. As our training data we used texts in different languages (english, spanish, french and chinese). In this phase, we tried to separate the languages so we can build models of each and let the algorithm determine which language is the text in and keep training the selected model.
We also cleaned the data by taking out stop words, greetings, farewells, and bringing the verbs back to their infinitive form.

## Data Transformation into Vectors
We turn the text into vectors so that the computer can interpret them. In our structure, each row in the vector represents one of the text's descriptions. We got our main inspiration from Mikolov's work, so we implemented Doc2Vec.
We train out model using Mikolov's proposed algorithm.

## Dimensionality Reduction
Because a computer will take forever to analyze all of the texts inputted, we used a dimensionality reduction algorithm to reduce the size of the vector into a two dimensional array.
In our pipeline we have two implementations of dimensionality reduction. The one to be used depends on the data.
PCA: We used it on data sets that tend to be linear.
T-SNE: We used it on data sets that had a circular shape.
We still are working on a solution on how to determine the shape of the dataset without having a human intervine in the process.

## Clustering
We proposed clustering using k-means. To determine the precise number of clusters, we used a measuring technique called Silhouette Analysis. This techniques measures the distances of each data point and thus would lead us to the ideal number of clusters. In other words, silhouette analysis would measure the similarity between each data point and thus tell us the ideal clusters number.

## Evaluation
As an evaluation we used the Calinski Harabasz Index. This evaluation technique in my opinion should be combined with another technique. CHI uses Euclidean distances to measure, so because K-means used Euclidean distance as well to cluster its points, then the results would tend to be positive.

## Models used
Spacy english corpora	en-core-web-sm	2.0.0

## Needed specs for running
Python 3x 64 bits
Anaconda with a built envirnoment
Pip version 10.0.1
Spyder version 3.2.8 (This can be changed to the user preference)

## Library versions
Numpy			1.14.5
Pandas			0.23.1
Matplotlib		2.2.2
Testfixtures		6.2.0
Spacy			2.0.11
Gensim			3.4.0
Scikit-learn		0.19.1
Scipy			1.1.0
Xlrd			1.1.0
