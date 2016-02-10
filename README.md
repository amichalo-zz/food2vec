###About###

Food2Vec is a simple project exposing ability of Word2Vec to learn and represent knowledge. In this case we are trying to teach an underling network, what humanity learned during thousands of years - which products are similar in taste and how to combine them.

In theory each ingredient contains many flavour compounds. In recipes we are more willing to combine the ingredients which are sharing more flavour components. If you would like to get to know more about it check an article [Flavor network and the principles of food pairing](http://www.nature.com/articles/srep00196) which together with H. Z. Lo's work was an inspiration to this project.

###Requirements###

To run the project you need to have installed:
- [gensim](https://radimrehurek.com/gensim/)
- [scikit-learn](http://scikit-learn.org/)

###Project###

- *receipts_data* - contains recipes for different cuisines
- *train.py* - trains Word2Vec model, which is then saved in *food2vec.model.txt*
- *plot.py* - loads and plots model using t-SNE dimensionality reduction
- *clustering.py* - loads model and creates groups of similar ingredients
