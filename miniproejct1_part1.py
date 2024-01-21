from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download('punkt')
from datasets import load_dataset
import numpy.linalg as la
import time

#Remaining to-do
#---------------
#1. Compare time-taken between GloVe and sentence-transformer
 
class TextSimilarityModel:
    def __init__(self, corpus_name, rel_name, model_name='all-MiniLM-L6-v2', top_k=10):
        """
        Initialize the model with datasets and pre-trained sentence transformer.
        """
        self.model = SentenceTransformer(model_name)
        self.corpus_name = corpus_name
        self.rel_name = rel_name
        self.top_k = top_k
        self.load_data()


    def load_data(self):
        """
        Load and filter datasets based on test queries and documents.
        """
        # Load query and document datasets
        dataset_queries = load_dataset(self.corpus_name, "queries")
        dataset_docs = load_dataset(self.corpus_name, "corpus")

        # Extract queries and documents
        self.queries = dataset_queries["queries"]["text"]
        self.query_ids = dataset_queries["queries"]["_id"]
        self.documents = dataset_docs["corpus"]["text"]
        self.document_ids = dataset_docs["corpus"]["_id"]

        # Filter queries and documents based on test set
        qrels = load_dataset(self.rel_name)["test"]
        self.filtered_query_ids = set(qrels["query-id"])
        self.filtered_doc_ids = set(qrels["corpus-id"])

        self.queries = [q for qid, q in zip(self.query_ids, self.queries) if qid in self.filtered_query_ids]
        self.query_ids = [qid for qid in self.query_ids if qid in self.filtered_query_ids]
        self.documents = [doc for did, doc in zip(self.document_ids, self.documents) if did in self.filtered_doc_ids]
        self.document_ids = [did for did in self.document_ids if did in self.filtered_doc_ids]

        self.query_id_to_relevant_doc_ids = {qid: [] for qid in self.filtered_query_ids}
        for qid, doc_id in zip(qrels["query-id"], qrels["corpus-id"]):
            if qid in self.query_id_to_relevant_doc_ids:
                self.query_id_to_relevant_doc_ids[qid].append(doc_id)
    
        """"BOILERPLATE STARTS....
    def load_glove_embeddings(filepath):
        """
        #Load GloVe embeddings from a file path.
        """
        embeddings = {}
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        return embeddings

    def sentence_embeddings(sentences, embeddings):
        """
        #Generate sentence embeddings by averaging word embeddings.
        """
        sentence_embeddings = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            sentence_vector = np.mean([embeddings[word] for word in words if word in embeddings], axis=0)
            sentence_embeddings.append(sentence_vector)
        return sentence_embeddings

    # Load GloVe embeddings
    embeddings = load_glove_embeddings('glove.6B.50d.txt')

    # List of sentences
    sentences = ['This is the first sentence.', 'This is another sentence.']

    # Generate sentence embeddings
    sentence_embs = sentence_embeddings(sentences, embeddings)
        #####BOILERPLATE STARTS"""

    def encode_with_glove(self, glove_file_path, sentences):
        """
        Encodes sentences by averaging GloVe 50d vectors of words in each sentence.
        Return a sequence of embeddings of the sentences.
        Download the glove vectors from here. 
        https://nlp.stanford.edu/data/glove.6B.zip
        """
        #TODO Put your code here. 
        ###########################################################################
        embeddings = {}
        with open(glove_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype='float32')
                embeddings[word] = vector
        
        sentence_embeddings = []
        for sentence in sentences:
            words = nltk.word_tokenize(sentence) #sentence.split()
            sentence_vector = np.mean([embeddings[word] for word in words if word in embeddings], axis=0)
            sentence_embeddings.append(sentence_vector)
        return sentence_embeddings
        ###########################################################################
    
    @staticmethod
    def cosine_similarity(x, y):

        return np.dot(x,y)/max(la.norm(x)*la.norm(y),1e-3)

    def rank_documents(self, encoding_method='sentence_transformer'):
        """
        (1) Compute cosine similarity between each document and the query
        (2) Rank documents for each query and save the results in a dictionary "query_id_to_ranked_doc_ids" 
            This will be used in "mean_average_precision"
            Example format {2: [125, 673], 35: [900, 822]}
        """
        if encoding_method == 'glove':
            query_embeddings = self.encode_with_glove(r"D:\UW courses\EE 596 A - LLM\Mini Project 1\glove.6B\glove.6B.50d.txt", self.queries)
            query_embeddings = np.array(query_embeddings)
            document_embeddings = self.encode_with_glove(r"D:\UW courses\EE 596 A - LLM\Mini Project 1\glove.6B\glove.6B.50d.txt", self.documents)
            document_embeddings = np.array(document_embeddings)
        elif encoding_method == 'sentence_transformer':
            query_embeddings = self.model.encode(self.queries)
            document_embeddings = self.model.encode(self.documents)
        else:
            raise ValueError("Invalid encoding method. Choose 'glove' or 'sentence_transformer'.")
        
        #TODO Put your code here.
        ###########################################################################
        from sklearn.metrics.pairwise import cosine_similarity

        # Initialize an empty dictionary to store the ranked document IDs for each query
        self.query_id_to_ranked_doc_ids = {}

        print('query_embeddings.shape: ',query_embeddings.shape)
        print('document_embeddings.shape: ',document_embeddings.shape)

        # Compute cosine similarity between each document and the query
        for i, query_embedding in enumerate(query_embeddings):
            try:
                similarities = cosine_similarity(query_embedding.reshape(1,-1), document_embeddings)
                # Get the indices of the documents in descending order of similarity
                ranked_doc_ids = np.argsort(similarities[0])[::-1]

                # Save the ranked document IDs for this query
                #query_id_to_ranked_doc_ids = [lst[i] for i in indices]
                self.query_id_to_ranked_doc_ids[self.query_ids[i]] = [self.document_ids[i] for i in ranked_doc_ids.tolist()]
            except:
                continue
        ###########################################################################

    @staticmethod
    def average_precision(relevant_docs, candidate_docs):
        """
        Compute average precision for a single query.
        """
        y_true = [1 if doc_id in relevant_docs else 0 for doc_id in candidate_docs]
        precisions = [np.mean(y_true[:k+1]) for k in range(len(y_true)) if y_true[k]]
        return np.mean(precisions) if precisions else 0

    def mean_average_precision(self):
        """
        Compute mean average precision for all queries using the "average_precision" function.
        A reference: https://towardsdatascience.com/breaking-down-mean-average-precision-map-ae462f623a52
        """
         #TODO Put your code here. 
        ###########################################################################
        average_precisions = []

        # For each query, compute the average precision
        for query_id, candidate_docs in self.query_id_to_ranked_doc_ids.items():
            relevant_docs = self.query_id_to_relevant_doc_ids[query_id][:self.top_k]
            print("relevant_docs: ",relevant_docs)
            print("candidate_docs[:self.top_k]: ",candidate_docs[:self.top_k])
            ap = self.average_precision(relevant_docs, candidate_docs[:self.top_k])
            average_precisions.append(ap)

        # Compute the mean average precision
        map_score = np.mean(average_precisions)

        return map_score
        
        ###########################################################################

    def show_ranking_documents(self, example_query):
        """
        (1) rank documents with given query with cosine similaritiy scores
        (2) prints the top 10 results along with its similarity score.
        
        """
        query_to_docs = {}
        #TODO Put your code here. 
        query_embedding = self.model.encode(example_query)
        document_embeddings = self.model.encode(self.documents)
        ###########################################################################
        similarities = cosine_similarity(query_embedding.reshape(1,-1), document_embeddings)

        # Get the indices of the documents in descending order of similarity
        ranked_doc_ids = np.argsort(similarities[0])[::-1]

        ranked_doc_scores = np.sort(similarities[0])[::-1]

        # Get documents in ranked order
        query_to_docs[example_query] = [self.documents[i] for i in ranked_doc_ids.tolist()]

        print("Top 10 documents matching the input query:")
        for i in range(10):
            print(i+1,". ",query_to_docs[example_query][i])
        #################################G##########################################


# Initialize and use the model
model = TextSimilarityModel("BeIR/nfcorpus", "BeIR/nfcorpus-qrels",top_k=15)

print("Ranking with sentence_transformer...")
model.rank_documents(encoding_method='sentence_transformer')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


print("Ranking with glove...")
model.rank_documents(encoding_method='glove')
map_score = model.mean_average_precision()
print("Mean Average Precision:", map_score)


model.show_ranking_documents("Breast Cancer Cells Feed on Cholesterol")