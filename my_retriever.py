import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()

        
    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms). Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        #print("QUERY:",query)
        return list(range(1,11))


    # Method for commputing the inverted document frequency of 
    # each term in the collection.
    # idf of a word(w) in a document collection(D) is: log(|D|/dfw) --> log of the total number of documents in collection over the number of documents containing w.
    def compute_idf_for_all_terms(self, index):

        all_terms = list(index.keys())
        total_doc_number = self.compute_number_of_documents()  

        idfs = {}
        for term in all_terms:
            idfs[term] = math.log(total_doc_number/self.compute_doc_freq(term))
                
        return idfs


    # Method for returning the document frequency of a term/word specified as a parameter.
    def compute_doc_freq(self, word):
        dfw = len(self.index[word])
        return dfw

