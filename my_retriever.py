import math

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.ALL_TERMS = list(self.index.keys())
        self.idfs = self.compute_idf_for_all_terms(self.index)

        query=['what', 'articles', 'exist', 'which', 'deal', 'with', 'tss', 'time', 'sharing', 'system', 'an', 'operating', 'system', 'for', 'ibm', 'computers']
        print(self.compute_cos(query,2020))

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)

    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
        #print("QUERY:",query)
        return list(range(1,11))


    # Method for commputing the inverted document frequency of 
    # each term in the collection.
    # idf of a word(w) in a document collection(D) is: log(|D|/dfw) --> log of the total number of documents in collection over the number of documents containing w.
    def compute_idf_for_all_terms(self, index):
        total_doc_number = self.compute_number_of_documents()  
        idfs = {}
        for term in self.ALL_TERMS:
            idfs[term] = math.log(total_doc_number/self.compute_doc_freq(term))

        return idfs


    # Takes a term and a document
    # Returns tfidf of that term for all documents
    # Note: Expensive to compute for all documents(because later I will need to index all values as well), 
    # many of them are 0 anyway, I can compute and store only the non zero,and for the vector length computation, check if
    # doc is present in the dict of all tfidfs for the term, if not present then we know the tfidf is 0.
    def compute_tdidfs_for_document_terms(self, document):

        words = self.getWordsInDoc(document)
        tfidfs_of_document = {}
        for word in words:
            tf = words.count(word)
            idf=self.idfs[word]
            tfidfs_of_document[word] = tf*idf
        
        return tfidfs_of_document

    
    def compute_tdidfs_for_query_terms(self, query):

        tfidfs_of_query={}
        for word in query:
            tf = query.count(word)
            idf=self.idfs[word]
            tfidfs_of_query[word] = tf*idf
        
        return tfidfs_of_query


    def getWordsInDoc(self, document):
        arr=[]
        items=self.index.items()
        for item in items:
            docs_with_word=item[1].keys()
            if document in docs_with_word:
                arr.append(item[0])
        return arr

    # Takes a term and a document.
    # Returns 1 if the term is present in the document, else returns 0.
    def compute_binary_weighting(self, term, document):
        term_in_index = self.index[term]    #{1234:1, 1432:2}
        docs = list(term_in_index.keys())
        if document in docs:
            return 1
        else:
            return 0 


    # Method for returning the document frequency of a term/word specified as a parameter.
    def compute_doc_freq(self, word):
        dfw = len(self.index[word])
        #test for commit
        return dfw

    
    def construct_vector_for_doc(self, document):

        #if we use tfidf weghting, use the tfidf method,if useing binary,use the binary version of the method etc.
        vector=self.compute_tdidfs_for_document_terms(document)
        
        return vector

    
    def construct_vector_for_query(self, query):

        vector = self.compute_tdidfs_for_query_terms(query)
        
        return vector

    
    def compute_numerator_of_cos(self, query,document):
    
        doc_vector = self.construct_vector_for_doc(document)
        query_vector = self.construct_vector_for_query(query)

        numer = 0
        for word in query:
            if word in list(doc_vector.keys()):
                numer+=query_vector[word] * doc_vector[word]

        return numer

    def compute_denominator_of_cos(self,query,document):
        doc_vector = self.construct_vector_for_doc(document)
        query_vector = self.construct_vector_for_query(query)  

        denom=0
        sumq=0
        sumd=0

        for word in query:
            sumq += query_vector[word]*query_vector[word]
        for word in doc_vector.keys():
            sumd+=doc_vector[word]*doc_vector[word]

        denom = math.sqrt(sumq) * math.sqrt(sumd)

        return denom

    def compute_cos(self,query,document):
        return self.compute_numerator_of_cos(query,document)/self.compute_denominator_of_cos(query,document)
