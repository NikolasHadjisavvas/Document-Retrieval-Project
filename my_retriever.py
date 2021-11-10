import math
import collections

class Retrieve:
    
    # Create new Retrieve object storing index and term weighting 
    # scheme. (You can extend this method, as required.)
    def __init__(self,index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.ALL_TERMS = list(self.index.keys())
        if self.term_weighting=='tfidf':
            self.idfs = self.compute_idf_for_all_terms()
        self.all_document_vectors,self.all_document_vec_lengths=self.construct_all_document_vectors()
        

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)


    # Method for commputing the inverted document frequency of 
    # each term in the collection.
    # idf of a word(w) in a document collection(D) is: log(|D|/dfw) --> log of the total number of documents in collection over the number of documents containing w.
    def compute_idf_for_all_terms(self):
        total_doc_number = self.compute_number_of_documents()  
        idfs = {}
        for term in self.ALL_TERMS:
            idfs[term] = math.log10(total_doc_number/self.compute_doc_freq(term))
        return idfs

    #Compute term weighting according to the selected scheme, at the same time build the document vectors and compute their length.
    def construct_all_document_vectors(self):
        all_doc_vectors={}
        doc_vector_element_summations={} #Initialise dict with the element wise summation of a vector.{doc vector->sum}.We use that later to apply sqrt and get the lenghts.
 
        #Iterate over all terms in index
        for term in self.ALL_TERMS:
            #For each document in the collection, compute the chosen term weighting for its terms.
            for doc in range(1,self.num_docs+1): 
                if doc in self.index[term]: #if doc contains term(thus, we drop all the vector elements that are going to be 0)
                    tf = self.index[term][doc] #Compute the tf of each term in each document(just by indexing the index)
                    if self.term_weighting=='tfidf':
                        idf = self.idfs[term] # Get the corresponding precomputed idf. 

                        # Since we take each term in index one by one, our vectors are going to be updated step by step in each loop of the outer for.
                        # Here we initialize the vector in case we found a term belonging to a specific document for the 1st time.
                        if doc not in all_doc_vectors: 
                            all_doc_vectors[doc] = {term:tf * idf}
                            doc_vector_element_summations[doc]= (tf * idf)**2
                        else: # In case the vector has already been initialised, just update its content.
                            all_doc_vectors[doc][term] = tf * idf
                            doc_vector_element_summations[doc]+= (tf * idf)**2
                    elif self.term_weighting =='tf': # Same as above but with tf
                        if doc not in all_doc_vectors:
                            all_doc_vectors[doc] = {term:tf}
                            doc_vector_element_summations[doc]= tf**2
                        else:
                            all_doc_vectors[doc][term] = tf
                            doc_vector_element_summations[doc]+= tf**2
                    elif self.term_weighting=='binary': # Same as above with tf
                        if doc not in all_doc_vectors:
                            all_doc_vectors[doc] = {term:1} # If term is in document, then weight in vector is 1.
                            doc_vector_element_summations[doc]= 1**2
                        else:
                            all_doc_vectors[doc][term] = 1
                            doc_vector_element_summations[doc]+= 1**2                      

        doc_vector_lengths={} #Initialise a dictionary {doc vector->doc length} to store vector lenghts.

        # Compute sqrt of element sums of vector elements to get lengths.
        for key,value in doc_vector_element_summations.items():
            doc_vector_lengths[key] = math.sqrt(value)

        return all_doc_vectors,doc_vector_lengths

    

    # Method for returning the document frequency of a term/word specified as a parameter.
    def compute_doc_freq(self, word):
        dfw = len(self.index[word])
        return dfw

    #Need to confirm that this is correct
    def compute_tfidfs_for_query_terms(self, query):
        tfidfs_of_query={} # Initialise dict {query term->tfidf}
        for word in query:
            tf = query.count(word) 
            if word in self.idfs: #If term is present in the document collection.Get the pre computed idf.
                idf=self.idfs[word]
                tfidfs_of_query[word] = tf*idf
        
        return tfidfs_of_query


    def construct_vector_for_query(self, query):

        if self.term_weighting=='tfidf': #If  we use tfidf, just use the above method.
            vector = self.compute_tfidfs_for_query_terms(query)
        if self.term_weighting=='tf': #If we use tf, just use the tf of query term in the query(only if the term also appears in index.)
            vector={}
            for word in query:
                if word in self.index:
                    vector[word]=query.count(word)
        if self.term_weighting=='binary': #If we use binary, just use the binary weight of query term in the query(only if the term also appears in index.)
            vector={}
            for word in query:
                if word in self.index:
                    vector[word] = 1

        return vector


    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
            query_vector = self.construct_vector_for_query(query)
            doc_vectors= self.all_document_vectors
            scores={}

            # Get all documents containing at least one query term.
            candidates=set()
            for word in query:
                if word in self.index:
                    candidates.update(self.index[word].keys())

            for i in candidates:
                #compute numerator
                numer = 0
                for word in list(query_vector.keys()):
                    if word in list(doc_vectors[i].keys()):
                        numer+=query_vector[word] * doc_vectors[i][word]
                
                denom = self.all_document_vec_lengths[i]
                
                scores[i] = numer/denom

            sorted_score_values = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            sorted_scores=collections.OrderedDict(sorted_score_values)
            
            return list(sorted_scores.keys())[:10]

