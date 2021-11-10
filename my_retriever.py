import math
import collections
import time

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
 
        for term in self.ALL_TERMS:
            idf = self.idfs[term]
            for doc in self.index[term]:
                if self.term_weighting == 'tfidf':
                    if doc not in all_doc_vectors:
                        tf=self.index[term][doc]
                        all_doc_vectors[doc] = {term: tf*idf}
                        doc_vector_element_summations[doc]= (tf * idf)**2
                    else: # In case the vector has already been initialised, just update its content.
                        tf=self.index[term][doc]
                        all_doc_vectors[doc][term] = tf * idf
                        doc_vector_element_summations[doc]+= (tf * idf)**2
                if self.term_weighting == 'tf':
                    if doc not in all_doc_vectors:
                        tf=self.index[term][doc]
                        all_doc_vectors[doc] = {term: tf}
                        doc_vector_element_summations[doc]= tf**2
                    else: # In case the vector has already been initialised, just update its content.
                        all_doc_vectors[doc][term] = tf
                        doc_vector_element_summations[doc]+= tf**2
                if self.term_weighting == 'binary':
                    if doc not in all_doc_vectors:
                        all_doc_vectors[doc] = {term: 1}
                        doc_vector_element_summations[doc]= 1**2
                    else: # In case the vector has already been initialised, just update its content.
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
            scores={}

            # Get all documents containing at least one query term.
            candidates=set()
            for word in query:
                if word in self.index:
                    candidates.update(self.index[word].keys())

            for i in candidates:
                #compute numerator
                numer = 0
                for word in query_vector.keys():
                    if word in self.all_document_vectors[i].keys():
                        numer+=query_vector[word] * self.all_document_vectors[i][word]
                
                denom = self.all_document_vec_lengths[i]
                
                scores[i] = numer/denom

            sorted_score_values = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            sorted_scores=collections.OrderedDict(sorted_score_values)
            
            return list(sorted_scores.keys())[:10]
