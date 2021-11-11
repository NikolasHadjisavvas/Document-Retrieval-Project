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
        if self.term_weighting=='tfidf': #If term weighting is tfidf, precompute the idfs.
            self.idfs = self.compute_idf_for_all_terms()
        self.all_document_vectors,self.all_document_vec_lengths=self.construct_all_document_vectors() #Compute the vectors and lengths for usage in for_query.

    def compute_number_of_documents(self):
        self.doc_ids = set()
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)
        
        
    def compute_idf_for_all_terms(self):
        idfs = {}
        for term in self.ALL_TERMS:
            idfs[term] = math.log10(self.num_docs/self.compute_doc_freq(term))
        return idfs

    #Compute term weighting according to the selected scheme, at the same time build the document vectors and compute their length.
    def construct_all_document_vectors(self):
        all_doc_vectors={} # 2 level dictonary for document vectors of the form {docID:{term:termWeighting}}
        doc_vector_element_summations={} #Initialise dict with the element wise summation of a vector.{doc vector->sum}.We use that later to apply sqrt and get the lenghts.
 
        for term in self.ALL_TERMS: # Iterate over all terms in index
            for doc in self.index[term]: # Now go through each docs the term appears in, and step by stem(each term iteration) build up on the document vectors.
                if self.term_weighting == 'tfidf': # If tfidf is selected 
                    idf = self.idfs[term] # Get the corresponding precomputed idf 
                    if doc not in all_doc_vectors: # In case it's the first time a vector id is encountered, initilasize the dictionaries.
                        tf=self.index[term][doc] # Get the tf by indexing the index
                        all_doc_vectors[doc] = {term: tf*idf} # Store the weighting(vector element) in the dict.
                        doc_vector_element_summations[doc]= (tf * idf)**2
                    else: # In case the vector has already been initialised, just update its content.
                        tf=self.index[term][doc]
                        all_doc_vectors[doc][term] = tf * idf
                        doc_vector_element_summations[doc]+= (tf * idf)**2
                if self.term_weighting == 'tf': # Same as above but for tf
                    if doc not in all_doc_vectors:
                        tf=self.index[term][doc]
                        all_doc_vectors[doc] = {term: tf}
                        doc_vector_element_summations[doc]= tf**2
                    else: # In case the vector has already been initialised, just update its content.
                        tf=self.index[term][doc]
                        all_doc_vectors[doc][term] = tf
                        doc_vector_element_summations[doc]+= tf**2
                if self.term_weighting == 'binary': # Same as above but for binary
                    if doc not in all_doc_vectors:
                        all_doc_vectors[doc] = {term: 1}
                        doc_vector_element_summations[doc]= 1**2
                    else: # In case the vector has already been initialised, just update its content.
                        all_doc_vectors[doc][term] = 1
                        doc_vector_element_summations[doc]+= 1**2
            
        doc_vector_lengths={} #Initialise a dictionary {doc vector->doc length} to store vector lenghts.

        # Compute sqrt of doc_vector_element_summations to get lengths.
        for key,value in doc_vector_element_summations.items():
            doc_vector_lengths[key] = math.sqrt(value)

        return all_doc_vectors,doc_vector_lengths

    

    # Method for returning the document frequency of a term/word specified as a parameter.
    def compute_doc_freq(self, word):
        dfw = len(self.index[word])
        return dfw

    # Takes a query as a list of strings, returns tfidfs for each term.
    def compute_tfidfs_for_query_terms(self, query):
        tfidfs_of_query={} # Initialise dict {query term->tfidf}
        for word in query:
            tf = query.count(word) 
            if word in self.idfs: #If term is present in the document collection.Get the pre computed idf, otherwise the weighting is 0 and not stored.
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
            scores={} #Dicironary of the form {docID:similarity score}

            # Get all documents containing at least one query term.
            candidates=set()
            for word in query:
                if word in self.index:
                    candidates.update(self.index[word].keys())

            for i in candidates:
                #compute numerator of cos similarity(element wise multiplication of common terms in query AND candidate document).
                numer = 0 # Initialise the summation
                for word in query_vector.keys(): # So that we only consider terms in query.
                    if word in self.all_document_vectors[i].keys(): # If word is in the document add the product of the weightings to the summation variable.
                        numer+=query_vector[word] * self.all_document_vectors[i][word]
                
                denom = self.all_document_vec_lengths[i] # Denominator is simply the precomputed document vector length.
                
                scores[i] = numer/denom # Compute similarity and store in scores dictionary

            # Sorting the score values descending order using the sorted method.
            sorted_score_values = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            sorted_scores=collections.OrderedDict(sorted_score_values)
            
            return list(sorted_scores.keys())[:10]
