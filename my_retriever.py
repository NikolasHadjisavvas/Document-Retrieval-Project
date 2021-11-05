import math
import json
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
        
        print(self.all_document_vec_lengths)


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

    #Compute term weighting accarding to the selected scheme, at the same time build the document vectors and compute their length.
    def construct_all_document_vectors(self):
        all_doc_vectors={}
        doc_vector_lengths={}

        for term in self.ALL_TERMS:
            for doc in range(1,self.num_docs+1): 
                if doc in self.index[term]:#if doc contains term
                    tf = self.index[term][doc] 
                    if self.term_weighting=='tfidf':
                        idf = self.idfs[term]
                        if doc not in all_doc_vectors:
                            all_doc_vectors[doc] = {term:tf * idf}
                            doc_vector_lengths[doc]= (tf * idf)**2
                        else:
                            all_doc_vectors[doc][term] = tf * idf
                            doc_vector_lengths[doc]+= (tf * idf)**2
                    elif self.term_weighting =='tf':
                        if doc not in all_doc_vectors:
                            all_doc_vectors[doc] = {term:tf}
                            doc_vector_lengths[doc]= tf**2
                        else:
                            all_doc_vectors[doc][term] = tf
                            doc_vector_lengths[doc]+= tf**2

                        
        doc_vector_lengths2={}

        for key,value in doc_vector_lengths.items():
            doc_vector_lengths2[key] = math.sqrt(value)

        return all_doc_vectors,doc_vector_lengths2

    
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

    #Need to confirm that this is correct
    def compute_tfidfs_for_query_terms(self, query):
        tfidfs_of_query={}
        for word in query:
            tf = query.count(word)
            if word in self.idfs:
                idf=self.idfs[word]
            else:
                idf = 0
            tfidfs_of_query[word] = tf*idf
        
        return tfidfs_of_query


    def construct_vector_for_query(self, query):

        if self.term_weighting=='tfidf':
            vector = self.compute_tfidfs_for_query_terms(query)
        if self.term_weighting=='tf':
            vector={}
            for word in query:
                vector[word]=query.count(word)

        return vector


    def compute_vector_length(self,vector):
        sum=0
        values=vector.values()
        for i in values:
            sum += i*i

        length = math.sqrt(sum)
    
        return length


    # Method performing retrieval for a single query (which is 
    # represented as a list of preprocessed terms).Returns list 
    # of doc ids for relevant docs (in rank order).
    def for_query(self, query):
            query_vector = self.construct_vector_for_query(query)
            doc_vectors= self.all_document_vectors

            scores={}

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
                
                denom = self.all_document_vec_lengths[i] #do this for all docs in construct_all_doc_vectors
                
                scores[i] = numer/denom

            sorted_score_values = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            sorted_scores=collections.OrderedDict(sorted_score_values)
            
            return list(sorted_scores.keys())[:10]

    """ #JUST FOR TESTING
        def forQuery(self, query):
            query_vector = self.construct_vector_for_query(query)
            doc_vectors= self.all_document_vectors


            scores={}

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
                        print('Mult. '+str(query_vector[word])+' of query with '+str(doc_vectors[i][word])+' of vector '+str(i), query_vector[word] * doc_vectors[i][word])
                
                
                denom = self.compute_vector_length(doc_vectors[i])
                
                scores[i] = numer/denom
                print('End for doc'+str(i)+' numerator is '+str(scores[i]))

            sorted_score_values = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            sorted_scores=collections.OrderedDict(sorted_score_values)
            
            return list(sorted_scores.keys())[:10]"""
