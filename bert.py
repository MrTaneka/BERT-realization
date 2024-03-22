import torch
from transformers import BertTokenizer , BertModel
from sklearn . metrics . pairwise import cosine_similarity
import numpy as np

model_name = " bert - base - uncased "
tokenizer = BertTokenizer . from_pretrained ( model_name )
model = BertModel . from_pretrained ( model_name )

document1 = " This is the first document . "
document2 = " Here is the first document . "
document3 = " A third document is also present . "

documents = [ document1 , document2 , document3 ]
encoded_documents = []
for doc in documents :
tokens = tokenizer ( doc , return_tensors = " pt " , padding = True , truncation = True )
 with torch . no_grad () :
  outputs = model (** tokens )
 embeddings = outputs . last_hidden_state . mean ( dim =1)
 encoded_documents . append ( embeddings . numpy () )

 encoded_documents = np . vstack ( encoded_documents )
 similarities = cosine_similarity ( encoded_documents )

 print ( " Similarity Matrix : " )
 print ( similarities )

 for i , doc in enumerate ( documents ) :

 most_similar_index = similarities [ i ]. argsort () [ -2]
 mos t_simi lar_do cumen t = documents [ most_similar_index ]
 similarity_score = similarities [ i ][ most_similar_index ]
 print ( f " \ nDocument { i +1}: { doc }\ nMost similar document : {
mos t_simi lar_d ocumen t }\ nSimilarity Score : { similarity_score
} " )
