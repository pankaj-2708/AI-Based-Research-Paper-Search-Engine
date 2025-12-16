import boto3
from transformers import AutoTokenizer
import torch 
from adapters import AutoAdapterModel

s3 = boto3.client("s3")
# function to load csv file from s3
def get_File_from_s3(filename="",bucketname=""):
    return s3.get_object(Bucket=bucketname,Key=filename)['Body']

def load_model_tokenizer(device):
    ad_hoc=AutoAdapterModel.from_pretrained("allenai/specter2_base")
    ad_hoc.load_adapter("allenai/specter2_adhoc_query", source="hf", load_as="specter2_adhoc_query", set_active=True)
    tokenizer=AutoTokenizer.from_pretrained("allenai/specter2_base")
    return ad_hoc.to(device),tokenizer
    
def generateEmbed(text,model,tokenizer,device):
    input=tokenizer(text,return_tensors='pt',padding=True,truncation=True,return_token_type_ids=False,max_length=512)
    out=model(**{k:v.to(device) for k,v in input.items()}).last_hidden_state[:,0,:]
    return out
    
def getTop5Results(faiss_ind,df,embed):
    D,I=faiss_ind.search(embed.to('cpu').detach().numpy(), 5)
    return {"title":df.iloc[I[0],:]['title'].values,"id":df.iloc[I[0],:]['title'].values}