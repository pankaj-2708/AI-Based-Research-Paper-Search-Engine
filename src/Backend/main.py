from fastapi import FASTAPI,Response
from fastapi.response import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from util import get_File_from_s3,load_model_tokenizer,generateEmbed
import tempfile
import faiss
import pandas as pd
import torch,getTop5Results

app=FASTAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device='cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
model,tokenizer=load_model_tokenizer(device)
csv_file=pd.read_csv(get_File_from_s3())

with tempfile.NamedTemporaryFile(suffix=".faiss") as f:
    f.write(get_File_from_s3().read())
    f.flush()
    faiss_ind=faiss.read_index(f.name)


@app.get("/search")
def search(text:str):
    try:
        embed=generateEmbed(text,model,tokenizer,device)
        # it will return a dict with two keys Title and id
        top_5_results=getTop5Results(faiss_ind,csv_file,embed)
        return JSONResponse(top_5_results,status_code=200)
    except Exception as E:
        print(E)
        return Response(500)


if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0',port=8000)