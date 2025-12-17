import streamlit as st
import requests

st.set_page_config(page_title="Research paper search engine", layout="centered")
st.markdown(
    "<h1 style='text-align: center;'>AI Based Research paper Search Engine</h1>",
    unsafe_allow_html=True,
)
text = st.text_input("Describe which research paper you want")

url = "http://localhost:8000/search"

if text:
    try:
        out = requests.get(url, params={"text": text}).json()
        # st.write(out)
        for i in range(len(out["id"])):
            x=str(out['id'][i])
            if len(str(out['id'][i]).split(".")[0])==3:
                x=f"0{out['id'][i]}"            
            st.markdown(f"[{out['title'][i]}](https://arxiv.org/pdf/{x}.pdf)")

    except Exception as E:
        print(E)
        st.write("An error occured . Please try again later")
