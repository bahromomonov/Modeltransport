from fastai.vision.all import *
import streamlit as st
import pathlib
import posixpath
temp=pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath
st.title('Transportlarni klassifikatsiyalovchi model')
file=st.file_uploader("Rasm yuklash:,type=['jpeg','jpg','svg','gif']")
if file:
    st.image(file)
    img=PILImage.create(file)
    model=load_learner('transport_model.pkl')
    pred,pred_id,probs=model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:1f}")