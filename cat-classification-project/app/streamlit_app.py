from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'models'/'cat_breeds.keras'
LABELS=ROOT/'models'/'labels.json'
st.set_page_config(page_title='Cat Breed Classifier',page_icon='')
st.title(' Cat Breed Classification')
st.write('Upload a cat photo to predict its breed from the breeds included in training.')
if not MODEL.exists() or not LABELS.exists():
    st.info('First add breed images to data/<BreedName>/ and run: python src/train.py')
    st.stop()
@st.cache_resource
def load():
    return tf.keras.models.load_model(MODEL),json.loads(LABELS.read_text(encoding='utf-8'))
model,labels=load()
upload=st.file_uploader('Upload a cat image',type=['jpg','jpeg','png'])
if upload:
    photo=Image.open(upload).convert('RGB')
    st.image(photo,caption='Your image',use_container_width=True)
    size=model.input_shape[1:3]
    x=np.asarray(photo.resize((size[1],size[0])),dtype=np.float32)[None,...]
    scores=model.predict(x,verbose=0)[0]
    best=int(np.argmax(scores))
    st.subheader(f'Predicted breed: {labels[best]}')
    st.metric('Model score',f'{scores[best]:.1%}')
    st.write('Top predictions:')
    for i in np.argsort(scores)[::-1][:min(3,len(labels))]:
        st.write(f'{labels[int(i)]}: {scores[int(i)]:.1%}')
        st.progress(float(scores[int(i)]))
    st.caption('Scores are not calibrated probabilities. The model cannot reliably recognize breeds absent from training or detect non-cat images.')
