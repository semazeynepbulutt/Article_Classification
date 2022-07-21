import streamlit as st
import data_cleaning
import logging
from PIL import Image
import os
import get_data
import requests

logging.getLogger('googleapicliet.discovery_cache').setLevel(logging.ERROR)

option = st.sidebar.selectbox(
     'How would you like to be contacted?',
     ('Multinominal Naive Bayes', 'SVM', 'XGBoost'))

st.write('You selected:', option)

articleSection_target_dict = {0: '3. Sayfa', 1: 'Dunya', 2: 'Ekonomi', 3: 'Futbol', 4: 'Kitap', 5: 'Magazin', 6: 'Otomobil', 7: 'Saglik', 8: 'Teknoloji'}

def tr2engchar(row:str) -> str:
    return row.replace('ı','i').replace('ü','u').replace('ö','o').replace('ş','s').replace('ç','c').replace('ğ','g').replace('İ','I').replace('Ü','U').replace('Ö','O').replace('Ş','S').replace('Ç','C')

def load_image(image_file):
	img = Image.open(image_file)
	return img

st.markdown("# Article Classification 🎈")
st.sidebar.markdown("# Article Classification 🎈")
headline_txt = st.text_input('Article Headline',)
content_txt = st.text_area('Article Content:', )
if st.button('Predict'):
    st.write('Prediction began')
tr2eng_content = tr2engchar(content_txt)
clean_content = data_cleaning.cleaning_text(tr2eng_content)
url = f'http://127.0.0.1:5005/article_info/{option}/article_body'
response = requests.post(url,headers={'key':'Content-Type','value':'application/json'},json={'article_body':clean_content})
json_data = response.json()
pred_name = json_data['prediction']
st.write(pred_name)
image_file = [filename for filename in os.listdir('words_images') if filename.startswith(pred_name)][0]
st.image(load_image('words_images/'+image_file), width=1000)

keywords = st.text_input('Article Headline #Please write with comma',)
if st.button('Save'):
    st.write('Saving began')
tr2eng_headline = tr2engchar(headline_txt)
tr2eng_keywords = tr2engchar(keywords)
db = get_data.get_database()
article_info_col = db.get_collection("article_info")
if len(tr2eng_keywords) > 0 and len(pred_name) > 0 :
    cur = article_info_col.find_one({'headline':tr2eng_headline})
    if cur is None:
        data = {'headline':tr2eng_headline,'articleSection':pred_name,'articleBody':tr2eng_content,'keywords':tr2eng_keywords,'URL':''}
        get_data.insert_one_data_to_database(db=db,data=data)


