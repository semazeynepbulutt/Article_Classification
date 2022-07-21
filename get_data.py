from typing import List
from typing import Tuple
from typing import Dict
from googleapiclient.discovery import build
from google.oauth2 import service_account
from bs4 import BeautifulSoup
import json
import pandas as pd
import requests
import re

SPREADSHEET_ID = '198xSnCkdUeHNvoytoUzZORVA8aJpcVDgSHNuGp1WWUg'
scope = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'creds.json'
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
service = build('sheets', 'v4', credentials=creds,cache_discovery=False)
sheet = service.spreadsheets()

def remove_control_chart(s):
    return re.sub(r'\\x..', '', s)

def get_all_links() -> List[str]:
    global SPREADSHEET_ID
    global sheet
    result = sheet.values().get(spreadsheetId=SPREADSHEET_ID,range='Sheet1!A2:A18409').execute()
    values = result.get('values', [])
    return values

def get_article_content(url:str) -> Tuple[str,str,str,str]:
    try:
        soup = BeautifulSoup(requests.get(url).content, "html.parser")
        data = [json.loads(x.string) for x in soup.find_all("script", type="application/ld+json")]
        return data[0]['headline'], data[0]['articleSection'], remove_control_chart(data[0]['articleBody']), ','.join(map(str, data[0]['keywords']))
    except:
        return '0','0','0','0'

def tr2engchar(row:str) -> str:
    return row.replace('ı','i').replace('ü','u').replace('ö','o').replace('ş','s').replace('ç','c').replace('ğ','g').replace('İ','I').replace('Ü','U').replace('Ö','O').replace('Ş','S').replace('Ç','C')

def create_data(url_list:List[str]) -> List[Dict]:
    df = pd.DataFrame(columns=["headline", "articleSection", "articleBody", "keywords"])
    df['URL'] = url_list
    df['URL'] = df['URL'].apply(lambda x:' '.join(map(str, x)))
    df['headline'], df['articleSection'], df['articleBody'], df['keywords'] = zip(*df['URL'].map(get_article_content))
    df['headline'] = df['headline'].apply(lambda x:tr2engchar(x))
    df['articleSection'] = df['articleSection'].apply(lambda x: tr2engchar(x))
    df['articleBody'] = df['articleBody'].apply(lambda x: tr2engchar(x))
    df['keywords'] = df['keywords'].apply(lambda x: tr2engchar(x))
    #df['articleBody'] = df['articleBody'].apply(lambda x: unicodedata.normalize("NFKD", x) if u'\xa0' in x else x)
    df_dict = df.to_dict('records')
    return df_dict

def get_database():
    import pymongo

    client = pymongo.MongoClient(
        "mongodb+srv://dbUser:db123456@cluster0.ytykqev.mongodb.net/?retryWrites=true&w=majority")

    mydb = client["mydatabase"]  # you can also use dot notation client.mydatabase
    return mydb

def insert_many_data_to_database(db,data : List[Dict]):
    db.article_info.insert_many(data) #article_info is the name of the collection

def insert_one_data_to_database(db,data:dict):
    db.article_info.insert_one(data)
if __name__ == '__main__':
    mydb = get_database()
    all_data_dict = create_data(get_all_links())
    insert_many_data_to_database(db=mydb,data=all_data_dict)




