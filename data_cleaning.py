import pandas as pd
import seaborn as sns
import get_data
import matplotlib.pyplot as plt
import re
from wordcloud import WordCloud
from collections import Counter

pd.options.mode.chained_assignment = None

def get_data_from_database() -> pd.DataFrame:
    db = get_data.get_database()
    article_info_col = db.get_collection("article_info")
    data_result = article_info_col.find({}) #"articleSection" : { "$in" : [ 'Eğitim', 'Ekonomi','Kültür Sanat','Politika','Teknoloji'] }
    df = pd.DataFrame(columns=["headline", "articleSection", "articleBody", "keywords"])
    for article in data_result:
        a = {"headline":article["headline"],"articleSection":article["articleSection"],"articleBody":article["articleBody"],"keywords":article["keywords"]}
        #df = df.append(a,ignore_index=True)
        df = pd.concat([df, pd.DataFrame.from_records([a])])
    visualize_value_counts(dataframe_name=df,columns_name='articleSection')

    return df

def visualize_value_counts(dataframe_name : pd.DataFrame,columns_name:str):
    columns_count = dataframe_name[columns_name].value_counts()[:15]
    plt.figure(figsize=(20, 10))
    sns.barplot(columns_count.index, columns_count.values, alpha=0.8)
    plt.title(f'Value Counts of {columns_name} in article_info collection')
    plt.ylabel('Number of Rows', fontsize=12)
    plt.xlabel(f'{columns_name}', fontsize=12)
    name = 'value_counts_of'+columns_name+'.png'
    plt.savefig(name)


def filtering_and_combining_article_section(df : pd.DataFrame) -> pd.DataFrame :
    columns_list = list(df['articleSection'].value_counts().index)[:15]
    df = df[df['articleSection'].isin(columns_list)]
    df['articleSection'] = df['articleSection'].str.replace('Transfer dosyasi','Futbol',regex=True)
    df['articleSection'] = df['articleSection'].str.replace('Dunyadan futbol', 'Futbol', regex=True)
    df['articleSection'] = df['articleSection'].str.replace('Medya', 'Magazin', regex=True)
    df = df[~df['articleSection'].isin(['Ic Haber','Gundem','Yasam'])]
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def remove_numbers(row : str) -> str:
    return re.sub('[0-9]+', ' ', row)

def remove_punc(row : str) -> str:
    return re.sub(r'[^\w\s]',' ',row)

def remove_whitespace(row : str) -> str:
    return re.sub("\s+", " " ,row.strip())

def remove_short_words(row : str) -> str:
    return ' '.join([x for x in row.split() if len(x)>2])

def insert_space(word : str, index : int) -> str:
    return word[0:index] + ' ' + word[index:]

def replace_unified_words(word : str) -> str:
    try:
        return insert_space(word,word.index(re.search('[a-z][A-Z]', word).group(0)[1]))
    except:
        return word

def get_stop_words_list():
    with open('tr_stopwords.txt') as f:
        words_data = f.read()
        stopwords_list = remove_short_words(words_data).split()
    another_stopwords = ['mi','mu','ki','dedi','soyledi','ediyor','etmek','edecek','devam','ilce','ilcesi','ilcesinde','mahalle','mahallesi','mahallesinde','kadin','erkek','yeni','son','iyi','haber','haberi','haberine göre','haberine','nedeniyle','ilk']
    stopwords_list.extend(another_stopwords)

    return stopwords_list

def cleaning_text(text : str) -> str:
    punc_removed = remove_punc(text)
    replaced_unified = ' '.join([replace_unified_words(s)for s in punc_removed.split(' ')])
    lower_text = replaced_unified.lower()
    number_removed = remove_numbers(lower_text)
    whitespace_removed = remove_whitespace(number_removed)
    short_words_removed = remove_short_words(whitespace_removed)
    stop_words_list = get_stop_words_list()
    stop_words_removed = ' '.join([w if w not in stop_words_list else '' for w in short_words_removed.split(' ')])
    result = remove_whitespace(stop_words_removed)
    return result

def visualize_articleSection_common_words(df: pd.DataFrame, section_name: str):
    all_txt = ' '.join(list(df['articleBody'][df['articleSection']==section_name].values))
    wordcloud = WordCloud(width = 800, height = 800,
                background_color ='white',
                stopwords = None,
                min_font_size = 10).generate(all_txt)
    plt.figure(figsize = (8, 8), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)
    png_name = section_name +".png"
    plt.savefig(png_name)

def creat_worldcloud_for_every_section(df:pd.DataFrame):

    section_list = list(df['articleSection'].unique())
    for s in section_list:
        visualize_articleSection_common_words(df=df,section_name=s)

def create_table_for_most_common_words_in_every_section(df:pd.DataFrame) -> pd.DataFrame:
    section_list = list(df['articleSection'].unique())
    most_common_words = pd.DataFrame()
    unique_words_count = pd.DataFrame(columns=['articlSection', 'numberOfUniqueWords'])
    for s in section_list:
        c = Counter(list(''.join(list(df['articleBody'][df['articleSection'] == s].values)).split(" ")))
        top15worddict = dict(sorted(c.items(), key=lambda item: item[1], reverse=True)[:15])
        key_list = list(top15worddict.keys())
        values_list = list(top15worddict.values())
        column_name_for_key = s + '-' + 'top15-word'
        column_name_for_value = s + '-' + 'top15-value'
        most_common_words[column_name_for_key] = key_list
        most_common_words[column_name_for_value] = values_list
        unique_words = {'articlSection': s, 'numberOfUniqueWords': len(c.keys())}
        unique_words_count = pd.concat([unique_words_count, pd.DataFrame.from_records([unique_words])])
    return most_common_words, unique_words_count

def most_common_words_total_top_50(df:pd.DataFrame) -> pd.DataFrame:
    c = Counter(list(''.join(list(df['articleBody'].values)).split(" ")))
    top50word = sorted(c.items(), key=lambda item: item[1],reverse=True)[:50]
    words_total = pd.DataFrame(top50word,columns=['words','count'])
    return words_total

def get_clean_data() -> pd.DataFrame: #for training and testing
    df = get_data_from_database()
    df = filtering_and_combining_article_section(df=df)
    df['articleBody'] = df['articleBody'].apply(lambda x: cleaning_text(x))
    return df

if __name__ == '__main__':
    df = get_data_from_database()
    df = filtering_and_combining_article_section(df=df)
    df['articleBody'] = df['articleBody'].apply(lambda x: cleaning_text(x))
    df.to_excel('all_data_new.xlsx')
    creat_worldcloud_for_every_section(df=df)
    words_total = most_common_words_total_top_50(df=df)
    words_total.to_excel('total_top_50.xlsx')
    most_common_words, unique_words_count = create_table_for_most_common_words_in_every_section(df=df)
    most_common_words.to_excel('most_common_15words_in_every_section.xlsx')
    unique_words_count.to_excel('unique_words_count_in_every_section.xlsx')

    #print(cleaning_text(df.loc[0,'articleBody']))
    #df["articleBody"] = df["articleBody"].apply(lambda x: cleaning_text(x))