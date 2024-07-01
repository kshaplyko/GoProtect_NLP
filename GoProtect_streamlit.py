#!/usr/bin/env python
# coding: utf-8

# Streamlit приложение
import streamlit as st

# Функция
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nlpaug.augmenter.char as nac

def process_data(df_file, reference_file):
    # Загрузка данных
    with st.spinner('Загрузка данных...'):
        df = pd.read_csv(df_file)
        reference = pd.read_csv(reference_file)

    # Создам столбец с именем и регионом в датасете reference
    reference['title'] = reference['name'] + ' ' + reference['region']

    # Уберу лишнии символы - пробелы, кавычки и тд. Для этого создам функцию
    def del_symbols(dataframe):
        for i in dataframe.columns:
            dataframe[i] = (dataframe[i].astype(str)
                  .replace(r'[^А-Яа-яёЁA-Za-z0-9\s]', ' ', regex=True)
                  .replace(r'\s+',' ',regex=True)
                  .str.strip())
        
    # Удаление лишних символов в датасете
    del_symbols(df)
    del_symbols(reference)

    # Удаление дубликатов в датасете
    reference = reference.drop_duplicates('title')
    df = df.drop_duplicates()

    # Загрузка модуля аугментации
    aug = nac.RandomCharAug()

    # Аугментация
    reference['aug'] = reference['title'].apply(lambda x: aug.augment(x, n=10))

    # Оставлю только оригинальное название и аугментированные данные, разделю их и добавлю как новые строки
    reference = reference.explode('aug')[['school_id','aug','title']].reset_index(drop=True)

    # Загрузка модели и векторизация
    model_name = 'sentence-transformers/LaBSE'
    model = SentenceTransformer(model_name)
    corpus = model.encode(reference['title'].values)
    queries = model.encode(reference['aug'].values)

    # Поиск утилитой semantic_search
    search_result = util.semantic_search(queries, corpus, top_k=5)

    # Вытащу найденный id и добавлю его в исходный датасет для сравнения
    reference['candidate_idx'] = [x[0]['corpus_id'] for x in search_result]
    reference['candidate_name'] = reference.title.values[reference.candidate_idx.values]

    # Посчитаю точность модели
    train_accuracy = (reference['title'] == reference['candidate_name']).sum() / len(reference)
    st.success('Загрузка данных завершена.')

    # Проверю на тестовых данных - датасете df
    with st.spinner('Обработка данных...'):
        queries_test = model.encode(df['name'].values)
        search_test = util.semantic_search(queries_test, corpus, top_k=5)
                                                                                      
        df['candidate_idx'] = [x[0]['corpus_id'] for x in search_test]
        df['candidate_name'] = reference.title.values[df.candidate_idx.values]
        reference = reference.drop(['aug', 'candidate_idx', 'candidate_name'], axis=1).drop_duplicates()
        df = df.merge(reference, left_on='candidate_name', right_on='title', how='left')

    # Удаление лишних столбцов
    df = df.drop(['candidate_idx', 'candidate_name'], axis=1)
    df.columns = ['school_id','name','predicted_school_id','predicted_title']
    st.success('Обработка данных завершена.')                                                                                  
                                                                                     
    # Посчитаю точность модели
    test_accuracy = (df['school_id'] == df['predicted_school_id']).sum() / len(df)
    
    # Сохранение обработанных данных
    df_processed = df.copy()  # Здесь вы можете задать путь и имя файла для сохранения
    return model_name,train_accuracy, test_accuracy, df_processed, reference  # Возвращаем df_processed и reference

# Код для streamlit
def main():
    st.title('Распознавание названий спортивных школ')

    df_file = st.file_uploader('Загрузите файл с примерным написанием', type='csv')
    reference_file = st.file_uploader('Загрузите файл с эталонными названиями', type='csv')

    if df_file and reference_file:
        model_name, train_accuracy, test_accuracy, df_processed, processed_reference = process_data(df_file, reference_file)
        
        st.write('Модель векторизации: ', model_name)
        st.write('Поиск совпадений с помощью: semantic_search')
        st.write('Accuracy на тренировочных данных: ', train_accuracy)
        st.write('Accuracy на тестовых данных: ', test_accuracy)
        
        st.write('Обработанный датафрейм:')
        st.dataframe(df_processed.head())

        output_df_file = st.text_input('Введите путь для сохранения файла', 'df_processed.csv')

        if st.button('Save Processed Files'):
            output_dir = os.path.dirname(output_df_file)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            with st.spinner('Сохранение файла...'):
                df_processed.to_csv(output_df_file, index=False)
            st.success('Файл успешно сохранен!')

if __name__ == '__main__':
    main()
