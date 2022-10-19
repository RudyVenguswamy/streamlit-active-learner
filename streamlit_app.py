#do any work here 

import streamlit as st 

import pandas as pd
import numpy as np
import time
import gdown
from tqdm import tqdm
tqdm.pandas()
import time
import json
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import spacy
from nltk.tokenize import RegexpTokenizer

import gensim.downloader

#######################################################################
def model_training(df: pd.DataFrame)-> pd.DataFrame:
  '''
  input: df: pd.DataFrame containing all unlabeled and labeled text, is same df as task_df from what_to_label below. See what_to_label method below for more details
  output: predictions: pd.DataFrame with same indices as task_df of predicted positive class score
  '''
  #train model
  labeled_df = df[~df['Labels'].isna()]
  unlabeled_df = df[df['Labels'].isna()]
  x = labeled_df['Input'].values
  y = labeled_df['Labels'].values
  
  vectorizer = CountVectorizer(max_features=1500) 

  x_v = vectorizer.fit_transform(x).toarray()

  model = RandomForestClassifier()
  model.fit(x_v, y)

  #predict on unlabeled data
  x_test = unlabeled_df['Input'].values
  x_test_v = vectorizer.transform(x_test).toarray()

  pred = model.predict_proba(x_test_v)
  pred_positive_class = pred[:,1]

  return pd.DataFrame(pred_positive_class, columns=['probability'], index = unlabeled_df.index)

def pick_uncertain(prediction_df: pd.DataFrame, n_label: int)-> list:
  '''
  input: pd.DataFrame of predictions with index corresponding to task_df
  output: list of indices corresponding to strategy of most uncertain
  '''
  return abs(prediction_df['probability'] - 0.5).sort_values()[:n_label].index.tolist()

def what_to_label(task_df: pd.DataFrame, n_label: int)-> list:
  '''
  input: pd dataframe of two text columns: 'Input' and 'Labels', n_label: num to sample in next label session
  output: list of indices corresponding to what to label
  '''
  #current: simple random forest ml classifier
  unlabeled = task_df[task_df['Labels'].isna()]
  labeled = task_df[~task_df['Labels'].isna()]

  #if num to label greater than num available to label
  if n_label > len(unlabeled):
    return unlabeled.index.tolist()

  #if no labels yet, pick random ones
  elif len(labeled) == 0:
    return unlabeled.sample(n_label).index.tolist()
  
  #if only one class labeled so far
  elif fetch_dataset()['Labels'].nunique() < 2:
    return unlabeled.sample(n_label).index.tolist()

  else:
    with st.spinner('Training machine learning model...'):
      predictions = model_training(task_df)

    idxs = pick_uncertain(predictions, n_label)
    st.success('Ready for human labeling!')
    return idxs
  
def label_session():
  '''
  note: secret input/output passed via session state
  input: to_label: np array of text items to label
  output: session_labels: list of labels corresponding to to_label
  '''

  #set up counter
  if 'seen_count' not in st.session_state:
    st.session_state.seen_count = 0 
    st.session_state.positive_count = 0
    st.session_state.negative_count = 0

  #set up progress bar
  progress_bar = st.progress(st.session_state.seen_count/len(st.session_state.to_label))

  #after setting up methods, we first check if we've labeled everything
  if st.session_state.seen_count == len(st.session_state.to_label):
    #if done with session, wrap it up, add labels to dict, and remove session state
    st.session_state.done_with_labeling_session = True
    st.session_state.session_labels = dict(zip(st.session_state.to_label, st.session_state.session_labels))
    st.session_state.dataset_labels.update(st.session_state.session_labels)
    purge_label_session_states()
    st.button('Proceed to next step') #just here to get a refresh
    return

  #if we're not done labeling, let's proceed with getting labels
  #set up store of results if beginning of label session
  if 'session_labels' not in st.session_state:
    st.session_state.session_labels = []  

  def label_positive():
    try: #catches spam clicks
      st.session_state.positive_count += 1
      st.session_state.seen_count +=1
      st.session_state.session_labels.append('Positive')
    except:
      return

  def label_negative():
    try: #catches spam clicks
      st.session_state.negative_count += 1
      st.session_state.seen_count +=1
      st.session_state.session_labels.append('Negative')
    except:
      return

  #labeling layout and display
  st.text(st.session_state.to_label[st.session_state.seen_count])

  #set up buttons and location
  neg_col, pos_col = st.columns([1,1])

  with pos_col:
      positive = st.button("Positive", on_click = label_positive)
  with neg_col:
      negative = st.button("Negative", on_click = label_negative)

  #print labels
  st.text(f"Positive Count: {st.session_state.positive_count}")
  st.text(f"Negative Count: {st.session_state.negative_count}")

def purge_label_session_states():
  for item in ['seen_count', 'positive_count', 'negative_count', 'session_labels', 'to_label']:
    del st.session_state[item]
  return

def get_predicted_probs(df):
  #train model
  labeled_df = df[~df['Labels'].isna()]
  x = labeled_df['Input'].values
  y = labeled_df['Labels'].values
  
  vectorizer = CountVectorizer(max_features=1500) 

  x_v = vectorizer.fit_transform(x).toarray()

  model = RandomForestClassifier()
  model.fit(x_v, y)

  #predict on all data
  x_test = df['Input'].values
  x_test_v = vectorizer.transform(x_test).toarray()

  pred = model.predict_proba(x_test_v)
  pred_positive_class = pred[:,1]

  return pd.DataFrame(pred_positive_class, columns=['probability'], index = df.index)

###################################################

#data getting section
@st.cache(persist=True)
def get_data():
  df = pd.read_csv('/content/small_abstract.csv')
  return np.array(df['small_abstract'])
  
def is_in(abstract, search_words):
  tokenizer = RegexpTokenizer(r'\w+')
  return any(search_word.lower() in tokenizer.tokenize(abstract.lower()) for search_word in search_words)

@st.cache(persist=True)
def filter_dataset(arr, filter_words):
  if len(filter_words) == 0:
    return arr
  else:
    to_include = list(map(lambda abstract: is_in(abstract, filter_words), arr))
    return arr[to_include]

#cannot cache this since we rely on human labels being updated
def fetch_dataset():
  input = get_data() #cached
  input = filter_dataset(input, st.session_state.filter_words) #cached
  dataset = pd.DataFrame(input, columns = ['Input'])
  dataset['Labels'] = dataset['Input'].map(st.session_state.dataset_labels)
  return dataset

def add_filter_word(word):
  tokenizer = RegexpTokenizer(r'\w+')
  r = tokenizer.tokenize(word.lower())
  if len(r) > 1:
    st.warning('Please enter one word at a time')
  elif len(r) == 0:
    st.warning('No word specified!')
  else:
    st.session_state.filter_words.append(r[0])

def add_filter_word_wrapper(word):
  def add_filter_word():
    st.session_state.filter_words.append(word)
    #stops the double click issue of streamlit
    st.session_state.filter_words = list(set(st.session_state.filter_words))

  return add_filter_word

def remove_filter_word_wrapper(word):
  def remove_filter_word():
    if word in st.session_state.filter_words:
      st.session_state.filter_words.remove(word)
  return remove_filter_word

def prepare_label_session():
  #fetch next portion of dataset to label
  dataset = fetch_dataset()
  st.session_state.to_label = list(dataset.loc[what_to_label(dataset, n_label = 50)]['Input'])
  if len(st.session_state.to_label) == 0:
    st.warning("You have finished labeling the entire dataset which matches your search criteria. There's nothing left to label! You can try adding in more keywords to expand your labeling pool.")
  else:
    st.session_state.done_with_labeling_session = False
  return

def get_vectors():
    glove_vectors = gensim.downloader.load('glove-twitter-25')
    return glove_vectors

###################################################

if __name__ == '__main__':
  st.markdown(""" Active Learning Labeler App """)
  if 'glove_vectors' not in st.session_state:
    with st.spinner('Retrieving NLP-Gensim-Glove Model...'):
      st.session_state.glove_vectors = get_vectors()
  
  label_space, filter_space = st.columns([3,1])

  #init empty list of labels for first time
  if 'dataset_labels' not in st.session_state:
    st.session_state.dataset_labels = {}
    
  #init for first time word filter
  if 'corpus_filtered' not in st.session_state:
    st.session_state.corpus_filtered = True
    st.session_state.filter_words = []

  #init label session for first time
  if 'done_with_labeling_session' not in st.session_state:
    st.session_state.done_with_labeling_session = True

  if 'pred_num_positive' not in st.session_state:
    st.session_state.pred_num_positive = 'Label more for estimate!'

  #get words to filter dataset on right tab
  with st.sidebar:
    if st.session_state.done_with_labeling_session:
      st.button("Start Labeling", on_click = prepare_label_session)
      text = st.text_input(label = 'Enter Search Word', key='search_word_input', on_change = lambda: add_filter_word(st.session_state.search_word_input))
      
      for w in st.session_state.filter_words:
        fn = remove_filter_word_wrapper(w)
        st.button(label = f"{w} |  x ", on_click = fn)

      #similar words
      if len(st.session_state.filter_words) > 0:
        st.markdown('You might consider adding these words to your search:')
        suggestions = st.session_state.glove_vectors.most_similar(positive=st.session_state.filter_words)
        suggestions = [i[0] for i in suggestions]
        for w in suggestions:
          fn = add_filter_word_wrapper(w)
          st.button(label = f"{w}", on_click = fn)
    
      else:
        st.text('Add a few search words to see suggestions')

  #left tab
  if not st.session_state.done_with_labeling_session:
      label_session()
  else:
    with label_space:
      filtered_dataset = fetch_dataset()

      #just to create positive class estimate
      with st.spinner('Compiling Results...'):
        input = get_data() #cached
        dataset = pd.DataFrame(input, columns = ['Input'])
        dataset['Labels'] = dataset['Input'].map(st.session_state.dataset_labels)

        if dataset['Labels'].nunique() > 1:
          probs = get_predicted_probs(dataset)
          filtered_dataset['Predicted Prob.'] = probs.loc[filtered_dataset.index]['probability']
          st.session_state.pred_num_positive = len(probs[probs['probability'] > 0.5])

      st.text('Predicted number of positive examples in dataset:')
      st.text(st.session_state.pred_num_positive)
      st.table(filtered_dataset)
 
