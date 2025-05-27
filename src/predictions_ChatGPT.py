import pandas as pd
from openai import OpenAI
from scripts.config import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

EMB_COL = "embeddings_ingles"
TEXT_COL = "translation"

DEVELOPER_PROMPT = "You are an assistant tasked with classifying Reddit posts based on the sentiment they express towards a given Keyword. There are three possible classes: Positive (POS), Negative (NEG), or Neutral (NEU)."
CONTEXT_PROMPT_POS = "To give you more context, I will provide three lists containing the most similar texts to the post to be classified. Examples of positive posts: "
CONTEXT_PROMPT_NEG = "Examples of negative posts: "
CONTEXT_PROMPT_NEU = "Examples of neutral posts: "
CONTEXT_PROMPT = "Remember, return only the label: POS, NEG, or NEU."
POST_PROMPT = "Classify this Reddit post as POS, NEG, or NEU based on its sentiment or polarity to this given keyword. "
POST_PROMPT1 = "Return only the label. "

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def load_dataset(path_to_dataset):
    return pd.read_csv(path_to_dataset)

def extract_similar_descriptions(e, embeddings, k):
    similarities = cosine_similarity([e], embeddings)[0]
    top_indices = np.argsort(similarities)[-k:][::-1]
    return top_indices
    
def compute_embeddings_distance(embedding_str, k, df):
    #df = load_dataset(path_to_dataset)

    # QUitamos del dataset la misma fila con la que estamos trabajando
    df = df[df["embeddings_ingles"]!= embedding_str]

    def parse_embedding(embedding_str):
        embedding_str = embedding_str.strip("[]")
        return np.array(embedding_str.split(), dtype=np.float32)
    
    embedding = parse_embedding(embedding_str)
    df["embeddings_ingles"] = df["embeddings_ingles"].apply(lambda x: parse_embedding(x))

    df_pos = df[df["label"]==2]
    df_neu = df[df["label"]==1]
    df_neg = df[df["label"]==0]

    pos_description_embeddings = np.vstack(df_pos["embeddings_ingles"].values)
    pos_nearest_descriptions = extract_similar_descriptions(embedding, pos_description_embeddings, k)
    pos_result = df_pos.iloc[pos_nearest_descriptions]["translation"]

    neu_description_embeddings = np.vstack(df_neu["embeddings_ingles"].values)
    neu_nearest_descriptions = extract_similar_descriptions(embedding, neu_description_embeddings, k)
    neu_result = df_neu.iloc[neu_nearest_descriptions]["translation"]

    neg_description_embeddings = np.vstack(df_neg["embeddings_ingles"].values)
    neg_nearest_descriptions = extract_similar_descriptions(embedding, neg_description_embeddings, k)
    neg_result = df_neg.iloc[neg_nearest_descriptions]["translation"]
    
    pos_result, neu_result, neg_result = pos_result.to_list(), neu_result.to_list(), neg_result.to_list()

    pos_str = ""
    for i in range(len(pos_result)):
        pos_str += str(i+1) + ". TEXT:" + pos_result[i] + " LABEL: POS\n"

    neu_str = ""
    for i in range(len(neu_result)):
        neu_str += str(i+1) + ". TEXT:" + neu_result[i] + " LABEL: NEU\n"

    neg_str = ""
    for i in range(len(neg_result)):
        neg_str += str(i+1) + ". TEXT:" + neg_result[i] + " LABEL: NEG\n"

    return pos_str, neu_str, neg_str

mapping = {"NEG":0, "NEU":1, "POS":2}
reverse_mapping = {0:"NEG", 1: "NEU", 2:"POS"}

def label2int(label):
    return mapping[label]

data = load_dataset("../data/data.csv")
train_df, test_df = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED, stratify=data['label'])


def initialize_client(key):

  client = OpenAI(
    api_key = key
  )

  return client

def predict_val_dataset(post, keyword, pos, neg, neu, client):

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
      {"role": "developer", "content": DEVELOPER_PROMPT},
      {"role" : "user", "content": f"{POST_PROMPT} Keyword: '{keyword}' Post to classify: '{post}'. {POST_PROMPT1} {CONTEXT_PROMPT_POS} {pos}. {CONTEXT_PROMPT_NEG} {neg}. {CONTEXT_PROMPT_NEU} {neu}. {CONTEXT_PROMPT}"}
    ]
  )

  return completion.choices[0].message.content

client = initialize_client(OPENAI_API_KEY)


# task_1 = load_dataset("../data/translated_dataset_task1_chatgpt.csv")

# preds_task1 = []
# for i, emb in enumerate(task_1[EMB_COL]):
#     post = task_1[TEXT_COL][i]
#     keyword = task_1["keyword"][i]
#     pos, neu, neg = compute_embeddings_distance(emb, 2, train_df)
#     response = predict_val_dataset(post, keyword, pos, neg, neu, client)
#     print(i, response)
#     preds_task1.append(response)

# task_1["preds"] = preds_task1
# task_1.to_csv("../data/pred_dataset_task1_chatgpt.csv")

print("Done task_1")

task_2 = load_dataset("../data/translated_dataset_task2_chatgpt.csv")

preds_task2 = []
for i, emb in enumerate(task_2[EMB_COL]):
    post = task_2[TEXT_COL][i]
    keyword = task_2["keyword"][i]
    pos, neu, neg = compute_embeddings_distance(emb, 2, train_df)
    response = predict_val_dataset(post, keyword, pos, neg, neu, client)
    print(i, response)
    preds_task2.append(response)
    
task_2["preds"] = preds_task2
task_2.to_csv("../data/pred_dataset_task2_chatgpt.csv")

print("Done task_2")
