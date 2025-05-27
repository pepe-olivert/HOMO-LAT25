import pandas as pd
from openai import OpenAI
from config import *

paraph_num = 2

data_path = "../../data/"
# train_path = data_path + "train_paraphrased.csv"
# val_path = data_path + "val_paraphrased.csv"
# test_path = data_path + "test_paraphrased.csv"
# train_df = pd.read_csv(train_path)
# val_df = pd.read_csv(val_path)
# test_df = pd.read_csv(test_path)

dev_path = data_path + "dev_paraphrased.csv"
train_df = pd.read_csv(dev_path)

# Print shapes
# print("Train shape:", train_df.shape)
# print("Val shape:", val_df.shape)
# print("Test shape:", test_df.shape)

# print("Train columns:", train_df.columns)
# print("Val columns:", val_df.columns)
# print("Test columns:", test_df.columns)

if paraph_num == 1:
  DEVELOPER_PROMPT = "Eres un asistente responsable de parafrasear textos para realizar aumento de datos." \
                   "Recibirás un texto de una publicación de Reddit de un país latinoamericano." \
                   "También recibirás una palabra clave que está presente en el texto." \
                   "Debes parafrasear el texto tanto como sea posible, excepto la palabra clave, de modo que sirva como un nuevo ejemplo de entrenamiento." \
                   "Ten en cuenta que el texto es una publicación de Reddit y de un país latinoamericano a la hora de parafrasear. Usa lenguaje informal y latinoamericano." \
                   "Tu respuesta debe ser únicamente el nuevo texto, sin agregar absolutamente nada más."

if paraph_num == 2:
  DEVELOPER_PROMPT = "Eres un asistente responsable de parafrasear textos para realizar aumento de datos." \
                   "Recibirás dos textos, el primero es el texto original de una publicación de reddit, y el segundo es un texto parafraseado a partir del original." \
                   "También recibirás una palabra clave que está presente en el texto." \
                   "Debes generar otro texto parafraseado que sea lo más diferente posible del original y del parafraseado que recibiste, excepto la palabra clave, que no debes cambiarla." \
                   "Ten en cuenta que el texto es una publicación de Reddit y de un país latinoamericano a la hora de parafrasear. Usa lenguaje informal y latinoamericano." \
                   "Tu respuesta debe ser únicamente el nuevo texto, sin agregar absolutamente nada más."


def initialize_client(key):

  client = OpenAI(
    api_key = key
  )

  return client

def generate_description(keyword, post, paraphrased, client):

  if paraph_num == 2:
    prompt = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role" : "user", "content": f"Texto original: {post} | Texto parafraseado: {paraphrased} | Palabra clave: {keyword}"}
      ]
  if paraph_num == 1:
    prompt = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role" : "user", "content": f"Texto original: {post} | Palabra clave: {keyword}"}
      ]

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    # n=2,
    # temperature=2,
    messages=prompt
  )

  # print(completion.choices[0].message.content)
  # print(completion.choices[1].message.content)
  # print()

  return completion.choices[0].message.content

client = initialize_client(OPENAI_API_KEY)


# index = 0
# post = train_df["post content"][index]
# paraphrased = train_df["paraphrase_esp"][index]
# keyword = train_df["keyword"][index]
# new_text = generate_description(keyword, post, paraphrased, client)
# print(f"Original Text: {post} | Paraphrased Text: {paraphrased} | Keyword: {keyword}")
# print()
# print(new_text)


### TRAIN AUGMENTATION ###
for i, post in enumerate(train_df["post content"]):
    print("Fila:", i)
    keyword = train_df["keyword"][i]
    try:
      paraphrased = train_df["paraphrase_esp"][i]
    except KeyError:
      paraphrased = ""
    response = generate_description(keyword, post, paraphrased, client)
    if paraph_num == 1:
      # Create list of paraphrased texts
      paraphrase_value = response
    elif paraph_num == 2:
      # Create list of paraphrased texts
      paraphrase_value = [train_df["paraphrase_esp"][i], response]
    # Replace the paraphrased text with the new one
    train_df.at[i, "paraphrase_esp"] = paraphrase_value

train_df.to_csv(data_path + "dev_paraphrased.csv", index=False)

# # VAL AUGMENTATION ###
# for i, post in enumerate(val_df["post content"]):
#     print("Fila:", i)
#     keyword = val_df["keyword"][i]
#     paraphrased = val_df["paraphrase_esp"][i]
#     response = generate_description(keyword, post, paraphrased, client)
#     # Create list of paraphrased texts
#     paraphrased_list = [val_df["paraphrase_esp"][i], response]
#     # Replace the paraphrased text with the new one
#     val_df.at[i, "paraphrase_esp"] = paraphrased_list

# val_df.to_csv(data_path + "val_paraphrased.csv", index=False)

# ### TEST AUGMENTATION ###
# for i, post in enumerate(test_df["post content"]):
#     print("Fila:", i)
#     keyword = test_df["keyword"][i]
#     paraphrased = test_df["paraphrase_esp"][i]
#     response = generate_description(keyword, post, paraphrased, client)
#     # Create list of paraphrased texts
#     paraphrased_list = [test_df["paraphrase_esp"][i], response]
#     # Replace the paraphrased text with the new one
#     test_df.at[i, "paraphrase_esp"] = paraphrased_list

# test_df.to_csv(data_path + "test_paraphrased.csv", index=False)

# print("Finished")
