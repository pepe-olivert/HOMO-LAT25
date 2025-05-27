import pandas as pd
from openai import OpenAI
from config import *
import sys

data_path = "../../data/"

test_file = sys.argv[1]

train_path = data_path + test_file
test_df = pd.read_csv(train_path)

# DEVELOPER_PROMPT = "You are an assistant in charge of translating texts from different Latin American dialects into English." \
#                    "You will receive a keyword, a text and the country from which it originates." \
#                    "First, you must translate the text into English, and then respond only with the part of the translated text that talks about the keyword you have received." \
#                    "Your response must be only the part of the translated text in English, without adding absolutely anything else"


DEVELOPER_PROMPT = "You are an assistant in charge of translating texts from different Latin American dialects into English." \
                   "You will receive a text and the country from which it originates." \
                   "Your response must be only the translated text in English, without adding absolutely anything else."

COUNTRY_PROMPT = "Country:"
TEXT_PROMPT = "Text:"

def initialize_client(key):

  client = OpenAI(
    api_key = key
  )

  return client

def generate_description(country, post, client):

  completion = client.chat.completions.create(
    model="gpt-4o-mini",
    store=True,
    messages=[
      {"role": "developer", "content": DEVELOPER_PROMPT},
      {"role" : "user", "content": f"{COUNTRY_PROMPT} {country}, {TEXT_PROMPT} {post}"}
    ]
  )

  return completion.choices[0].message.content

client = initialize_client(OPENAI_API_KEY)

# index = 0
# country = train_df["country"][index]
# print(country)
# post = train_df["post content"][index]
# print(post)
# translation = generate_description(country, post, client)
# print(f"{COUNTRY_PROMPT} {country}, {TEXT_PROMPT} {post}")
# print()
# print(translation)

translations = []
for i, post in enumerate(test_df["post content"]):
    print("Fila:", i)
    country = test_df["country"][i]
    response = generate_description(country, post, client)
    translations.append(response)

test_df["translation"] = translations

output_file = test_file.replace("cleaned", "translated")

test_df.to_csv(data_path + output_file, index=False)
