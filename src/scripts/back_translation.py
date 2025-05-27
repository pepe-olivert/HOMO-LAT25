import pandas as pd
from openai import OpenAI
from config import *
import sys

paraph_num = 1

data_path = "../../data/"

test_file = sys.argv[1]

test_path = data_path + test_file
test_df = pd.read_csv(test_path)

if paraph_num == 1:
  DEVELOPER_PROMPT = "You are an assistant responsible for paraphrasing texts to perform data augmentation." \
                   "You will receive a text from a reddit post." \
                   "You will also receive a keyword that is present in the text. You must translate this keyword to English" \
                   "You must paraphrase the text as much as possible, except the keyword, which you must only translate it to English, so that it serves as a new training example." \
                   "Take into account that the text is a reddit post, so it may contain informal language." \
                   "Your response must be only the new text in English, without adding absolutely anything else."

if paraph_num == 2:
  DEVELOPER_PROMPT = "You are an assistant responsible for paraphrasing texts to perform data augmentation." \
                   "You will receive 2 texts, the first one is an original text from a reddit post, and the second one is a paraphrased text of the original." \
                   "You will also receive a keyword that is present in the text. You must translate this keyword to English" \
                   "You must generate another paraphrase text that is the most different as possible from the original and the paraphrased you recieved, except the keyword, which you must only translate it to English." \
                   "Take into account that the text is a reddit post." \
                   "Your response must be only the new text in English, without adding absolutely anything else."


def initialize_client(key):

  client = OpenAI(
    api_key = key
  )

  return client

def generate_description(keyword, post, paraphrased, client):

  if paraph_num == 2:
    prompt = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role" : "user", "content": f"Original Text: {post} | Paraphrased Text: {paraphrased} | Keyword: {keyword}"}
      ]
  if paraph_num == 1:
    prompt = [
        {"role": "developer", "content": DEVELOPER_PROMPT},
        {"role" : "user", "content": f"Original Text: {post} | Keyword: {keyword}"}
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
# post = train_df["translation"][index]
# paraphrased = train_df["paraphrase"][index]
# keyword = train_df["keyword"][index]
# new_text = generate_description(keyword, post, paraphrased, client)
# print(f"Original Text: {post} | Paraphrased Text: {paraphrased} | Keyword: {keyword}")
# print()
# print(new_text)


### TRAIN AUGMENTATION ###
# for i, post in enumerate(train_df["translation"]):
#     print("Fila:", i)
#     keyword = train_df["keyword"][i]
#     try:
#       paraphrased = train_df["paraphrase"][i]
#     except KeyError:
#       paraphrased = ""
#     response = generate_description(keyword, post, paraphrased, client)
#     if paraph_num == 1:
#       # Create list of paraphrased texts
#       paraphrase_value = response
#     elif paraph_num == 2:
#       # Create list of paraphrased texts
#       paraphrase_value = [train_df["paraphrase"][i], response]
#     # Replace the paraphrased text with the new one
#     train_df.at[i, "paraphrase"] = paraphrase_value
    
# train_df.to_csv(data_path + "dev_paraphrased.csv", index=False)

### VAL AUGMENTATION ###
# for i, post in enumerate(val_df["translation"]):
#     print("Fila:", i)
#     keyword = val_df["keyword"][i]
#     try:
#       paraphrased = val_df["paraphrase"][i]
#     except KeyError:
#       paraphrased = ""
#     response = generate_description(keyword, post, paraphrased, client)
#     if paraph_num == 1:
#       # Create list of paraphrased texts
#       paraphrase_value = response
#     elif paraph_num == 2:
#       # Create list of paraphrased texts
#       paraphrase_value = [val_df["paraphrase"][i], response]
#     # Replace the paraphrased text with the new one
#     val_df.at[i, "paraphrase"] = paraphrase_value

# val_df.to_csv(data_path + "val_paraphrased.csv", index=False)

### TEST AUGMENTATION ###
for i, post in enumerate(test_df["translation"]):
    print("Fila:", i)
    keyword = test_df["keyword"][i]
    try:
      paraphrased = test_df["paraphrase"][i]
    except KeyError:
      paraphrased = ""
    response = generate_description(keyword, post, paraphrased, client)
    if paraph_num == 1:
      # Create list of paraphrased texts
      paraphrase_value = response
    elif paraph_num == 2:
      # Create list of paraphrased texts
      paraphrase_value = [test_df["paraphrase"][i], response]
    # Replace the paraphrased text with the new one
    test_df.at[i, "paraphrase"] = paraphrase_value
    
output_file = test_file.replace("translated", "augmented")
print("\nOutput file:", output_file)

test_df.to_csv(data_path + output_file, index=False)

print("Finished")
