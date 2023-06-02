#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
tqdm.pandas()


# In[2]:


data = pd.read_csv('data/may_conversation_data.csv')


# In[3]:


conversations = data[['user_prompt', 'bot_response', 'conversation_id']]
# dropna user_prompt
conversations.dropna(subset=['user_prompt', 'bot_response'], inplace=True)
# group using conversation_id, turn to list: ["user:", user_prompt1, "bot", bot_response1, "user:", user_prompt2, "bot", bot_response2, ...]
conversations['user_prompt'] = conversations['user_prompt'].apply(lambda x: "user: " + str(x))
conversations['bot_response'] = conversations['bot_response'].apply(lambda x: "bot: " + str(x))
conversations['turn'] = conversations['user_prompt'] + '\n' + conversations['bot_response']
conversations = conversations.groupby('conversation_id')['turn'].apply(list)
conversations = conversations.reset_index()

# only reserve conversations with more than 2 turns
conversations = conversations[conversations['turn'].apply(lambda x: len(x) > 2)]
# all turns after the second turn constructs a conversation, e.g., if the conversation has [1, 2, 3, 4], then we have two conversations: [1, 2, 3] and [1, 2, 3, 4]
turn_wise_conversations = []
for i in range(len(conversations)):
    for j in range(2, len(conversations.iloc[i]['turn'])):
        turn_wise_conversation = conversations.iloc[i]['turn'][:j]
        turn_wise_conversations.append("\n".join(turn_wise_conversation))

df = pd.DataFrame(turn_wise_conversations, columns=['dialog'])

# the dialog should end with user utterance, so we should delete the last bot utterance
df['dialog'] = df['dialog'].apply(lambda x: x.split('\n')[:-1])
df['dialog'] = df['dialog'].apply(lambda x: "\n".join(x))

df


# In[4]:


print(df.iloc[11]['dialog'])
df.to_json('data/may_conversation_data_processed.jsonl', orient='records', lines=True)


# In[5]:


tokenizer = AutoTokenizer.from_pretrained("model/train_on_maybe_valid/")
model = AutoModelForSeq2SeqLM.from_pretrained("model/train_on_maybe_valid/")
model.to('cuda')
PREFIX = "Does the context require to search something on the Internet: "
POSTFIX = " __is-search-required__"


# In[6]:


def predict(input_id, decoder_input, model):
    next_token_logits = model(input_ids=input_id, decoder_input_ids=decoder_input)[0]
    # use softmax and get the probability of token 2264 (not) and 13173 (search)
    do_not_search_prob = torch.softmax(next_token_logits[0, -1, :], dim=0)[2264].item()
    do_search_prob = torch.softmax(next_token_logits[0, -1, :], dim=0)[13173].item()
    output = model.generate(input_id, max_length=50)
    # print(torch.cuda.memory_allocated())
    return tokenizer.decode(output[0], skip_special_tokens=True), do_not_search_prob, do_search_prob

decoder_input = torch.tensor([0, 3, 834, 834, 26, 32, 18]).unsqueeze(0).to('cuda')

flattened_convos = [
    """bot:Hello, welcome to Alexa social bot. What do you want to chat? user:what is the weather like by you? bot:Its raining here :( so im indoors playing board games, what about you? user:what games are you playing bot:Im a big fan of chess and i sometimes play checkers, do you play a lot of board games? user:not a lot, but i do play checkers.  since you like chess have you watched the queens gamit on netflix? bot:How much do you play? and I havent watched it yet, do you know if its any good? user:i just started watching, it is phenomenal, and i dont know anything about chess bot:Ill definitely check it out then! Have you ever played much chess? user:no i have not, but i would love to learn.""",
    """bot:Hello, welcome to Alexa social bot. What do you want to chat? user:Hoodies are my favorite type of clothing; I like how versatile they are. bot:I agree! my favorite brand of hoodies is the North Face Gordon Lyons Hoodie; this is currently on clearance sale at Macys  user:Oh wow, really? I should totally go to Macy's and get one! bot:Good for you; you can get it online also with free shipping user:That's awesome. Did you already get one? """,
    """bot:Hello, welcome to Alexa social bot. What do you want to chat? user:I went to the card shop today and picked up a pack of baseball cards. I ended up getting a rare Ken Griffey Jr! I can't wait to get a nice protective sleeve for it. bot:There are many rare Ken Griffey Jr cards one of the most rare would be  the 1989 Bowman Tiffany cards that were of a rare set of special factory print that only printed 6000 cards user:Wow! It would be quite something for me to own that card one day. It would be a prized possession to me. Do you watch baseball? bot:I do not watch baseball. Although baseball is really good for cardiovascular training.  user:That's too bad, I think baseball is fascinating with all its statistics. Do you train for anything in particular, a marathon perhaps? bot:I do not train for anything specific, I just feel that one should be healthy on the inside.  user:That's a good point. It's important to keep the mind and body sharp, but eating healthy is paramount too. What are you doing tonight? """
]

for flattened_convo in flattened_convos:
    flattened_convo = flattened_convo.replace(" user:", "\n").replace(" bot:", "\n").replace("bot:", "")
    print("Input:", flattened_convo)
    input_id = tokenizer(PREFIX + flattened_convo, return_tensors='pt').input_ids.to('cuda')
    outputs = predict(input_id, decoder_input, model)
    print("Output:", outputs[0])
    print("Do not search prob:", outputs[1])
    print("Do search prob:", outputs[2])
    print("=============")


# In[10]:


df['flattened'] = df['dialog'].apply(lambda x: x.replace("user:", "").replace("bot:", ""))
# reserve only the last 3 turns ([-7:])
df['reserved'] = df['flattened'].apply(lambda x: '\n'.join(x.split('\n')[-7:]))
reserved = df['reserved'].tolist()

print("tokenizing...")
input_ids = []
for i in tqdm(range(len(reserved))):
    input_ids.append(tokenizer.encode(PREFIX + reserved[i] + POSTFIX, return_tensors="pt").to('cuda'))

# create a column 'do_search_prob' to store the probability of searching
print("predicting...")
df['do_search_prob'] = 0.0
for i, input_id in enumerate(tqdm(input_ids)):
    outputs = predict(input_id, decoder_input, model)
    df['do_search_prob'].iloc[i] = outputs[2]

# save the conversations
df.to_json('data/may_conversation_data_processed_with_prob.jsonl', orient='records', lines=True)


# In[ ]:




