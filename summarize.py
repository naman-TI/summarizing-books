import os
import re
import openai
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize

print('re: ', re.__version__)
print('nltk: ', nltk.__version__)

FILENAME = "./Math App SME Interview - Transcript.txt"

def count_tokens(filename):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    return len(tokens)


def break_up_file(tokens, chunk_size, overlap_size):
    if len(tokens) <= chunk_size:
        yield tokens
    else:
        chunk = tokens[:chunk_size]
        yield chunk
        yield from break_up_file(tokens[chunk_size-overlap_size:], chunk_size, overlap_size)


def break_up_file_to_chunks(filename, chunk_size=2000, overlap_size=100):
    with open(filename, 'r') as f:
        text = f.read()
    tokens = word_tokenize(text)
    return list(break_up_file(tokens, chunk_size, overlap_size))

def convert_to_prompt_text(tokenized_text):
    prompt_text = " ".join(tokenized_text)
    prompt_text = prompt_text.replace(" 's", "'s")
    return prompt_text

if __name__ == '__main__':
    filename = FILENAME
    openai.api_key = os.getenv("OPENAI_API_KEY")
    ## Divide the text into segments of 2,000 tokens, 
    ## with an overlap of 100 tokens to avoid losing any information from the split.
    chunks = break_up_file_to_chunks(filename)
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {len(chunk)} tokens")

    prompt_response = []
    chunks = break_up_file_to_chunks(filename)

    for i, chunk in enumerate(chunks):

        prompt_request = "Summarize this meeting transcript: " + convert_to_prompt_text(chunks[i])
        
        messages = [{"role": "system", "content": "This is text summarization."}]    
        messages.append({"role": "user", "content": prompt_request})

        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=.5,
                max_tokens=500,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
        )
        
        prompt_response.append(response["choices"][0]["message"]['content'].strip())

    
    ## Consolidate the meeting summaries 
    prompt_request = "Consolidate these meeting summaries: " + str(prompt_response)
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_request,
        temperature=.5,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    meeting_summary = response["choices"][0]["text"].strip()
    print(meeting_summary)

    
