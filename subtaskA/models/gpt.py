import os
from openai import OpenAI
import pandas as pd

client = OpenAI()

OpenAI.api_key = os.getenv('OPENAI_API_KEY')

test_df = pd.read_json('subtaskA/data/subtaskA_dev_monolingual.jsonl', lines=True)    

output_file = 'subtaskA/predictions/gpt.json'  
with open(output_file, 'w', encoding='utf-8') as f:
    for idx, ex in enumerate(test_df['text'][:10]):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a language model designed to determine if the following text is written by a human or a machine."},
                {"role": "user", "content": f"Determine if this text is written by a human or a machine. Answer with a single word, 'human' or 'machine'\n\n{ex}"},
            ]
        )

        model_output = response.choices[0].message.content
        print(model_output)
        prediction = 1 if "machine" in model_output.lower() else 0

        output_line = f'{{"id": {int(test_df.loc[idx, "id"])}, "label": {prediction}}}\n'
        f.write(output_line)

print(f'Predictions saved to {output_file}')