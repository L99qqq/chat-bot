# request gpt's api

import requests
import json
from retrying import retry

@retry(stop_max_attempt_number=7)
def request_api(prompt, idx):

    url = 'https://abc.gptmf.top/v1/chat/completions'

    gpt_key_list = []
    
    gpt_key = gpt_key_list[idx]
        
    headers = {
        'Authorization': f'Bearer {gpt_key}',
        'Content-Type': 'application/json'
        }
    try:
        response =  requests.post(url, headers=headers, json={
                    "model": "gpt-3.5-turbo",
                    "temperature": 0.7,
                    'max_tokens': 512,
                    'top_p': 1.,
                    'frequency_penalty': 0,
                    'presence_penalty': 0,
                    'messages':[{
                        "role":'user',
                        'content': prompt
                    }]
                    },
                    stream=False,
                    verify=True,
                    timeout=30
                )
        data = response.content.decode()
        json_data = json.loads(data)
        return json_data['choices'][0]['message']['content']
    except requests.exceptions.Timeout:
        print("api error: timeout", "api key idx:", idx)
        raise Exception("raise exception")
        return 'timeout' 
    except Exception as e:
        print("api error:", e)
        raise Exception("raise exception")

    return None
    pass

if __name__ == '__main__':
    # test
    test_prompt = 'hello!'
    res = request_api(test_prompt, idx=3)
    print(res)