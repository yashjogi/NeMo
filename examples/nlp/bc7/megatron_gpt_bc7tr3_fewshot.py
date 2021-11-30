import requests, json, random

NUM_PROMPT_SAMPLES = 2

def run(model):
    if model == 'nemo_1.3b':
        url = "http://10.110.42.59:9000/megagpt"
    elif model == 'megatron_530b':
        url = "http://10.14.74.235:5000/api"
    else:
        raise('unknown model/api')

    with open('/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-fewshot_train.json') as f:
        few_json = json.load(f)

    with open('/datasets/bc7/Track-3_Med-Tweets/bc7_tr3-fewshot_val.json') as f:
        val_json = json.load(f)

    val_prompts = []
    val_true = []
    val_pred = []
    for dri_val in val_json.keys():

        if dri_val == 'none':
            val_json_dri = random.sample(val_json[dri_val], 100)
        else:
            val_json_dri = val_json[dri_val]
        for vali in val_json_dri:

            prompts = []
            for dri_tr in random.sample(list(few_json.keys()), NUM_PROMPT_SAMPLES - 1) + ['none']:
                fewshot_sample = random.sample(few_json[dri_tr], 1)[0]
                prompts.append(fewshot_sample['text'] + 'DRUG:' + fewshot_sample['drug'])
            prompts_str = '\n'.join(prompts)

            val_prompt = prompts_str + '\n' + vali['text'] + ' DRUG:'
            val_true.append(vali['drug'])
            val_prompt_ = 'what is the meaning of a life? A:'

            headers = {
                "Content-Type": "application/json; charset=UTF-8"
            }
            payload = '{"prompts":["%s"], "tokens_to_generate":5}'%(val_prompt).encode('utf-8')

            response = requests.request("PUT", url, headers=headers, data=payload)
            print(response.text.encode('utf-8'))

            aa = 1

if __name__ == "__main__":
    # run('nemo_1.3b')
    run('megatron_530b')