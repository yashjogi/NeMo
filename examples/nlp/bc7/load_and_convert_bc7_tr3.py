import os, json, unittest
import pandas as pd

from pdb import set_trace as bp

bc7_tr3_datadir = '/datasets/bc7/Track-3_Med-Tweets'

class DataTest(unittest.TestCase):

    def test_train(self):
        org = load_tsv_convert_to_json('train', testing=True)
        rec = load_json('train')
        self.assertTrue(org.equals(rec))

    def test_val(self):
        org = load_tsv_convert_to_json('val', testing=True)
        rec = load_json('val')
        self.assertTrue(org.equals(rec))


def load_json(part: str) -> str:
    if part == 'train':
        json_file = os.path.join(bc7_tr3_datadir, 'bc7_tr3-train.json')
    elif part == 'val':
        json_file = os.path.join(bc7_tr3_datadir, 'bc7_tr3-val.json')
    else:
        raise('Unsupported partition - option: [train; val]')

    dict_list = []
    with open(json_file, 'r') as rf:
        while True:
            line = rf.readline()
            if not line:
                break
            dict_list.append(json.loads(line))

        df = pd.DataFrame(dict_list)
    return df


def load_tsv_convert_to_json(part: str, testing: bool = False) -> str:
    # part: train/val
    # returns JSON str when testing

    if part == 'train':
        tr_df_0 = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_TrainTask3.0.tsv'), sep='\t')
        tr_df_1 = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_TrainTask3.1.tsv'), sep='\t')
        df = pd.concat([tr_df_0, tr_df_1])
    elif part == 'val':
        df = pd.read_csv(os.path.join(bc7_tr3_datadir, 'BioCreative_ValTask3_corrected.tsv'), sep='\t')
    else:
        raise('Unsupported partition - option: [train; val]')

    df.loc[df['drug']=='-', 'drug'] = 'none'
    df['drug'] = '"' + df['drug'] + '"'

    df2 = df.groupby(['tweet_id', 'text'])['drug'].apply(','.join).reset_index()

    df_ft = df2.copy()
    df_ft['text'] = '<|endoftext|>' + df2['text'] + '<|drug|>' + df2['drug'] + '<|endoftext|>'

    if testing:
        return df_ft
        
    df_json = df_ft[['tweet_id', 'text', 'drug']].to_json(orient='records')
    parsed = json.loads(df_json)

    with open(os.path.join(bc7_tr3_datadir, 'bc7_tr3-' + part + '.json'), 'w', encoding='utf-8') as f:
        for jsoni in parsed:
            f.write(json.dumps(jsoni) + '\n')

    if part == 'train':
        zero_shot_drug_samples_dict = {}
        drugnames_list = df[~df["drug"].str.contains("none")]['drug'].unique().tolist()
        for drni in drugnames_list:
            dfni = df2[df2['drug'].str.contains(drni)]
            zero_shot_drug_samples_dict[drni.lstrip('"').rstrip('"')] = json.loads(dfni[['text','drug']].to_json(orient='records'))
        with open(os.path.join(bc7_tr3_datadir, 'bc7_tr3-fewshot_train.json'), 'w', encoding='utf-8') as f:
            json.dump(zero_shot_drug_samples_dict, f, indent=2)


    

if __name__ == "__main__":
    load_tsv_convert_to_json('train')
    load_tsv_convert_to_json('val')
    # unittest.main()
