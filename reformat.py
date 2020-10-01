import os
import argparse
import json 

def reformat(dataset, filename):
    srcpath = "data/keyphrase/{}/{}_src.txt".format(dataset, filename)
    trgpath = "data/keyphrase/{}/{}_trg.txt".format(dataset, filename)
    src = open(srcpath, "r", encoding='utf-8')
    trg = open(trgpath, "r", encoding='utf-8')

    src_list = []
    for idx, text in enumerate(src):
        curr_dict = {}
        curr_dict["title"] = ""
        curr_dict["abstract"] = ""
        curr_dict["id"] = str(idx)
        curr_dict["src"] = text.strip()
        src_list.append(curr_dict)
    
    savepath = "data/keyphrase/meng17/{}".format(dataset)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath_file = "{}/{}_{}.src".format(savepath, dataset, filename)
    with open(savepath_file, 'w', encoding='utf-8') as json_file:
        for item in src_list:
            json.dump(item, json_file)
            json_file.write('\n')

    hashtag_list = []
    for idx, tags in enumerate(trg):
        curr_dict = {}
        curr_dict["keywords"] = tags.strip().split(';')
        curr_dict["id"] = str(idx)
        curr_dict["tgt"] = tags.strip().split(';')
        hashtag_list.append(curr_dict)

    savepath_file = "{}/{}_{}.tgt".format(savepath, dataset, filename)
    with open(savepath_file, 'w', encoding='utf-8') as json_file:
        for item in hashtag_list:
            json.dump(item, json_file)
            json_file.write('\n')


def main(config):
    dataset = config.dataset.lower()
    for data_type in ["test", "valid", "train"]:
        reformat(dataset, data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="twitter")
    config = parser.parse_args()

    main(config)
