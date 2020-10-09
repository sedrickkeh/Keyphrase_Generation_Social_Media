import os
import argparse
import json 

def reformat(dataset, filename):
    postpath = "data/keyphrase/{}/{}_post.txt".format(dataset, filename)
    convpath = "data/keyphrase/{}/{}_conv.txt".format(dataset, filename)
    trgpath = "data/keyphrase/{}/{}_tag.txt".format(dataset, filename)
    post = open(postpath, "r", encoding='utf-8')
    conv = open(convpath, "r", encoding='utf-8')
    trg = open(trgpath, "r", encoding='utf-8')

    src_list = []
    for idx, (text_1, text_2) in enumerate(zip(post, conv)):
        curr_dict = {}
        curr_dict["title"] = ""
        curr_dict["abstract"] = ""
        curr_dict["id"] = str(idx)
        curr_dict["src"] = text_1.strip() + " " + text_2.strip()
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

def reformat_json(dataset, filename):
    postpath = "data/keyphrase/{}/{}_post.txt".format(dataset, filename)
    convpath = "data/keyphrase/{}/{}_conv.txt".format(dataset, filename)
    trgpath = "data/keyphrase/{}/{}_tag.txt".format(dataset, filename)
    post = open(postpath, "r", encoding='utf-8')
    conv = open(convpath, "r", encoding='utf-8')
    trg = open(trgpath, "r", encoding='utf-8')

    src_list = []
    for idx, (text_1, text_2, tag) in enumerate(zip(post, conv, trg)):
        curr_dict = {}
        curr_dict["title"] = ""
        curr_dict["abstract"] = text_1.strip() + " " + text_2.strip()
        curr_dict["id"] = str(idx)
        curr_dict["keywords"] = tag.strip()
        src_list.append(curr_dict)

    savepath = "data/keyphrase/json/{}".format(dataset)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath_file = "{}/{}_{}.json".format(savepath, dataset, filename)
    with open(savepath_file, 'w', encoding="utf-8") as json_file:
        for item in src_list:
            json.dump(item, json_file)
            json_file.write('\n')


def main(config):
    dataset = config.dataset.lower()
    for data_type in ["test", "valid", "train"]:
        reformat(dataset, data_type)
        reformat_json(dataset, data_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="twitter_conv")
    config = parser.parse_args()

    main(config)
