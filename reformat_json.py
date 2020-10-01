import os
import argparse 
import json 

def reformat(dataset, filename):
    srcpath = "data/keyphrase/{}/{}_src.txt".format(dataset, filename)
    tgtpath = "data/keyphrase/{}/{}_trg.txt".format(dataset, filename)
    src = open(srcpath, "r", encoding="utf-8")
    tgt = open(tgtpath, "r", encoding="utf-8")

    src_list = []
    for idx, (post, tag) in enumerate(zip(src, tgt)):
        curr_dict = {}
        curr_dict["title"] = ""
        curr_dict["abstract"] = post.strip()
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="twitter")
    config = parser.parse_args()

    main(config)
