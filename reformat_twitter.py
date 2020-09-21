import json 

def reformat_tags(filename):
    readpath = "data/keyphrase/twitter/twitter-naacl/{}_tag.txt".format(filename)
    tags = open(readpath, "r")
    tags_list = []
    for idx, x in enumerate(tags):
        curr_dict = {}
        curr_dict["keywords"] = x.strip().split(';')
        curr_dict["id"] = str(idx)
        curr_dict["tgt"] = x.strip().split(';')
        tags_list.append(curr_dict)

    savepath = "data/keyphrase/meng17/twitter/twitter_{}.tgt".format(filename)
    with open(savepath, 'w') as json_file:
        for item in tags_list:
            json.dump(item, json_file)
            json_file.write('\n')

def reformat_post(filename):
    postpath = "data/keyphrase/twitter/twitter-naacl/{}_post.txt".format(filename)
    convpath = "data/keyphrase/twitter/twitter-naacl/{}_conv.txt".format(filename)
    posts = open(postpath, "r")
    convs = open(convpath, "r")
    src_list = []
    for idx, (post, conv) in enumerate(zip(posts, convs)):
        curr_dict = {}
        curr_dict["title"] = ""
        curr_dict["abstract"] = post.strip()
        curr_dict["id"] = str(idx)
        curr_dict["src"] = conv.strip()
        src_list.append(curr_dict)

    savepath = "data/keyphrase/meng17/twitter/twitter_{}.src".format(filename)
    with open(savepath, 'w') as json_file:
        for item in src_list:
            json.dump(item, json_file)
            json_file.write('\n')

def main():
    for data_type in ["test", "valid", "train"]:
        reformat_tags(data_type)
        reformat_post(data_type)

if __name__ == "__main__":
    main()