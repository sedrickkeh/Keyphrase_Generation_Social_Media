import json 

def reformat(filename):
    postpath = "data/keyphrase/twitter/twitter-naacl/{}_post.txt".format(filename)
    convpath = "data/keyphrase/twitter/twitter-naacl/{}_conv.txt".format(filename)
    tagpath = "data/keyphrase/twitter/twitter-naacl/{}_tag.txt".format(filename)
    posts = open(postpath, "r")
    convs = open(convpath, "r")
    tags = open(tagpath, "r")

    src_list = []
    for idx, (post, conv, tag) in enumerate(zip(posts, convs, tags)):
        curr_dict = {}
        curr_dict["title"] = post
        curr_dict["abstract"] = conv
        curr_dict["id"] = str(idx)
        curr_dict["src"] = ""
        curr_dict["keywords"] = tag
        src_list.append(curr_dict)

    savepath = "data/keyphrase/json/twitter/twitter_{}.json".format(filename)
    with open(savepath, 'w') as json_file:
        for item in src_list:
            json.dump(item, json_file)
            json_file.write('\n')


def main():
    for data_type in ["test", "valid", "train"]:
        reformat(data_type)

if __name__ == "__main__":
    main()
