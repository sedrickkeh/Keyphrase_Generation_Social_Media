import os
import argparse
import json

def reformat(model, steps):
    preds = []
    filepath = "output/{}/pred/stackexchange_step_{}/stackexchange_2.pred".format(model, steps)
    for line in open(filepath, 'r'):
        output = json.loads(line)
        pred_sents = output["pred_sents"]
        pred = ""
        for sent in pred_sents:
            hashtag = ""
            for word in sent: hashtag = hashtag + word + " "
            hashtag = hashtag.strip()
            pred = pred + ";" + hashtag
        preds.append(pred[1:])

    savepath = "output/{}/pred/stackexchange_step_{}/stackexchange_processed.pred".format(model, steps)
    with open(savepath, 'w') as f:
        for pred in preds:
            f.write("%s\n" % pred)


def main(config):
    model = config.model.lower()
    steps = config.steps
    reformat(model, steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="stackexchange_0")
    parser.add_argument('--steps', type=str, default="10000")
    config = parser.parse_args()

    main(config)
