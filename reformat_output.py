import json

preds = []
for line in open('output/stackexchange_2_2/pred/stackexchange_2.one2seq.cdkgen_step_10000/stackexchange_2.pred', 'r'):
    output = json.loads(line)
    pred_sents = output["pred_sents"]
    pred = ""
    for sent in pred_sents:
        hashtag = ""
        for word in sent: hashtag = hashtag + word + " "
        hashtag = hashtag.strip()
        pred = pred + ";" + hashtag
    preds.append(pred[1:])

with open('output/stackexchange_2_2/pred/stackexchange_2.one2seq.cdkgen_step_10000/stackexchange_2_processed.pred', 'w') as f:
    for pred in preds:
        f.write("%s\n" % pred)