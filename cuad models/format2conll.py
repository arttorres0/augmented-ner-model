import json

tag_list = [
  "B-AGMT_DATE",
  "B-DOC_NAME",
  "B-PARTY",
  "I-AGMT_DATE",
  "I-DOC_NAME",
  "I-PARTY",
  "O"
]

tag_dict = {i:j for i,j in enumerate(tag_list)}
print(tag_dict)

f1 = open(f"cuad-v1-annotated.json", "r", encoding="utf8")
data = json.load(f1)

print(data["data"][0])

sentence_i = 0
new_lines = []

f2 = open(f"data.csv", "w", encoding="utf8")

for sentence in data["data"]:
    sentence_i += 1
    if len(sentence["ner_tags"]) != len(sentence["split_tokens"]):
        print("AEA!!!!!!!!!!!!!!!!!!!!!!!!!!")

    for i,tag_index in enumerate(sentence["ner_tags"]):
        tag = tag_dict[tag_index]
        word = sentence["split_tokens"][i]

        if('"' in word):
            word = word.replace('"', "'")
        if("," in word or "=" in word):
            word = f'"{word}"'

        new_lines.append(f"Sentence: {sentence_i},{word},{tag}\n")

f2.writelines(new_lines)
