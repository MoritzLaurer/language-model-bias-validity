

#import pandas as pd

#df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Judicial-supreme_court_cases_20.1.csv", sep=",", encoding='utf-8') #encoding='utf-8',  # low_memory=False  #lineterminator='\t',

#df = pd.read_csv("https://comparativeagendas.s3.amazonaws.com/datasetfiles/US-Exec_SOTU_2022.csv", sep=",", encoding='utf-8', on_bad_lines='skip', encoding_errors="replace")


"""
tokens_lst = ["ĠWouter", "Ġvan", "ĠAt","te", "veld", "ĠNel", "ĠR", "uig"]
tokens_lst_concatenated = []
tokens_part_of_one_word = []
tokens_part_of_one_entity = []
for i, token in enumerate(tokens_lst):
    if ("Ġ" in token) and (len(tokens_part_of_one_word) == 0):
        tokens_part_of_one_word.append(token)
    elif "Ġ" not in token:
        tokens_part_of_one_word.append(token)
        # for very last word in list
        if i == len(tokens_lst)-1:
            tokens_merged_to_word = "".join(tokens_part_of_one_word).replace("Ġ", "")
            tokens_lst_concatenated.append(tokens_merged_to_word)
    elif "Ġ" in token:
        tokens_merged_to_word = "".join(tokens_part_of_one_word).replace("Ġ", "")
        tokens_lst_concatenated.append(tokens_merged_to_word)
        tokens_part_of_one_word = []
        tokens_part_of_one_word.append(token)
print(tokens_lst_concatenated)


print(tokens_lst_concatenated)

"""