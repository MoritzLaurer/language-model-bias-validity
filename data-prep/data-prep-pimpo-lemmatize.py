

import pandas as pd
import spacy

### !!! not sure if necessary, because already done well in file for multilingual paper
## ! just saving samller file with less columns for now

## load already translated data
df = pd.read_csv(f"/Users/moritzlaurer/Dropbox/PhD/Papers/multilingual/multilingual-repo/data-clean/df_pimpo_samp_trans_m2m_100_1.2B_embed_tfidf.zip", engine='python')

df.columns

df_cl = df[['label', 'label_text', 'country_iso', 'language_iso', 'doc_id',
           'text_original', 'text_preceding', 'text_following', 'selection',
           'certainty_selection', 'topic', 'certainty_topic', 'direction',
           'certainty_direction', 'rn', 'cmp_code', 'partyname', 'partyabbrev',
           'parfam', 'parfam_text', 'date', 'language_iso_fasttext',
           'text_preceding_trans', 'text_original_trans', 'text_following_trans',
           'language_iso_trans',
           #'text_concat', 'text_concat_embed_multi',
           #'text_trans_concat',
           #'text_trans_concat_embed_en',
           'text_trans_concat_tfidf', #'text_prepared'
            ]]


# save to disk
#df_cl.to_csv(f"./data-clean/df_pimpo_samp_trans_lemmatized_stopwords.zip",
#                compression={"method": "zip", "archive_name": f"df_pimpo_samp_trans_lemmatized_stopwords.csv"}, index=False)



