import numpy as np
import pandas as pd
import seaborn as sns
#%pylab inline
import re
import matplotlib.pyplot as plt

df = pd.read_csv('../hacker-news-corpus/hacker_news_sample.csv')
df.info()
df['domain'] = df['url'].str.extract(
    '^http[s]*://([0-9a-z\-\.]*)/.*$', flags=re.IGNORECASE, expand=False)
df_groupby = df.groupby(by='domain')
df_groupby['score'].count().sort_values(ascending=False)[0:20]
'''
domain
github.com                 14368
medium.com                 12225
www.youtube.com            11818
techcrunch.com              7858
www.nytimes.com             7831
arstechnica.com             4714
www.wired.com               3733
en.wikipedia.org            2961
www.bbc.co.uk               2747
www.theguardian.com         2686
www.theverge.com            2548
www.bloomberg.com           2467
www.businessinsider.com     2421
www.washingtonpost.com      2289
venturebeat.com             2183
www.forbes.com              2148
www.theatlantic.com         2002
mashable.com                1954
thenextweb.com              1933
twitter.com                 1842
'''

# Plot a horizontal bar graph displaying the frequency of a given topic by domain


def freqMentioned(df, domain_list, category, topic_list, colors):
    data = df.loc[df['domain'].isin(domain_list)]
    topic_code_list = []
    for i in topic_list:
        topic_code = i.split('|')[0].upper()
        topic_code_list.append(topic_code)
        data[topic_code] = data['text'].map(lambda x: len(
            re.findall(r'\b%s\b' % i, x)) > 0)
        data.loc[data[topic_code] == False, topic_code] = np.nan
    domain = domain_list[0]
    data_out = pd.DataFrame(data.loc[data['domain'] == domain].count())
    data_out = (data_out.T)[topic_code_list]
    # sort the columns by summed occurence in domain specified
    domains = domain_list.copy()
    domains.remove(domain)
    for i in domains:
        a = pd.DataFrame(data.loc[data['domain'] == i].count())
        a = (a.T)[topic_code_list].copy()
        data_out = pd.concat([data_out, a], axis=0)
    dictionary = {}
    for i in topic_code_list:
        dictionary[i] = data_out[i].sum()
    sorted_dictionary = sorted(
        dictionary.items(), key=lambda item: item[1], reverse=True)
    data_out = data_out[[i[0] for i in sorted_dictionary]]
    data_out.index = domain_list
    data_out.T.plot(kind="barh", width=.6, stacked=True, figsize=(
        10, len(topic_list) // 2), color=colors).legend(bbox_to_anchor=(1, 1))
    plt.savefig('domain_%s.png' % category, dpi=200)
    plt.clf()
    plt.cla()
    plt.close()
    return data_out


df = df.dropna(subset=['text'])
df['text'] = df['text'].str.lower()

hot_domains = ['github.com', 'medium.com', 'www.youtube.com', 'techcrunch.com',
               'www.nytimes.com', 'arstechnica.com', 'www.wired.com', 'en.wikipedia.org']
hot_colors = ['dodgerblue', 'navy', 'r', 'green',
              'black', 'orange', 'purple', 'grey', 'lemon']
nation_topics = ['usa|america|american|new york|washington|white house', 'rus|russia|russian|moscow', 'chn|china|chinese|beijing|shanghai|shenzhen|guangzhou',
                 'jpn|japan|japanese|tokyo', 'deu|german|germany|berlin', 'fra|france|french|paris', 'gbr|english|london|england|united kingdom|uk']


freqMentioned(df, hot_domains, 'nation', nation_topics, hot_colors)

sci_topics = ['bio|genome|genomic||sequence alignment|sequence data', 'nlp|natural language|ls[ai]|lda|curpos|curpora|translation|[a-z]*hash|lsh', 'ml|machine learning|deep learning|brain|neural|regression|bayesian|ai|bot',
              'bigdata|big data|data analysis|cluster|time series|series analysis|network', 'bitcoin|blockchain|litecoin|dogecoin|crypto[- ]?currency', 'vr|ar|virtual reality|3d', 'p2p|torrents?|dht|[ae]mule', 'iot|internet of thing']
freqMentioned(df, hot_domains, 'sci', sci_topics, hot_colors)
