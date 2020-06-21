# Customer Service Text Classification

#Prerequisites
torch version 1.5.0+cu101
pandas version 1.0.4
gensim 3.6.0

#Authors
Code by: Yunhao Yang, Zhaokun Xue

#Instruction
In order to run and test the result, please prepare the following dataset:
"2018_tagged_data.xlsx"
"2018 payroll.xlsx"
"chinese_words.txt"

then run:
python lstm.py

Tokenizer.ipynb and Word2Vec.ipynb are only for documentation purpose

#Dataset
Customer Service Dataset: "2018_tagged_data.xlsx"
https://drive.google.com/file/d/1HblWvsC6AGI0vfEJ-MUZRamd0SmrQaxX/view?usp=sharing

Parsed Dataset: "2018 payroll.xlsx"
https://drive.google.com/file/d/1qTB07Xptr5Dl_fGt6lkG9T1Bb-FmxmX5/view?usp=sharing

Chinese dictionary: "chinese_words.txt"
https://drive.google.com/file/d/1XaEePtpKpNOogIIDLqrr-r5Wu545KQLf/view?usp=sharing

Bigram Chinese word to vector from Sogou: "sgns.sogou.bigram"
https://drive.google.com/file/d/1VuzgLcvp2hCITdK--HAoBY9ynv0HB_PV/view?usp=sharing
