import os
import json
import pandas as pd

datapath=''
traindata = [data for data in os.listdir(datapath) if data.endswith('.json')]

dataidentify = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
count = 0


def parsing_logic(data, train_data_name, count):
    final_dataset = []
    for sentence in data[train_data_name]:
        words = sentence['data']
        temp = ''
        for word in words:
            temp = temp + word['text']
        final_dataset.append([temp.rstrip(), dataidentify[count][0], dataidentify[count][1],
                              dataidentify[count][2], dataidentify[count][3]])  # rstrip() -> Stripping next line
    return final_dataset


final_datasets = []
for traindata in traindata:
    try:
        with open('./data/train/' + traindata, encoding="utf8") as data_file:
            data = json.load(data_file)
    except:
        with open('./data/train/' + traindata, encoding="Latin-1") as data_file:
            data = json.load(data_file)
    start_index = 8
    end_index = traindata.find('_full')
    train_data_name = traindata[start_index:end_index]

    print(traindata)
    print(len(data[train_data_name]))
    final_dataset = parsing_logic(data, train_data_name, count)
    count = count + 1
    print(len(final_dataset))
    final_datasets = final_datasets + final_dataset
    print('----------------------------------------------------')

    df = pd.DataFrame(final_datasets, columns=['sentence', 'BookRestaurant', 'GetWeather', 'PlayMusic', 'RateBook'])
    len(df)
    df.to_csv('./data/train/train.csv', index=False, encoding='utf-8')

