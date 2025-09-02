import os
import pandas as pd
import numpy as np

# folder = "./data/mp3s/"
# # print(os.getcwd())
# files = []
# for file in os.listdir(folder):
#     if file.endswith(".mp3"):
#             files.append(file[:-4])

# df = pd.DataFrame(files)
# df.to_csv('./data/labels.csv', index = False, header = False)

# with open('./data/labels.csv', 'r') as file:
#     with open('./data/classes.csv', 'w') as wfile:
#         prevName = ""
#         for line in file:
#             name, index = line.split(',')
#             name = name.strip().split('_')[0][:-1]
#             index = int(index)

#             if name == prevName:
#                 pass
#             else:
#                 prevName = name
#                 wfile.write(f'{index},{name}\n')

with open('./data/labels.csv') as labelFile:
    header = ["fileName","Bat","Cockatoo","Crocodile","Dingo","Duck","Frog","FrogmouthTawny","Koala","Kookaburra","Magpie","Platypus","Possum","Snake","Wombat"]
    df = pd.DataFrame(columns = header)

    # print(len(labelFile.readlines()))

    for i, line in enumerate(labelFile):
        name, label = line.strip().split(',')
        df.loc[i] = [name] + [1 if int(label) == i else 0 for i in range(len(header) - 1)]

        # print(df.iloc(i))

    df.to_csv("./data/multi_labels.csv", index=False)
