from utils import event_type
from utils import MAX_SEQ_LEN, train_file_path, test_file_path, dev_file_path
from keras.models import load_model
import numpy as np
from collections import defaultdict
from albert_zh.extract_feature import BertVector
import json
from pprint import pprint


bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
f = lambda text: bert_model.encode([text])["encodes"][0]

# 读取label2id字典
with open("%s_label2id.json" % event_type, "r", encoding="utf-8") as h:
    label_id_dict = json.loads(h.read())
id_label_dict = {v: k for k, v in label_id_dict.items()}

model = load_model("%s_ner.h5" % event_type)


def get_entity(sent, tags_list):

    entity_dict = defaultdict(list)
    i = 0
    for char, tag in zip(sent, tags_list):
        if 'B-' in tag:
            entity = char
            j = i+1
            entity_type = tag.split('-')[-1]
            while j < min(len(sent), len(tags_list)) and 'I-%s' % entity_type in tags_list[j]:
                entity += sent[j]
                j += 1

            entity_dict[entity_type].append(entity)

        i += 1

    return dict(entity_dict)


with open("./predict_data/test.txt",'r',encoding = "utf-8") as f:
    text = f.read().replace(" ","")
    text = '昨天进行的女单半决赛中，陈梦4-2击败了队友王曼昱，伊藤美诚则以4-0横扫了中国选手丁宁。'.replace(' ', '')
    print(text)
    train_x = np.array([f(text)])
    y = np.argmax(model.predict(train_x), axis=2)
    y = [id_label_dict[_] for _ in y[0] if _]
    # 输出预测结果
    pprint(get_entity(text, y))
    out_file = open('results.txt','w',encoding='utf-8')
    out_file.write(y)
    out_file.close()
    f.close()




