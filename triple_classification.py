# -*-coding:utf-8-*-
# Author: Kang Liu <kangliu@stu.xidian.edu.cn>

from pykeen.triples import TriplesFactory
from pykeen.pipeline import pipeline
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import torch

warnings.filterwarnings('ignore')


def get_train_test(train_path, test_path):
    """
    返回TriplesFactory类型的train, test
    :param train_path: str, train.txt路径, train.txt中每一行为一个三元组<h, r, t>, 并且h, r, t以\t分割。
    h，r分别表示头、尾实体；r表示关系
    :param test_path: str, test.txt路径, test.txt中每一行为一个三元组<h, r, t>, 并且h, r, t以\t分割。
    h，r分别表示头、尾实体；r表示关系
    :return: training, testing
    """
    training = TriplesFactory.from_path(train_path)
    testing = TriplesFactory.from_path(test_path)
    # print(training.num_entities, testing.num_entities)
    return training, testing


def fit_model(training, testing, model_name="transe", is_save_model=False, save_path=None, num_epochs=2000):
    """
    训练模型
    :param training: TriplesFactory
    :param testing: TriplesFactory
    :param model_name: str, 可以选择：['conve', 'convkb', 'distmult', 'complex',
    #         'ermlp', 'ermlpe', 'hole', 'kg2e', 'mure', 'ntn', 'pairre', 'proje', 'rescal',
    #         'rotate', 'simple', 'structuredembedding', 'transd', 'transe', 'transh',
    #         'transr']
    :param is_save_model: bool, 确定是否保存模型
    :param save_path: str, 如果保存模型，提供的保存模型的位置
    :param num_epochs: int (2000 default) 模型训练的epochs数
    :return: model
    """

    pipe_result = pipeline(
        training=training,
        testing=testing,
        model=model_name,
        # loss='BCEWithLogitsLoss',
        loss='nssa',
        # regularizer='lp',
        training_kwargs=dict(num_epochs=num_epochs),
        # use_testing_data=False
    )
    model = pipe_result.model
    if is_save_model:
        try:
            pipe_result.save_model(save_path)
        except ValueError as e:
            print(repr(e))
            exit(-1)
    return model


def predict(training, testing, model):
    """
    判断test中三元组是否正确，1：True; 0: False
    :param training: TriplesFactory
    :param testing: TriplesFactory
    :param model: model
    :return: predict_label, candidate_answers
    predict_label: ndarray, 1: True; 0: False
    candidate_answers: 2D-list. 如果h, r, t中任意一个在知识图谱中没有出现过，则返回["Nan", "Nan", "Nan"]
    """
    # obtain the basic types of train and test
    entity_to_id = training.entity_to_id
    id_to_entity = training.entity_id_to_label
    id_to_relation = training.relation_id_to_label
    relation_to_id = training.relation_to_id

    # 初始化test的labels
    test = testing.triples  # testing原始三元组
    predict_label = [0] * len(test)
    candidate_answers = []

    # 构造test_batch
    test_batch = []
    for i in range(len(test)):
        t = test[i]
        try:
            test_batch.append([entity_to_id[t[0]], relation_to_id[t[1]], entity_to_id[t[2]]])
        except KeyError as k:
            predict_label[i] = -1
            candidate_answers.append(["Nan"] * 3)
            print("知识图谱中不能在实体或者关系：", str(k), ", 原始三元组为：", t, sep='')

    test_batch = torch.LongTensor(test_batch)
    # print(test_batch, "test_batch")

    # 预测头实体、关系、尾实体
    scores_h = model.predict_h(test_batch[:, 1:])
    scores_r = model.predict_r(test_batch[:, [0, 2]])
    scores_t = model.predict_t(test_batch[:, :2])
    # print(scores_h.shape, scores_r.shape, scores_t.shape)

    j = 0
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    for i in range(test.shape[0]):
        if predict_label[i] == -1:
            print("这个三元组中的实体或者关系不在知识图谱中", test[i], sep="")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            continue
        orig_triple = test[i, :].tolist()
        print("original triple: ", orig_triple, sep="")
        temp_scores = scores_h[j, :].tolist()
        max_score_idn = np.argmax(temp_scores)
        h_triple = [id_to_entity[max_score_idn], test[i, 1], test[i, 2]]
        print("predict h: ", h_triple)

        temp_scores = scores_r[j, :].tolist()
        max_score_idn = np.argmax(temp_scores)
        r_triple = [test[i, 0], id_to_relation[max_score_idn], test[i, 2]]
        print("predict r: ", r_triple)

        temp_scores = scores_t[j, :].tolist()
        max_score_idn = np.argmax(temp_scores)
        t_triple = [test[i, 0], test[i, 1], id_to_entity[max_score_idn]]
        print("predict t: ", t_triple)

        temp_result = 0
        temp_result += h_triple == orig_triple
        temp_result += r_triple == orig_triple
        temp_result += t_triple == orig_triple
        # temp_result += h_triple == r_triple
        # temp_result += h_triple == t_triple
        # temp_result += r_triple == t_triple
        print("original triple: ", orig_triple, ", equal times:", temp_result, sep='')
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        j += 1

        candidate_answers.append([h_triple, r_triple, t_triple])

        predict_label[i] = temp_result
        # result.append(temp_result)
    print("times:", predict_label)
    predict_label = np.array(predict_label) >= 2

    # Candidate answers for error triples
    # candidate_answers = np.array(candidate_answers)
    # candidate_answers = candidate_answers[predict_label]
    print("predict_label:", predict_label.astype(int))
    return predict_label.astype(int), candidate_answers

    # temp_auc = accuracy_score(label, predict_label)
    # print(classification_report(true_label, predict_label))
    # auc.append(temp_auc)


if __name__ == '__main__':
    print("get training and testing...")
    training, testing = get_train_test("data/school_name.txt", "data/test_data.txt")

    print("fit model...")
    model = fit_model(training, testing)

    print("predict_label...")
    predict_label, candidate = predict(training, testing, model)

    # print the result and candidate answers
    test = testing.triples
    for i, j, k in zip(predict_label, test, candidate):
        print(i, j, k)