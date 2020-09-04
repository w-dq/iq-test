from iqtest import iqtest_base
import torch
import numpy as np

class IQTestVerbalSample(iqtest_base.IQTestModelBase):
    def pre_run(self):
        # setup your model here
        pass

    def solve(self, question):
        result_list = list()
        import gensim
        from scipy.spatial.distance import cosine as dist_cosine
        model = gensim.models.KeyedVectors.load_word2vec_format(
            './train_data/GoogleNews-vectors-negative300.bin', binary=True)
        for question in question:
            max_sim, curr_idx = 0, 0
            # if word not in model_vocabulary, return a fixed answer [1]
            if question['stem'] not in model:
                result_list.append([question['id'], [1]])
                continue
            flag = False
            for idx, answer in enumerate(question['options']):
                if answer in model:
                    flag = True
                else:
                    continue
                if not flag:
                    result_list.append([question['id'], [1]])
                sim = 1 - dist_cosine(model[question['stem']], model[answer])
                if sim > max_sim:
                    max_sim = sim
                    curr_idx = idx
            result_list.append([question['id'], [curr_idx+1]])

        return result_list

class IQTestSeqSample(iqtest_base.IQTestEvalBase):
    def pre_run(self):
        # setup your model here
        pass

    def solve(self, question):
        # question_id, answer_list
        return [question['id'], [2]]

class IQTestDiagramSample(iqtest_base.IQTestModelBase):
    def pre_run(self):
        # setup your model here
        pass

    def solve(self, question):
        # question_id, answer_list
        return [question['id'], [1]]


def get_model_object(category: str) -> object:
    """ model by category
    if don't support category, return None 

    :param category: test category 
    :type category: str
    :return: Model 
    :rtype: object
    """
    if category == 'seq':
        return IQTestSeqSample()
    elif category == 'diagram':
        return IQTestDiagramSample()
    elif category == 'verbal':
        return IQTestVerbalSample()
    return None

def global_pre_run():
    """ global pre run function used to set up environment
    """
    device = torch.device("cuda:0")
    try:
        data = torch.tensor(np.random.rand(10), device=device)
        print(data)
    except:
        pass

