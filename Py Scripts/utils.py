import numpy as np
import math
import random as random
from random import randint
from collections import OrderedDict

brackets = dict()
brackets["("] = ")"
brackets["["] = "]"
brackets["{"] = "}"
brackets["<"] = ">"
brackets = OrderedDict(sorted(brackets.items(), key=lambda t: t[0]))

def calculate_prob(my_brackets, cur_sent, num_types, prob, num_open):
    '''
    Calculate probability of a sentence given the probabilities of open brackets. 
    '''
    sent_prob = 0
    counter = 0
    for bracket in cur_sent:
        if bracket in my_brackets:
            counter += 1
            ind = my_brackets.index(bracket)
            sent_prob += math.log(prob[ind],2)
        else:
            if counter<num_open:
                sent_prob += math.log(prob[-1],2)
            else:
                sent_prob += math.log(1,2)
    if counter<num_open:
        # sentence contains less open brackets than max number of open brackets,
        # i.e. instead of an open bracket a sentence end symbol was generated.
        sent_prob += math.log(prob[-1],2)
    return pow(2,sent_prob)


def create_sent_dict(brackets_dict, num_types, prob, max_len_open):
    '''
    Create a dictionary that contains all possible valid bracket sequences that have
    either max_len_open brackets or less as keys and sentence probabilities as values.
    '''
    sent_dict = dict()
    my_brackets = list(brackets_dict.keys())[:num_types]
    start = 1
    while start <= max_len_open:
        if len(sent_dict.keys())==0:
            # sentence dictionary is empty => build all valid sentences that contain only
            # one open bracket, i.e. of length 2.
            sent_dict[start] = dict()
            for open_bracket in my_brackets:
                cur_sent = (open_bracket,brackets_dict[open_bracket])
                sent_dict[start][cur_sent] = calculate_prob(my_brackets, cur_sent, num_types, prob, start)
        else:
            # take an existing sentence dicionary and create a new one based on it
            sent_dict[start] = dict()
            for sent in sent_dict[start-1].keys():
                # copy the sentences from the old dictionary, calculate new probabilities
                sent_dict[start][sent] = calculate_prob(my_brackets, sent, num_types, prob, start)
                if len(sent)==start:
                    # concatenate the existing sentence with itself to build a new one, e.g. ( ) -> ( ) ( )
                    cur_sent = sent+sent
                    # add it to the new dictionary, calculate probability
                    sent_dict[start][cur_sent] = calculate_prob(my_brackets, cur_sent, num_types, prob, start)
            # sentences in an old dictionary can be concatenated with the ones in the new dictionary
            temp = dict()
            for tpl in sent_dict[start-1].keys():
                for sent in sent_dict[start].keys():
                    n_sent = tpl+sent
                    if len(n_sent)/2 <= start and not n_sent in sent_dict[start].keys():
                        temp[n_sent] = calculate_prob(my_brackets, n_sent, num_types, prob, start)
                    t_sent = sent+tpl
                    if len(t_sent)/2 <= start and not t_sent in sent_dict[start].keys():
                        temp[t_sent] = calculate_prob(my_brackets, t_sent, num_types, prob, start)
            sent_dict[start].update(temp)
            # just take a sentence from an old dictionary and add a couple of brackets more
            # do it for all possible bracket types
            for open_bracket in my_brackets:
                for tpl in sent_dict[start-1].keys():
                    # at the front
                    cur_sent_front = (open_bracket,)+(brackets_dict[open_bracket],)+tpl
                    if not cur_sent_front in sent_dict[start].keys():
                        sent_dict[start][cur_sent_front] = calculate_prob(my_brackets, cur_sent_front, num_types, prob, start)
                    # at the back
                    cur_sent_back = tpl+(open_bracket,)+(brackets_dict[open_bracket],)
                    if not cur_sent_back in sent_dict[start].keys():
                        sent_dict[start][cur_sent_back] = calculate_prob(my_brackets, cur_sent_back, num_types, prob, start)
                    # around
                    cur_sent_around = (open_bracket,)+tpl+(brackets_dict[open_bracket],)
                    if not cur_sent_around in sent_dict[start].keys():
                        sent_dict[start][cur_sent_around] = calculate_prob(my_brackets, cur_sent_around, num_types, prob, start)                
        start += 1
    # normalize sentence probabilities so that they sum up to 1
    s = sum(sent_dict[max_len_open].values())
    for k,v in sent_dict[max_len_open].items():
        sent_dict[max_len_open][k] = float(v)/s
    return sent_dict[max_len_open] #, sum(sent_dict[max_len_open].values()), len(sent_dict[max_len_open].keys())


def generate_text_with_ppl(name, sent_dict, num_sent):
    with open(name, "w") as f:
        sent_dict = OrderedDict(sorted(sent_dict.items(), key=lambda t: t[0]))
        prob = list(sent_dict.values())
        l = list(sent_dict.keys())
        text_prob = 0
        text_length = 0
        while num_sent>0:
            # choose a sentence according to its probability
            curr_sent = np.random.choice(l, 1, p=prob)[0]
            text_prob += math.log(sent_dict[curr_sent],2)
            text_length += len(curr_sent)+1
            num_sent -= 1
            curr = " ".join(curr_sent)+"\n"
            f.write(curr)
        # return estimated perplexity of the generated text
        return pow(2, -(text_prob/text_length))


def read_ppl(file_test, file_train, offset, counter_step):
    '''
    Read perplexity results based on RNN model from file.
    '''
    ind = []
    ppl_test = []
    ppl_train = []
    with open(file_test, 'r') as f, open(file_train, 'r') as g:
        counter = offset
        for line in f:
            ind.append(counter)
            counter += counter_step
            ppl_test.append(float(line.strip()))
        for line in g:
            ppl_train.append(float(line.strip()))
    return ind, ppl_test, ppl_train


def calc_baseline_ppl(max_len, p):
    '''
    Calculate baseline perplexity for a text consisting of sentences that contain
    only one type of brackets. Maximal number of open brackets in a sentence = max_len
    (less is possible), p is a probability of an open bracket. Length of a sentence is
    defined by a number of brackets in it + </s> symbol (end of sentence).

    !!! Probably to be changed later to include various types of brackets and only certain lengths !!!
    '''
    num_symb_len_max = sum([(2*max_len+1)*p**(max_len-1)*(1-p)**k for k in range(0,max_len)])
    num_symb_len_k = sum([(2*k+1)*p**(k-1)*(1-p)**(k+1) for k in range(1,max_len)])
    denominator = num_symb_len_max + num_symb_len_k
    part1 = sum([p**(max_len-1)*(1-p)**k*math.log(p**(max_len-1)*(1-p)**k,2) for k in range(0,max_len)])
    part2 = sum(p**(k-1)*(1-p)**(k+1)*math.log(p**(k-1)*(1-p)**(k+1),2) for k in range(1,max_len))
    numerator = -(part1+part2)
    ppl = pow(2,numerator/denominator)
    return ppl
