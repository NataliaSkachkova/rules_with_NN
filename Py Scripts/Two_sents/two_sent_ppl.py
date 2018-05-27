import math
import numpy as np
import matplotlib.pyplot as plt

def gen_text(name, p, num_sents):
    '''
    Generate a text consisting of sentences of only two types
    where each sentence type has certain probability.
    '''
    with open(name, 'w')as f:
        sent1 = "( ( ) )"
        sent2 = "( ) ( )"
        while num_sents>0:
            curr = np.random.choice([sent1,sent2], 1, p=[p,1-p])[0]
            num_sents -= 1
            f.write(curr+"\n")


def calc_baseline_ppl(p):
    '''
    Calculate perplexity of a text consisting of sentences of two types,
    where one sentence has probability p and the other one 1-p.
    Both sentences have length 4 and contain 2 open and 2 closed brackets.
    '''
    return p**(-p/4)*(1-p)**(-(1-p)/4)


def calc_text_ppl(file, p):
    '''
    Estimate sentence probabilities and perplexity from text.
    '''
    with open(file, 'r') as f:
        text_prob = 0
        text_len = 0
        sent1 = 0
        sent2 = 0
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line=='( ( ) )':
                text_prob += math.log(p,2)
                text_len += 4
                sent1 += 1
            else:
                text_prob += math.log(1-p,2)
                text_len += 4
                sent2 += 1
        print("Sentences probabilities estimated from text:", "( ( ) ) ", sent1/len(lines), "( ) ( ) ", sent2/len(lines))
        ppl = pow(2,-(text_prob/text_len))
        print("Text perplexity:", ppl)
        return ppl


def read_ppl(file_test, file_train, offset, counter_step):
    '''
    Read perplexity results based on RNN model.
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


def plot_RNN_ppl():
    fig = plt.figure(figsize=(20, 10))

    plt.subplot(2,2,1)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.size.17.diff.hd.txt", "data/ppl.train.size.17.diff.hd.txt", 0, 1)
    plt.plot(n, ppl_train, '-D', markevery=[3], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[3])
    plt.plot(n, [1.164994161612539 for x in n], label="True PPL")
    plt.annotate("Min train PPL: "+str(min(ppl_train))+"\nMin test PPL: "+str(min(ppl_test)),xy=(3,1.129925),xytext=(3, 5.4), rotation=45)
    plt.title("Estimating PPL with RNNLM: 2 sentence types of length 4, text size=2^17")
    plt.legend()
    plt.xlabel("Hidden layer size (powers of 2)")
    plt.ylabel("PPL net")

    plt.subplot(2,2,2)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.size.17.hd.16.diff.bptt.txt", "data/ppl.train.size.17.hd.16.diff.bptt.txt", 0, 1)
    plt.plot(n, ppl_train, '-D', markevery=[4], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[4])
    plt.plot(n, [1.164994161612539 for x in n], label="True PPL")
    plt.annotate("Min train PPL: "+str(min(ppl_train))+"\nMin test PPL: "+str(min(ppl_test)),xy=(4,1.129925),xytext=(4, 1.142), rotation=45)
    plt.title("Estimating PPL with RNNLM: 2 sentence types of length 4, text size=2^17, hidden layer size=16")
    plt.legend()
    plt.xlabel("BPTT step")
    plt.ylabel("PPL net")
    
    plt.subplot(2,2,3)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.size.17.hd.16.bptt.4.diff.cls.txt", "data/ppl.train.size.17.hd.16.bptt.4.diff.cls.txt", 1, 1)
    plt.plot(n, ppl_train, '-D', markevery=[67], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[100])
    plt.plot(n, [1.164994161612539 for x in n], label="True PPL")
    plt.annotate("Min train PPL: "+str(min(ppl_train)),xy=(67,1.129925),xytext=(67, 1.1425), rotation=45)
    plt.annotate("Min test PPL: "+str(min(ppl_test)),xy=(100,1.129591),xytext=(100, 1.142), rotation=45)
    plt.title("Estimating PPL with RNNLM: 2 sentence types of length 4, text size=2^17, hidden layer size=16, bptt=4")
    plt.legend()
    plt.xlabel("Number of classes")
    plt.ylabel("PPL net")

    fig.savefig('RNN_2sents_ppl.png', bbox_inches='tight')
    #plt.show()


def read_prob(file):
    '''
    Read probabilities of separate brackets based on RNN model.
    '''
    round_open = []
    round_closed = []
    end_of_sent = []
    with open(file, 'r') as f:
        for line in f:
            l = line.split()
            bracket = l[2].strip()
            if bracket=="(":
                round_open.append(float(l[1].strip()))
            elif bracket==")":
                round_closed.append(float(l[1].strip()))
            else:
                end_of_sent.append(float(l[1].strip()))
    return round_open, round_closed, end_of_sent


def plot_probs(ro, rc, eos):
    fig = plt.figure(figsize=(20, 8))
    plt.subplot(1,3,1)
    plt.hist(ro, bins=100)
    plt.title("Probabilities of round open brackets (")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1,3,2)
    plt.hist(rc, bins=100)
    plt.title("Probabilities of round closed brackets )")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    plt.subplot(1,3,3)
    plt.hist(eos, bins=100)
    plt.title("Probabilities of <\s> sign")
    plt.xlabel("Value")
    plt.ylabel("Frequency")

    fig.savefig('RNN_2sents_probs.png', bbox_inches='tight')
    #plt.show()

def main():
    print("All texts contain only two types of sentences: either \"( ( ) )\" or \"( ) ( )\".")
    print("Baseline perplexity (PPL) for any text where sentence \"( ( ) )\" occurs with probability 0.3:", calc_baseline_ppl(0.3))
    print("Estimated PPL for train data of size 131,072:")
    calc_text_ppl("data/train.17.2sents.txt", 0.3)
    print("Estimated PPL for test data of size 10,000:")
    calc_text_ppl("data/test.10000.2sents.txt", 0.3)
    print("Estimating PPL with RNNs. Finding optimal parameters:")
    plot_RNN_ppl()
    print("Best RNN model: text size=131,072, hidden layer size=16, bptt=4, number of classes = 101.")
    print("Best perplexity results on test data: 1.129591.")
    #print("Probabilities of separate words (brackets) based on the RNN model with these parameters:")
    ro, rc, eos = read_prob("data/test.ppl.result.10000.2sents.debug.txt")
    plot_probs(ro, rc, eos)


if __name__ == '__main__':
    main()
