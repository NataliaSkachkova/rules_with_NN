import sys
sys.path.insert(0, 'C:/Users/Natalia/Desktop/BA/Py Scripts')

from utils import *
import matplotlib.pyplot as plt

def plot_RNN_ppl(base_ppl):
    fig = plt.figure(figsize=(25, 15))

    plt.subplot(2,2,1)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.all.size17.diff.hidd.txt", "data/ppl.train.all.size17.diff.hidd.txt", 0, 1)
    plt.plot(n, ppl_train, '-D', markevery=[8], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[8])
    plt.plot(n, [base_ppl for x in n], label="True PPL")
    #plt.annotate("Min train PPL: "+str(min(ppl_train))+"\nMin test PPL: "+str(min(ppl_test)),xy=(8,2.129925),xytext=(8, 6.4), rotation=45)
    plt.title("Estimating PPL with RNNLM: sentences contain round brackets, \nmax number of open brackets is 4, text size=2^17")
    plt.legend()
    plt.xlabel("Hidden layer size (powers of 2)")
    plt.ylabel("PPL net")

    plt.subplot(2,2,2)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.all.size17.hidd64.diff.bptt.txt", "data/ppl.train.all.size17.hidd64.diff.bptt.txt", 0, 1)
    plt.plot(n, ppl_train, '-D', markevery=[72], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[72])
    plt.plot(n, [base_ppl for x in n], label="True PPL")
    #plt.annotate("Min train PPL: "+str(min(ppl_train))+"\nMin test PPL: "+str(min(ppl_test)),xy=(72,2.129925),xytext=(72, 1.142), rotation=45)
    plt.title("Estimating PPL with RNNLM: sentences contain round brackets, \nmax number of open brackets is 4, text size=2^17, hidden layer size=64")
    plt.legend()
    plt.xlabel("BPTT step")
    plt.ylabel("PPL net")
    
    plt.subplot(2,2,3)
    n, ppl_test, ppl_train = read_ppl("data/ppl.test.all.size17.hidd64.bptt72.diff.cls.txt", "data/ppl.train.all.size17.hidd64.bptt72.diff.cls.txt", 1, 1)
    plt.plot(n, ppl_train, '-D', markevery=[72], label="RNN estimated PPL of train data")
    plt.plot(n, ppl_test, '-D', label="RNN estimated PPL of test data", markevery=[72])
    plt.plot(n, [base_ppl for x in n], label="True PPL")
    #plt.annotate("Min train PPL: "+str(min(ppl_train))+"\nMin test PPL: "+str(min(ppl_test)),xy=(72,1.129925),xytext=(72, 1.1425), rotation=45)
    #plt.annotate("Min test PPL: "+str(min(ppl_test)),xy=(100,1.129591),xytext=(100, 1.142), rotation=45)
    plt.title("Estimating PPL with RNNLM: sentences contain round brackets, \nmax number of open brackets is 4, text size=2^17, hidden layer size=64, bptt=72")
    plt.legend()
    plt.xlabel("Number of classes")
    plt.ylabel("PPL net")

    fig.savefig('RNN_round_max_open_4_ppl.png', bbox_inches='tight')


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

    fig.savefig('RNN_round_max_open_4_probs.png', bbox_inches='tight')
    #plt.show()

def main():
    print("Creating sentence dictionary with sentences consisting of round brackets.")
    print("Maximal number of open brackets is 4, probability of an open bracket is 0.5.")
    sent_dict = create_sent_dict(brackets, 1, [0.5,0.5], 4)
    print()
    print("Training data: 131,072 sentences, perplexity estimated from text: 1.4770440991905749")
    print("Test data: 10,000 sentences, perplexity estimated from text: 1.4781559451539232")
    baseline = calc_baseline_ppl(4, 0.5)
    print("Baseline perplexity:", baseline)
    plot_RNN_ppl(baseline)
    print()
    print("Best RNN model: text size = 131,072, hidden layer size = 64, bptt = 72, number of classes = 73.")
    print("Best perplexity result on test data: 1.479702.")
    ro, rc, eos = read_prob("data/test.ppl.result.10000.round.4.debug.txt")
    plot_probs(ro, rc, eos)
          
    
if __name__ == '__main__':
    main()
