
#!/bin/bash
> /nethome/nskachkova/RNNLM/temp/test.ppl.result.txt
> /nethome/nskachkova/RNNLM/temp/train.ppl.result.txt
> /nethome/nskachkova/RNNLM/temp/ppl.test.txt
> /nethome/nskachkova/RNNLM/temp/ppl.train.txt
#> /nethome/nskachkova/NGRAM/ppl/train.true.ppl.txt
cls=1
while true
    do
        #python gen_round_brackets_text.py 1 [0.5,0.5] 16 3 $n_sent >> /nethome/nskachkova/NGRAM/ppl/train.true.ppl.txt
        rm /nethome/nskachkova/RNNLM/models/model.txt
        rm /nethome/nskachkova/RNNLM/models/model.txt.output.txt
        /nethome/nskachkova/RNNLM/rnnlm -train /nethome/nskachkova/bin/train.17.2sents.txt -rnnlm /nethome/nskachkova/RNNLM/models/model.txt -valid /nethome/nskachkova/RNNLM/valid_data/valid.10000.2sents.txt -hidden 16 -bptt 4 -class $cls
        /nethome/nskachkova/RNNLM/rnnlm -rnnlm /nethome/nskachkova/RNNLM/models/model.txt -test /nethome/nskachkova/RNNLM/test_data/test.10000.2sents.txt > /nethome/nskachkova/RNNLM/temp/test.ppl.result.txt
        /nethome/nskachkova/RNNLM/rnnlm -rnnlm /nethome/nskachkova/RNNLM/models/model.txt -test /nethome/nskachkova/bin/train.17.2sents.txt > /nethome/nskachkova/RNNLM/temp/train.ppl.result.txt
        cls=`expr $cls + 1`
        line="`sed -n '6p' /nethome/nskachkova/RNNLM/temp/test.ppl.result.txt`"
        ppl=${line##* }
        echo $ppl >> /nethome/nskachkova/RNNLM/temp/ppl.test.txt
        line="`sed -n '6p' /nethome/nskachkova/RNNLM/temp/train.ppl.result.txt`"
        ppl=${line##* }
        echo $ppl >> /nethome/nskachkova/RNNLM/temp/ppl.train.txt
    done
