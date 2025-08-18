## Important:

The `main` branch contains the original implementation of the DeepSilencer architecture. However, the accuracy and other metrics, obtained in the `unseen` test data is quite low (as opposed to that proposed in the paper). Due to this, several new techniques (inspired from literature and other related papers) were tried out in an attempt to improve the performance of the model.

This repo contains 2 other branches apart from `main`, namely `saranya` and `attsioff_deepsl`.

#### saranya branch:

This branch incorporates **Positional Encodings** into the architecture. Positional Encodings are an integral part and has been used in other architectures as well (as mentioned in papers like AttSioff, Oligoformer etc). However, the DeepSillencer architecture did not have this in this methodology. Currently, only the Sinusoidal PE have been tried here. Other variants (RoPE, Alibi etc) could be tried and thus remains as a future work.

Another major drawback is that only the siRNA sequence is taken as input, not the mRNA sequence. Changes were also made in the architecture so that both the mRNA and siRNA sequence is taken as input (this part of the architecture is also inspired from AttSioff).
 
After including the PE, the performance improved slightly (57% -> 61%). Despite the improvement, the overall performance is still quite less...


#### attsioff_deepsl branch:

As the name of this branch suggests, this repo contains many modules which are directly copied from the AttSioff implementation. These include the Biological features such as GC content etc. The original paper did not include these biological features which might have made convergence difficult. This part of the project tries to solve that in some way.