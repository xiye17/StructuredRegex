# StructuredRegex
Data and Code for paper **Benchmarking Multimodal Regex Synthesis with Complex Structures** . 

(It is currently a preliminary release. Code for **HIT Generation and Figure Generation** comming soon. Please stay tuned if you're interested.)

## Data
We provide *raw* data, *tokenized* data, and data with *anonymized* const strings.

Natural Language Descriptions of *raw* version contain the original raw annotations from Turkers. In *tokenized version*, we preprocessed and tokenized the descriptions. In *anonymized* version, we further replaced the contants mentioned in the descriptions with anonymous symbols. E.g., given the NL-regex pair [*it must contain the string "ABC".*  -->  contain(\<ABC\>)], we replace *"ABC"* with symbol *const0* in both NL and regex, and hence the const-anonymized NL-regex pair should be [*it must contain the string const0.*  -->  contain(const0)].

All data is presented in **TSV** format, with fields including：

* problem_id -- unique ID of the target regex.
* description -- Turker annotated description.
* regex -- target regex. 
* pos_examples -- positive examples.
* neg_examples -- negative examples.
* const_values -- mapping from symbols to the real string values, only existing in anonymized version. 

## Code (Preliminary Release)
#### Requirements
* pytroch > 1.0.0

We've attached pretrained checkpoints in `code/checkpoints/pretrained.tar`, which is ready to use. You can also reproduce the experimental results following the steps below (Please execute the commands in `code` directory)

**Train** 

`python train.py StReg --model_id <model_id>`. The models will be stored in `checkpoints/StReg` directory with names following model_id*.tar.

**Decode**

`python decode.py StReg <model_id> --split test*`. The derivations will be generated using the `checkpoints/StReg/<model_id>.tar` and be outputed to `decodes/StReg/` directory.

**Evaluate**

`python eval.py StReg <decode_id> --split test*`. Note that we report DFA accuracy (refer to the paper for more details).


