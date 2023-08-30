# CSC6203: Selected Topics in CS III (Large Language Models)

## Basic info of the course
- instructor: Benyou Wang (wangbenyou@cuhk.edu.cn)
- lectures: Friday, 13:30 - 16:30
- WeChat Group: 
- Contact: If you have any questions about the course, contact us at wangbenyou@cuhk.edu.cn


Large language models (LLMs) have utterly transformed the field of natural language processing (NLP) in the last 3-4 years. They form the basis of state-of-art systems and become ubiquitous in solving a wide range of natural language understanding and generation tasks. With the unprecedented potential and capabilities, these models also give rise to new ethical and scalability challenges. This course aims to cover cutting-edge research topics centering around pre-trained language models. We will discuss their technical foundations (BERT, GPT, T5 models, mixture-of-expert models, retrieval-based models), emerging capabilities (knowledge, reasoning, few-shot learning, in-context learning), fine-tuning and adaptation, system design, as well as security and ethics. We will cover each topic and discuss important papers in depth. Students will be expected to routinely read and present research papers and complete a research project at the end.

This is an advanced graduate course and all the students are expected to have taken machine learning and NLP courses before and are familiar with deep learning models such as Transformers.

## Learning goals:
This course is intended to prepare you for performing cutting-edge research in natural language processing, especially topics related to pre-trained language models. We will discuss the state-of-the-art, their capabilities and limitations.
Practice your research skills, including reading research papers, conducting literature survey, oral presentations, as well as providing constructive feedback.
Gain hands-on experience through the final project, from brainstorming ideas to implementation and empirical evaluation and writing the final paper.


## Course Outline

| Date      | Topics                                                                                     | Recommended Reading    | Pre-Lecture Questions            | Lecture Note     | Events Deadlines | Feedback Providers |
|-----------|---------------------------------------------------------------------------------------------------|---------------------------------------------------|----------------------------------|------------------|------------------|--------------------|
| Sep. 8th  | Introduction to Large Language Models (LLMs)                     | [Open AI's blog](https://openai.com/blog/chatgpt)  [On the Opportunities and Risks of Foundation Models](https://arxiv.org/abs/2108.07258) [Sparks of Artificial General Intelligence: Early experiments with GPT-4](https://arxiv.org/abs/2303.12712) | What is ChatGPT and how to use?  | [slide]() [note]()| Assignment       | Xidong Wang        |
| Sep. 8th  | Language models and beyond  | [A Neural Probabilistic Language Model](A Neural Probabilistic Language Model)  [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | What is language model and why is it important?  | [slide]() [note]()| Assignment       | Xidong Wang        |
| TBD       | Architecture engineering and scaling law: Transformer and beyond                                   | [nanoGPT GitHub](https://github.com/karpathy/nanoGPT) [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [HuggingFace's course on Transformers](https://huggingface.co/learn/nlp-course/chapter1/1) [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)  [The Transformer Family Version 2.0](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/)   [On Position Embeddings in BERT](https://openreview.net/forum?id=onxoVA9FxMw) | Why does Transformer become the backbone of PLMs? | [slide]() [note]() | TBD              | TBD                |
| TBD       | Training LLMs from scratch                                                                        | [LLMZoo](https://github.com/FreedomIntelligence/LLMZoo), [LLMFactory](https://github.com/FreedomIntelligence/LLMFactory) [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)  [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)  [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)  | How to train LLMs from scratch?      | [slide]() [note]()| TBD              | TBD                |
| TBD       | Efficiency in LLM                                                                                 | [llama2.c GitHub](https://github.com/karpathy/llama2.c)  [Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)   [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)  [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/pdf/2101.03961.pdf)  [Towards a Unified View of Parameter-Efficient Transfer Learning](Towards a Unified View of Parameter-Efficient Transfer Learning)  | how to make LLMs train/inference faster?    | [slide]() [note]() | TBD              | TBD                |
| TBD       | Prompt engineering                                                                                | [Best practices for prompt engineering with OpenAI API](https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-openai-api)    [prompt engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)     | What is in-context learning and how to better prompt LLMs?    | [slide]() [note]()| TBD              | TBD                |
| TBD       | Mid review of final project                                                                       | N/A                                                  |  N/A    | [slide]() [note]()| TBD              | TBD                |
| TBD       | Knowledge and Reasoning                                                                           | MMLU/ C-eval, knowledge injection, RAG/ mathmatical reasoning, [Natural Language Reasoning, A Survey](https://arxiv.org/abs/2303.14725)         | Can LLMs reason?  | [slide]() [note]()| TBD              | TBD                |
| TBD       | Multimodal LLMs                                                                                   | [CLIP, Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/pdf/2103.00020.pdf)  [MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models](https://arxiv.org/abs/2304.10592)         [Stable Diffusion](https://stablediffusionweb.com/)                               | Can LLMs see?       | [slide]() [note]()| TBD              | TBD                |
| TBD       | LLMs in vertical domains                                                                          | [Large Language Models Encode Clinical Knowledge](https://arxiv.org/abs/2212.13138) [Capabilities of GPT-4 on Medical Challenge Problems](https://arxiv.org/abs/2303.13375) [Performance of ChatGPT on USMLE: Potential for AI-assisted medical education using large language models](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000198) [HuatuoGPT](https://github.com/FreedomIntelligence/HuatuoGPT), [Medical-NLP](https://github.com/FreedomIntelligence/Medical_NLP), ChatLaw | Can LLMs be a mature doctors/lawyer? | [slide]() [note]()| TBD              | TBD                |
| TBD       | Tools and Large language models                                                                   |[ToolBench](https://github.com/OpenBMB/ToolBench), [AgentBench](https://arxiv.org/abs/2308.03688)  [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/pdf/2005.11401.pdf)  [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)      | How are LLMs used in planning?  | [slide]() [note]()| TBD              | TBD                |
| TBD       | Privacy, bias, fairness, Toxicity and Holistic Evaluation                                         |   [On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?](https://dl.acm.org/doi/pdf/10.1145/3442188.3445922) [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) [Extracting Training Data from Large Language Models](https://arxiv.org/pdf/2012.07805.pdf)                                           |In which aspects should we evaluate LLMs? | [slide]() [note]()| TBD            | TBD                |
| TBD       | Alignment, Limitations, and broader Impact                                                                         | [Superalignment](https://openai.com/blog/introducing-superalignment)     [GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models](https://arxiv.org/abs/2303.10130)    [ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks](https://arxiv.org/abs/2303.15056)    [Theory of Mind Might Have Spontaneously Emerged in Large Language Models](https://arxiv.org/abs/2302.02083)                                    | What are LLMs' limitations?     | [slide]() [note]()| TBD              | TBD                |
| TBD       | Guest lecture  | N/A                                               | TBD                              | [slide]() [note]()| TBD              | TBD                |  --> 
| TBD       | In-class presentation (extended class)                                                             | N/A                                                  | how to solve real-world problems using LLMs                              | [slide]() [note]()| TBD              | TBD                |


## Teaching Assistant

- Xidong Wang (Leading TA)
- Fei Yu
- Zhengyang Tang
- Junying Chen
- Juhao Liang
- Zhuoheng Ma

##  Course Project

All projects will be archived  and open-sourced. Please specify reasons if you decide not to do so.

### Default Projects

### Customized Projects

### Sponsored Projects
[Call for Sponsorship]()

## Awards
- Best Research Award
- Best Presentation Award
- Best Poster Award
- Promising Research Award
- TA favorites

> Anyone who gets one of the above awards would likely get expected recomendations in any occasion. 

## Problems unsolved



## Bonus

- Anyone who releases a public LLM-related paper with permission from his/her supervisors could get 5 bonus marks (total marks should not exceed 100).



## References
[COS 597G (Fall 2022): Understanding Large Language Models by Danqi](https://www.cs.princeton.edu/courses/archive/fall22/cos597G/)
[CS25: Transformers United V2](https://web.stanford.edu/class/cs25/)
[CS324 - Large Language Models](https://stanford-cs324.github.io/winter2022/calendar/)




