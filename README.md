# AtakapaMachineTranslation
## This project is subject to updates

This project is an attempt to create a Multi-agent system to help with machine translation of a now dormant, low-resource language isolate (Atakapa).
Atakapa poses a number of challenges:
1. It has been dormant since the early 20th century, so there are no fluent speakers of the language remaining.
2. It is poorly documented. There exists a technical description, a lexicon of roughly 2,200 words, around 500 glossed examples, and 9 longer texts. This is all that remains of the languge.
3. It is a language isolate, meaning that it is not related to any other known language, so we are unable to rely on knowledge of and models trained on related languages.
4. It is polysynthetic, meaning it has an incredibly complex morphology.

For this project, 6 SLMs were fine tuned as linguistic glossing agents trained using Atakapa glosses, embedding models served as RAG agents, trained on the grammar and lexicon, and a commercial LLM, accessed via API, to handle to final translation from the gloss to English. As well, a custom rules-based tokenizer was build to handle segmenting and pre-tokenizing the Atakapa sentences before feeding them to the SLM. The SLMs were compared against one another in terms of their effectiveness at providing glosses. Due to the resource scarcity surrounding the language, quality output is not expected to be very high. 


Overall, quality was indeed low for all models. The highest performing model was Phi-3.5 (3.8B) Reasoning model, with an overal quality of 22.5% when fed the Lexicon for Retreval Augmentation, up from 17.8% without augmentation.

Next steps include addressing the resource scarce nature of the language by generating a synthetic dataset of valid glosses to boost the amount of data available, aided by the language's fairly regular, templatic morphological nature; as well as working with larger models.


This project was inspired by LingoLLM <https://github.com/LeiLiLab/LingoLLM>

Model weights can be found at: <https://huggingface.co/avovelekastrada/Atakapa_MAS_Translation>
