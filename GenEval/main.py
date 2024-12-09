import pandas as pd
from distilabel.llms import OLlamaLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromHub, CombineColumns, tasks

llama8B = OLlamaLLM(model="llama3.1", host=host_llama_8b)
llama70B = OLlamaLLM(model="llama3.1:70b-instruct-q2_k", host=host_llama_70b)

with Pipeline(name="preference-datagen-llama3.1") as pipeline:
    load_dataset = LoadDataFromHub(
        name="load_dataset",
        output_mappings={"prompt": "instruction"},
    )

    # generate two responses
    generate = [
        tasks.TextGeneration(name="text_generation_8B", llm=llama8B),
        tasks.TextGeneration(name="text_generation_70B", llm=llama70B),
    ]

    # combine responses into one col
    combine = CombineColumns(
        columns=["generation", "model_name"],
        output_columns=["generations", "model_names"]
    )

    # rate responses with 405B LLM-as-a-judge
    evaluate = tasks.UltraFeedback(aspect="overall-rating", llm=llama70B)

    # define and run pipeline
    load_dataset >> generate >> combine >> evaluate
