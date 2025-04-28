"""
command exmaple:
TRITON_INTERPRET=1 FUSE_EXPERIMENTAL=1 DEBUG_LORA=1 TRITON_DEBUG=1 python test_lora.py 
"""


import argparse
import os
import multiprocessing as mp
from typing import Tuple

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
mp.set_start_method("fork", force=True)


def get_vllm_engine(args: argparse.Namespace) -> Tuple[SamplingParams, LLM]:
    """
    Load the LLM engine on a local machine and define sampling parameters.
    """
    sampling_params = SamplingParams(
        max_tokens=256,
        temperature=1.0,
        logprobs=1,
    )
    try:
        llm = LLM(
            model=args.base_model_name_or_path,
            # tensor_parallel_size=args.tp_size,
            # max_logprobs=args.max_logprobs,
            # enable_prefix_caching=args.enable_prefix_caching,
            enforce_eager=args.enforce_eager,
            max_model_len=args.max_model_len,
            max_seq_len_to_capture=args.max_seq_len_to_capture,
            max_num_seqs=args.max_num_seqs,
            gpu_memory_utilization=args.gpu_memory_utilization,
            # distributed_executor_backend="mp",
            enable_lora=True,
        )
    except:
        raise NotImplementedError()
        print("--> get_vllm_engine(): LLM engine initialized with no prefix caching.")
    return sampling_params, llm


if __name__ == "__main__":
    args = argparse.Namespace(
        base_model_name_or_path="meta-llama/Llama-2-7b-hf",
        tp_size=1,
        max_logprobs=1,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=2048,
        max_seq_len_to_capture=2048,
        max_num_seqs=16,
        gpu_memory_utilization=0.80,
        max_loras=4,
    )
    sampling_params, llm = get_vllm_engine(args)
    lora_request = LoRARequest(
        # "lora1", 1, lora_path="yard1/llama-2-7b-sql-lora-test"
        "lora1", 1, lora_path="jjssuh/llama-2-7b-subpop"
    )
    print(f"--> get_vllm_engine(): LLM engine initialized.")

    prompts = ["Hello, how are you doing today?", "What is the capital of France?", "Can you tell me a joke?", "What is the weather like today?"]
    output = llm.generate(prompts, sampling_params, lora_request=lora_request)

    import pdb; pdb.set_trace()





# # SPDX-License-Identifier: Apache-2.0
# """
# This example shows how to use the multi-LoRA functionality
# for offline inference.

# Requires HuggingFace credentials for access to Llama2.
# """

# import os
# import multiprocessing as mp
# from typing import Optional

# from vllm import SamplingParams, LLM
# from vllm.lora.request import LoRARequest

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def create_test_prompts(
#         lora_path: str
# ) -> list[tuple[str, SamplingParams, Optional[LoRARequest]]]:
#     """Create a list of test prompts with their sampling parameters.

#     2 requests for base model, 4 requests for the LoRA. We define 2
#     different LoRA adapters (using the same model for demo purposes).
#     Since we also set `max_loras=1`, the expectation is that the requests
#     with the second LoRA adapter will be ran after all requests with the
#     first adapter have finished.
#     """
#     return [
#         ("A robot may not injure a human being",
#          SamplingParams(temperature=0.0,
#                         logprobs=1,
#                         prompt_logprobs=1,
#                         max_tokens=128), None),
#         ("To be or not to be,",
#          SamplingParams(temperature=0.8,
#                         top_k=5,
#                         presence_penalty=0.2,
#                         max_tokens=128), None),
#         (
#             "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
#             SamplingParams(temperature=0.0,
#                            logprobs=1,
#                            prompt_logprobs=1,
#                            max_tokens=128,
#                            stop_token_ids=[32003]),
#             LoRARequest("sql-lora", 1, lora_path)),
#         (
#             "[user] Write a SQL query to answer the question based on the table schema.\n\n context: CREATE TABLE table_name_74 (icao VARCHAR, airport VARCHAR)\n\n question: Name the ICAO for lilongwe international airport [/user] [assistant]",  # noqa: E501
#             SamplingParams(temperature=0.0,
#                            logprobs=1,
#                            prompt_logprobs=1,
#                            max_tokens=128,
#                            stop_token_ids=[32003]),
#             LoRARequest("sql-lora2", 2, lora_path)),
#     ]


# def process_requests(engine: LLM,
#                      test_prompts: list[tuple[str, SamplingParams,
#                                               Optional[LoRARequest]]]):
#     """Continuously process a list of prompts and handle the outputs."""

#     print("-" * 50)
#     while test_prompts:
#         if test_prompts:
#             prompt, sampling_params, lora_request = test_prompts.pop(0)
#             output = engine.generate(
#                 prompt,
#                 sampling_params,
#                 lora_request=lora_request,
#             )
#             print(output[0].outputs[0].text)


# def initialize_engine() -> LLM:
#     """Initialize the LLMEngine."""
#     # max_loras: controls the number of LoRAs that can be used in the same
#     #   batch. Larger numbers will cause higher memory usage, as each LoRA
#     #   slot requires its own preallocated tensor.
#     # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
#     #   numbers will cause higher memory usage. If you know that all LoRAs will
#     #   use the same rank, it is recommended to set this as low as possible.
#     # max_cpu_loras: controls the size of the CPU LoRA cache.
#     # engine_args = EngineArgs(model="meta-llama/Llama-2-7b-hf",
#     #                          enable_lora=True,
#     #                          max_loras=1,
#     #                          max_lora_rank=8,
#     #                          max_cpu_loras=2,
#     #                          max_num_seqs=256)
#     # return LLMEngine.from_engine_args(engine_args)

#     return LLM(
#         model="meta-llama/Llama-2-7b-hf",
#         # enforce_eager=args.enforce_eager,
#         # max_model_len=args.max_model_len,
#         # max_seq_len_to_capture=args.max_seq_len_to_capture,
#         max_num_seqs=256,
#         gpu_memory_utilization=0.9,
#         enforce_eager=True,
#         enable_lora=True,
#         max_loras=4,
#     )


# def main():
#     """Main function that sets up and runs the prompt processing."""
#     os.environ['FUSE_EXPERIMENTAL'] = "1"
#     os.environ['TRITON_INTERPRET'] = "1"
#     mp.set_start_method("fork", force=True)
#     print("FUSE_EXPERIMENTAL:", os.environ['FUSE_EXPERIMENTAL'])
#     print("TRITON_INTERPRET:", os.environ['TRITON_INTERPRET'])
#     engine = initialize_engine()
#     lora_path = "yard1/llama-2-7b-sql-lora-test"
#     # mp.set_start_method("fork", force=True)
#     test_prompts = create_test_prompts(lora_path)
#     process_requests(engine, test_prompts)


# if __name__ == '__main__':
#     main()