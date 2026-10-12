import os
import re
import ast
import time
import random
import json
import torch
import uuid
from pydantic import BaseModel, Field
from typing import Optional, List, Union
from swarms import Agent
import openai
from openai import OpenAI, RateLimitError, APIError
from typing import Any
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from agents.prompt import *

def string_to_dict(thought_string):
    return eval(thought_string)


class Caller:
    def __init__(
        self, 
        system_prompt: str=None, 
        base_model: BaseModel=None, 
        model_name: str="gpt-3.5-turbo", 
        parallel_tool_calls: bool=False, 
        max_tokens: int=128, 
        temperature: float=0.4,
        client: OpenAI=None, 
        top_p: float=0.9
    ):
        super().__init__()
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.base_model = base_model
        self.parallel_tool_calls = parallel_tool_calls
        self.client = client
        self.top_p = top_p
        if base_model is not None:
            tool_raw = openai.pydantic_function_tool(base_model)
            if "function" in tool_raw:
                tool = tool_raw["function"].copy()
                # 可以增加 description 基于 Pydantic docstring
                if "description" not in tool:
                    tool["description"] = base_model.__doc__ or ""
                # 移除 strict 参数，因为它应该在 tools 级别而不是函数级别
                if "strict" in tool:
                    del tool["strict"]
            else:
                tool = tool_raw  # 已是正确顶层格式
            tool = [tool]
        else:
            tool = None
        self.tools = tool
        self.add = []
        self.question = ""

    def clear_hint(self):
        self.add = []

    def run(self, task: str, question: str, prompt: Optional[List[str]], prev_answer: Optional[List[str]]=None, stream: bool=True, retries: int=4, *args, **kwargs) -> dict:
        task += f"Question:{question}\n"

        if prev_answer is not None:
            prev_answers = ""
            for i, ans in enumerate(prev_answer):
                prev_answers += f"'Prev Answer_{i}': '{ans}'\n"
            task = prev_answers+task
        
        messages=[
            {"role":"system", "content":self.system_prompt}
        ]

        if prompt:
            for add in prompt:
                messages.append({
                    "role": "user",
                    "content": f"hint: {add}"
                })

        messages.append({"role":"user", "content":task})
        messages.append({
            "role": "system",
            "content": "now generate an answer as following json format:"+
                '{"thought":"xxx", "Answer": "xxx"}'
        })

        response = self._call_once(messages, stream)
        content, function_call, usage = "", None, None

        for chunk in response:
            if hasattr(chunk, "usage") and not chunk.choices:
                usage = chunk.usage
                continue

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta.function_call:
                if function_call is None:
                    function_call = {"name": delta.function_call.name, "arguments": ""}
                function_call["arguments"] += delta.function_call.arguments or ""
            elif delta.content:
                content += delta.content

        if function_call:
            func_args = json.loads(function_call["arguments"])
            # 将 function_call 添加到对话中
            messages.append({
                "role": "assistant",
                "content": "",
                "function_call": function_call
            })
            # 运行函数并添加返回值到对话
            result = self.base_model(**func_args).dict() if self.base_model else func_args
            messages.append({
                "role": "function",
                "name": function_call["name"],
                "content": json.dumps(result)
            })
            # 再次调用 API 得到最终内容
            messages.append({
                "role": "system",
                "content": "now generate a final answer as following json format:"+
                    '{"thought":"xxx", "Answer": "xxx"}'
            })

            attempt = 0
            while True:
                try:
                    final_resp = self._call_final(messages)
                    break # if success then quit the loop
                except RateLimitError as e:
                    if attempt>=retries:
                        raise RuntimeError(f"Rage limit reached after {retries} attempts for final.") from e
                    delay = min(60, 2 ** attempt + random.random())
                    print(f"[Caller] Rate limit (429), retry in {delay:.1f}s (attempt {attempt+1}/{retries})")
                    time.sleep(delay)
                    attempt += 1
                except APIError as e:
                    if attempt >= retries:
                        raise
                    delay = 2 ** attempt + random.random()
                    print(f"[Caller] API error: {e}. Retry in {delay:.1f}s")
                    time.sleep(delay)
                    attempt += 1
            
            content = final_resp.choices[0].message.content

        return content
        # 如果使用 base_model，可返回结构化对象
        # if self.base_model:
        #     decoder = json.JSONDecoder()
        #     idx = 0
        #     results = []
        #     while idx < len(content):
        #         content = content.lstrip()
        #         try:
        #             obj, offset = decoder.raw_decode(content, idx)
        #             breakpoint()
        #         except json.JSONDecodeError:
        #             break
        #         results.append(self.base_model.parse_obj(obj))
        #         idx = offset
        #     return results
        # return {"content": content, "usage": usage}

    def _call_once(self, messages, stream=True):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            functions=self.tools,
            function_call="auto" if self.tools else None,
            stream=stream,
        )
        return response

    def _call_final(self, messages):
        final_resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=False
        )
        return final_resp


class Thought(BaseModel):
    thought: str
    Answer: str


class QwenModel:
    def __init__(self, model_name="/home/lj/tlx/Qwen2.5-VL-7B-Instruct", device=None):
        self.device = device if device is not None else "cuda"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(model_name)
        _ = self.model.generate(past_key_values=None, max_new_tokens=1, input_ids=torch.ones(1,1).to(torch.int32).to("cuda:0"))
    
    def generate(self, messages, max_new_tokens=128, **gen_kwargs):
        chat = [
            {
                "role":"user", 
                "content": [
                    {"type": "text", "text": messages}
                ]
            }
        ]
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        outs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed = [out[len(i):] for i, out in zip(inputs.input_ids, outs)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    
    def run(self, task=None, *args, **kwargs):
        text = None
        if task is not None:
            text = self.generate(task, *args, **kwargs)
        else:
            raise ValueError("No prompt/messages provided to run()")
        # 返回一个长度为1的列表，使 Agent 可以迭代
        return [text]


class TotAgent:
    """
    Represents a Tree of Thoughts (ToT) agent.

    Attributes:
        id (str): The unique identifier for the agent.
        max_loops (int): The maximum number of loops the agent can run.
        model (OpenAIFunctionCaller): The OpenAI function caller for the agent.
        agent (Agent): The agent responsible for running tasks.

    Methods:
        run(task: str) -> dict: Runs a task using the agent and returns the output as a dictionary.
    """

    def __init__(
        self,
        id: str = uuid.uuid4().hex,
        max_loops: int = None,
        use_openai_caller: bool = True,
        model_name: Optional[str] = None,
        TREE_OF_THOUGHTS_SYS_PROMPT: str = None,
        client: OpenAI=None,
        *args,
        **kwargs,
    ):
        """
        Initializes a new instance of the TotAgent class.

        Args:
            id (str, optional): The unique identifier for the agent. Defaults to a randomly generated UUID.
            max_loops (int, optional): The maximum number of loops the agent can run. Defaults to None.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.id = id
        self.max_loops = max_loops
        self.model_name = model_name

        if use_openai_caller:
            # self.model = OpenAIFunctionCaller(
            self.model = Caller(
                system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
                base_model=Thought,
                model_name=self.model_name,
                parallel_tool_calls=False,
                client=client,
                max_tokens=3000,
            )
        else:
            self.model = QwenModel(device="cuda:0")


        self.agent = Agent(
            agent_name=f"ToT-Agent-{self.id}",
            system_prompt=TREE_OF_THOUGHTS_SYS_PROMPT,
            llm=self.model,
            max_loops=1,
            autosave=True,
            dashboard=False,
            verbose=True,
            dynamic_temperature_enabled=True,
            saved_state_path=f"tot_agent{self.id}.json",
            user_name="swarms_corp",
            retry_attempts=1,
            context_length=200000,
            return_step_meta=False,
            best_of_n=1,
            output_type="string",
            *args,
            **kwargs,
        )

    def run(self, task: Any, question: str, additional_info: Optional[Union[str, List[str]]]=None, prev_answer: Optional[Union[str, List[str]]]=None) -> dict:
        """
        Runs a task using the agent and returns the output as a dictionary.

        Args:
            task (str): The task to be run by the agent.

        Returns:
            dict: The output of the agent as a dictionary.
        """
        self.model.clear_hint()
        if additional_info:
            self.model.hint(additional_info)
        
        task += f"Question:{question}\n"

        if prev_answer is not None:
            prev_answers = ""
            for i, ans in enumerate(prev_answer):
                prev_answers += f"'Prev Answer_{i}': '{ans}'\n"
            task = prev_answers+task

        agent_output = self.agent.run(task)
        breakpoint()

        if type(agent_output) is str:
            agent_output = ast.literal_eval(agent_output)
        try:
            if task in agent_output:
                split_outputs = agent_output.split(task+"\n")
                if len(split_outputs) == 2:
                    agent_output = split_outputs[-1]
            output = []
            for out in agent_output:
                if type(out) is dict:
                    output += [out]
                elif type(out) is str:
                    output += [string_to_dict(out)]
        except:
            breakpoint()
        return output

    @staticmethod
    def vote_prompt_wrap(x: str, ys: str) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1