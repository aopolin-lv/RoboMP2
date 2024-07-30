import os
import json

import torch
import cv2
import numpy as np
from easydict import EasyDict

from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper

from engine_robotic import *
from utils.data_prepare import *
from utils.common_utils import create_logger

from retrieval.utils import *
from utils.data_prepare import prepare_obs

from retrieval.rewrite import *
from retrieval.sort import coarse_grained_rank, fine_grained_rank
from data_process.gptutils import gpt4v
from data_process.utils import analysis_gpt4_code, insert_task_into_prompt

def exception_handler(exception, logger, **kwargs):
    logger.error("Exception: {}".format(exception))
    task_info = {
        "task_id": kwargs["task_id"],
        "task": kwargs["whole_task"],
        "exec": kwargs["exec_codes"],
        "skip": True,
        "success": False,
        "exception": str(exception),
    }
    return task_info

class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )

class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)

@torch.no_grad()
def main(cfg, logger):
    logger.info("cfg: {}".format(cfg))
    debug_flag = cfg.debug_flag
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]
    seed = cfg.seed
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
                seed=seed,
                render_prompt=False,
                display_debug_window=debug_flag,
                hide_arm_rgb=cfg.hide_arm,
            )
        ),
        bonus_steps=2,
    )
    single_model_flag = True if cfg.prompt_modal == "single" else False
    result_folder = (
        cfg.save_dir + "/" + cfg.partition + "/" + cfg.task + "/" + cfg.prompt_modal
    )
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    eval_res_name = cfg.partition + "_" + cfg.task + ".json"
    eval_result_file_path = os.path.join(result_folder, eval_res_name)

    task_id = 0
    all_infos = []
    if cfg.reuse and os.path.exists(eval_result_file_path):
        with open(eval_result_file_path, "r") as f:
            all_infos = json.load(f)

    while True:
        env.global_seed = seed
        obs = env.reset()
        env.render()
        meta_info = env.meta_info
        prompt = env.prompt
        prompt_assets = env.prompt_assets

        # Get relevant information about the current task and environment
        obs_list = prepare_obs(obs, meta=meta_info)
        whole_task, templates, task_setting = prepare_prompt(
            prompt, prompt_assets, single_model=single_model_flag, task=cfg.task
        )

        task_id += 1
        if task_id >= 2:
            break

        logger.info(f"==================Task {task_id}=========================")
        if not single_model_flag:
            # get full task description for debug
            whole_task_debug, _, _ = prepare_prompt(
                prompt, prompt_assets, single_model=True, task=cfg.task
            )
            logger.info(f"The initial intention {whole_task_debug}")

        if cfg.reuse and already_executed(all_infos, task_id, whole_task):
            logger.info("Already executed, skip")
            continue

        logger.info(whole_task)

        BOUNDS = meta_info["action_bounds"]

        IMAGE_INIT_top = np.transpose(obs["rgb"]["top"], (1, 2, 0))

        # rewrite
        try:
            rewrite_prompt = rewrite(whole_task)
        except Exception as e:
            logger.info(f"ERROR: {e}")

        # coarse-grained rank
        retrieval_top_k = 2
        coarse_rank_list, _ = coarse_grained_rank(rewrite_prompt, retrieval_top_k)
        
        # Retrieve index based on rough rating
        score_list = fine_grained_rank(rewrite_prompt, coarse_rank_list)
        sorted_indices = sorted(range(len(score_list)), key=lambda k: score_list[k])

        # Add API Search
        prompt = get_topk_examples_and_relevant_api_prompt(coarse_rank_list,sorted_indices,)

        # Generate prompt
        prompt = insert_task_into_prompt(whole_task, prompt)

        # Code generation
        try:
            result = gpt4v(prompt,IMAGE_INIT_top)
            logger.info(result)
        except Exception as e:
            result = "request failed"
            logger.info(f"ERROR: {e}")

        exec_codes = analysis_gpt4_code(result, cfg.task)

        templates = get_templates(obs,prompt_assets, cfg.task)
        all_images = templates
        PROMPT = prompt
        PROMPT_ASSETS = prompt_assets
        OBS_LIST = obs_list

        done = False
        elapsed_steps = 0
        ACTIONS = []
        ACTION = None
        IMAGE_INIT_top = np.transpose(obs["rgb"]["top"], (1, 2, 0))
        IMAGE_INIT_front = np.transpose(obs["rgb"]["front"], (1, 2, 0))
        while True:
            if cfg.task == "scene_understanding" or cfg.task == "rearrange" or cfg.task == "rearrange_then_restore":
                IMAGE_SCENE = prompt_assets["scene"]["rgb"]["top"]
            elif cfg.task == "novel_adj" or cfg.task == "novel_adj_and_noun":
                IMAGE_FRONT = IMAGE_INIT_front
                IMAGE_PROMPT = list(prompt_assets.values())
            elif cfg.task == "twist" or cfg.task == "follow_order":
                IMAGE_SCENE_LIST = [x["rgb"]["top"] for x in prompt_assets.values()]
            elif cfg.task == "follow_motion":
                IMAGE_SCENE_LIST = [x["rgb"]["top"] for x in prompt_assets.values()][1:]
            IMAGE = IMAGE_INIT_top
            info = None
            # for code in exec_codes.splitlines():
            for code in exec_codes:
                code = code.strip()
                try:
                    if code.startswith("#"):
                        continue
                    if "EXE".lower() in code.lower() or len(code) < 4 or "return" in code:
                        # the exe is done by the simulator
                        continue
                    elif "PickPlace".lower() in code.lower():
                        code = "PickPlace" + code.split("PickPlace")[-1]
                        ACTION = eval(code)
                        ACTIONS.append(ACTION)
                    elif "Actions".lower() in code.lower() and ("DistractorActions".lower() not in code.lower()) and ("RearrangeActions".lower() not in code.lower()):
                        if "Actions" in code:
                            ACTIONS_ = eval(code.split("Actions = ")[-1])
                        elif "actions" in code:
                            ACTIONS_ = eval(code.split("actions = ")[-1])
                        ACTIONS.extend(ACTIONS_)
                    elif "DistractorActions".lower() in code.lower():
                        code = "DistractorActions" + code.split("DistractorActions")[-1]
                        ACTIONS_ = eval(code)
                        ACTIONS.extend(ACTIONS_)
                    elif "RearrangeActions".lower() in code.lower():
                        code = "RearrangeActions" + code.split("RearrangeActions")[-1]
                        ACTIONS_ = eval(code)
                        ACTIONS.extend(ACTIONS_)
                    elif "action =".lower() in code.lower():
                        ACTIONS_ = eval(code.split("action = ")[-1])
                        ACTIONS.extend(ACTIONS_)
                    else:
                        exec(code)
                except Exception as e:
                    logger.info(f"Exception: {e} for {code}")
                    task_info = exception_handler(
                        e,
                        logger,
                        task_id=task_id,
                        whole_task=whole_task,
                        exec_codes=exec_codes,
                    )
                    all_infos.append(task_info)
                    with open(eval_result_file_path, "w") as f:
                        json.dump(all_infos, f)
                    done = True
                    break

            while len(ACTIONS) > 0 and not done:
                ACTION = ACTIONS.pop(0)

                if isinstance(ACTION, tuple):
                    ACTION = ACTION[0]

                if not isinstance(ACTION, dict):
                    # this uses to skip the task, mainly due to the generated code is not correct
                    task_info = exception_handler(
                        "not a dict",
                        logger,
                        task_id=task_id,
                        whole_task=whole_task,
                        exec_codes=exec_codes,
                    )
                    all_infos.append(task_info)
                    with open(eval_result_file_path, "w") as f:
                        json.dump(all_infos, f)
                    break

                obs, _, done, info = env.step(ACTION)
            elapsed_steps += 1
            if done and info:
                task_info = {
                    "task_id": task_id,
                    "task": whole_task,
                    "exec": exec_codes,
                    "steps": elapsed_steps,
                    "success": info["success"],
                    "failure": info["failure"],
                }
            else:
                task_info = {
                    "task_id": task_id,
                    "task": whole_task,
                    "exec": exec_codes,
                    "steps": elapsed_steps,
                    "success": False,
                    "failure": False,
                }
            logger.info(
                f"task id: {task_info['task_id']} success: {task_info['success']}"
            )
                
            if cfg.reuse and task_id - 1 < len(all_infos):
                all_infos[task_id - 1] = task_info

            all_infos.append(task_info)
            with open(eval_result_file_path, "w") as f:
                json.dump(all_infos, f)

            if debug_flag or (info and not info["success"]):
                img_path = os.path.join(
                    result_folder, "imgs", f"{task_id}_{whole_task}_top.png"
                )
                if not os.path.exists(os.path.dirname(img_path)):
                    os.makedirs(os.path.dirname(img_path))
                IMAGE_INIT_top = cv2.cvtColor(IMAGE_INIT_top, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_path, IMAGE_INIT_top)
            break
                
    success_rate = sum([info["success"] for info in all_infos]) / len(all_infos)
    logger.warning(msg="==================Evaluation Done=========================")
    logger.info(cfg)
    logger.info("Success rate: {}".format(success_rate))
    env.env.close()
    del env


if __name__ == "__main__":
    prompt_modal = ["single"]
    # prompt_modal = ["multi", "single"]
    tasks = [
        # "visual_manipulation",
        # "rotate",
        "rearrange_then_restore",
        # "rearrange",
        "scene_understanding",
        "novel_adj",
        "novel_noun",
        "twist",
        "follow_order",
        "sweep_without_exceeding",
        "same_shape",
        "manipulate_old_neighbor",
        "pick_in_order_then_restore",
        "novel_adj_and_noun",
        "follow_motion",
        "same_texture",
    ]
    partitions = [
        "placement_generalization",
        "combinatorial_generalization",
        "novel_object_generalization",
        "novel_task_generalization",
    ]
    save_dir = "./visual_programming_prompt/save_demo_output"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    seed = 42
    hide_arm =False # False for demo usage, True for eval usage
    for task in tasks:
        for partition in partitions:
            for modal in prompt_modal:
                eval_cfg = {
                    "partition":partition,
                    "task": task,
                    "device": "cuda:0",
                    "prompt_modal": modal,
                    "reuse": False,
                    "save_dir": save_dir,
                    "debug_flag": False,
                    "hide_arm": hide_arm,
                    "seed": seed,
                }
                logger_file = (
                    save_dir
                    + "/eval_on_seed_{}_hide_arm_{}_{}_{}_{}_modal.log".format(
                        eval_cfg["seed"],
                        eval_cfg["hide_arm"],
                        partition,
                        task,
                        modal,
                    )
                )
                if os.path.exists(path=logger_file):
                    os.remove(logger_file)
                logger = create_logger(logger_file)
                main(EasyDict(eval_cfg), logger)
                del logger
