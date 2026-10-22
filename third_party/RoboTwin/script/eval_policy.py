import sys
import os
import subprocess
import time

sys.path.append("./")
sys.path.append(f"./policy")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH
from envs.utils.create_actor import UnStableError

import numpy as np
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import argparse
import pdb

from generate_episode_instructions import *

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance


def eval_function_decorator(policy_name, model_name):
    try:
        policy_model = importlib.import_module(policy_name)
        return getattr(policy_model, model_name)
    except ImportError as e:
        raise e

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def get_eval_video_size(args):
    head_camera_cfg = get_camera_config(args["camera"]["head_camera_type"])
    video_w = int(head_camera_cfg["w"])
    video_h = int(head_camera_cfg["h"])

    if args["camera"].get("collect_wrist_camera", False):
        wrist_camera_cfg = get_camera_config(args["camera"]["wrist_camera_type"])
        wrist_w = int(wrist_camera_cfg["w"])
        wrist_h = int(wrist_camera_cfg["h"])
        video_w = max(video_w, wrist_w * 2)
        video_h = video_h + wrist_h

    return f"{video_w}x{video_h}"


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _result_suffix_from_task_config(task_config):
    if task_config == "demo_clean":
        return "clean"
    if task_config == "demo_randomized":
        return "random"
    raise ValueError(
        f"Unsupported `task_config` for fixed result naming: {task_config}. "
        "Expected one of: ['demo_clean', 'demo_randomized']."
    )


def main(usr_args):
    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = usr_args["task_name"]
    task_config = usr_args["task_config"]
    ckpt_setting = usr_args["ckpt_setting"]
    # checkpoint_num = usr_args['checkpoint_num']
    policy_name = usr_args["policy_name"]
    instruction_type = usr_args["instruction_type"]
    skip_get_obs_within_replan = parse_bool(usr_args.get("skip_get_obs_within_replan", False))
    eval_num_episodes = int(usr_args.get("eval_num_episodes", 100))
    if eval_num_episodes <= 0:
        raise ValueError(f"`eval_num_episodes` must be > 0, got: {eval_num_episodes}")
    max_attempts = int(usr_args.get("max_attempts", 1))
    if max_attempts < 1:
        raise ValueError(f"`max_attempts` must be >= 1, got: {max_attempts}")
    evaluation_profile = usr_args.get("evaluation_profile")
    attempt_protocol = str(usr_args.get("attempt_protocol", "stop_on_success"))
    if attempt_protocol not in {"single_attempt", "stop_on_success", "fixed_attempts"}:
        raise ValueError("Unknown attempt_protocol")
    if attempt_protocol != "stop_on_success" and evaluation_profile != "main_observed":
        raise ValueError("Independent SR/fixed-attempt protocols require the explicit main_observed profile")
    if attempt_protocol == "single_attempt" and max_attempts != 1:
        raise ValueError("single_attempt requires max_attempts=1")
    fixed_seed = usr_args.get("fixed_seed", None)
    if isinstance(fixed_seed, str) and fixed_seed.strip().lower() in {"", "none", "null"}:
        fixed_seed = None
    if fixed_seed is not None and eval_num_episodes != 1:
        raise ValueError(
            "`fixed_seed` is a single-episode retry protocol; set eval_num_episodes=1 "
            f"when fixed_seed is set, got {eval_num_episodes}"
        )
    zeva_mode = str(usr_args.get("zeva_mode", "base"))
    # get_model validates observation skipping against the resolved CTE and
    # normalization contract. Legacy completed-effect CTE requires every action;
    # predictive RGB CTE can read only boundaries with stateless normalization.
    eval_output_dir = usr_args.get("eval_output_dir")
    save_dir = None
    video_save_dir = None
    video_size = None

    get_model = eval_function_decorator(policy_name, "get_model")
    # Import after the policy establishes the project root on sys.path. The
    # helpers themselves are stdlib-only and can be used without the simulator.
    if evaluation_profile == "main_observed":
        from fastwam.zeva_main_evaluation import load_manifest
    else:
        from experiments.robotwin.multiattempt import load_manifest
    manifest_episodes = None
    manifest_sha256 = None
    manifest_path = usr_args.get("episode_manifest")
    if manifest_path:
        if fixed_seed is not None:
            raise ValueError("episode_manifest and fixed_seed are mutually exclusive")
        manifest, manifest_sha256 = load_manifest(
            manifest_path, expected_sha256=usr_args.get("expected_manifest_sha256"),
        )
        for name, value in (("task_config", task_config), ("instruction_type", instruction_type)):
            if manifest[name] != value:
                raise ValueError(f"Manifest {name} differs from evaluation configuration")
        for name in ("action_horizon", "replan_steps", "max_attempts"):
            if int(usr_args.get(name, manifest[name])) != manifest[name]:
                raise ValueError(f"Manifest {name} differs from policy execution contract")
        if evaluation_profile == "main_observed":
            if manifest["attempt_protocol"] != attempt_protocol:
                raise ValueError("Main manifest attempt protocol differs from runtime")
            for name in ("pim_success_replay_enabled", "pim_success_replay_phase_threshold", "pim_success_replay_blend"):
                if usr_args.get(name) != manifest[name]:
                    raise ValueError(f"Main manifest {name} differs from runtime")
        manifest_episodes = [row for row in manifest["episodes"] if row["task"] == task_name]
        if not manifest_episodes:
            raise ValueError(f"Task {task_name} is not in the frozen manifest")
        eval_num_episodes = len(manifest_episodes)
        max_attempts = manifest["max_attempts"]

    with open(f"./task_config/{task_config}.yml", "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    if manifest_episodes is not None:
        # The formal protocol requires one auditable recording per attempt.
        args["eval_video_log"] = True

    args['task_name'] = task_name
    args["task_config"] = task_config
    args["ckpt_setting"] = ckpt_setting

    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")

    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment_type):
        robot_file = _embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise "No embodiment files"
        return robot_file

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise "embodiment items should be 1 or 3"

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    if eval_output_dir is not None and str(eval_output_dir).strip() != "":
        save_dir = Path(str(eval_output_dir))
    else:
        save_dir = Path(f"eval_result/{task_name}/{policy_name}/{task_config}/{ckpt_setting}/{eval_ts}")
    save_dir.mkdir(parents=True, exist_ok=True)

    if args["eval_video_log"]:
        video_save_dir = save_dir
        video_size = get_eval_video_size(args)
        video_save_dir.mkdir(parents=True, exist_ok=True)
        args["eval_video_save_dir"] = video_save_dir

    # output camera config
    print("============= Config =============\n")
    print("\033[95mMessy Table:\033[0m " + str(args["domain_randomization"]["cluttered_table"]))
    print("\033[95mRandom Background:\033[0m " + str(args["domain_randomization"]["random_background"]))
    if args["domain_randomization"]["random_background"]:
        print(" - Clean Background Rate: " + str(args["domain_randomization"]["clean_background_rate"]))
    print("\033[95mRandom Light:\033[0m " + str(args["domain_randomization"]["random_light"]))
    if args["domain_randomization"]["random_light"]:
        print(" - Crazy Random Light Rate: " + str(args["domain_randomization"]["crazy_random_light_rate"]))
    print("\033[95mRandom Table Height:\033[0m " + str(args["domain_randomization"]["random_table_height"]))
    print("\033[95mRandom Head Camera Distance:\033[0m " + str(args["domain_randomization"]["random_head_camera_dis"]))

    print("\033[94mHead Camera Config:\033[0m " + str(args["camera"]["head_camera_type"]) + f", " +
          str(args["camera"]["collect_head_camera"]))
    print("\033[94mWrist Camera Config:\033[0m " + str(args["camera"]["wrist_camera_type"]) + f", " +
          str(args["camera"]["collect_wrist_camera"]))
    print("\033[94mEmbodiment Config:\033[0m " + embodiment_name)
    print("\n==================================")

    TASK_ENV = class_decorator(args["task_name"])
    args["policy_name"] = policy_name
    usr_args["left_arm_dim"] = len(args["left_embodiment_config"]["arm_joints_name"][0])
    usr_args["right_arm_dim"] = len(args["right_embodiment_config"]["arm_joints_name"][1])

    seed = usr_args["seed"]

    st_seed = 100000 * (1 + seed)
    suc_nums = []
    test_num = eval_num_episodes
    topk = 1

    model = get_model(usr_args)
    st_seed, suc_num = eval_policy(task_name,
                                   TASK_ENV,
                                   args,
                                   model,
                                   st_seed,
                                   test_num=test_num,
                                   video_size=video_size,
                                   instruction_type=instruction_type,
                                   skip_get_obs_within_replan=skip_get_obs_within_replan,
                                   fixed_seed=fixed_seed,
                                   max_attempts=max_attempts,
                                   attempt_protocol=attempt_protocol,
                                   evaluation_profile=evaluation_profile,
                                   manifest_episodes=manifest_episodes,
                                   fixed_instruction=usr_args.get("fixed_instruction"),
                                   attempt_log_path=usr_args.get("attempt_log_path") or str(save_dir / "attempts.jsonl"),
                                   record_context={
                                       "manifest_sha256": manifest_sha256,
                                       "run_id": usr_args.get("run_id") or eval_ts,
                                       "condition": usr_args.get("condition") or zeva_mode,
                                       "task_config": task_config,
                                       "instruction_type": instruction_type,
                                       "skip_get_obs_within_replan": skip_get_obs_within_replan,
                                   })
    # Commit only after every frozen episode in this task completed. A failed
    # task must not advance the next task's model stream or enable partial resume.
    if hasattr(model, "save_model_rng_state"):
        model.save_model_rng_state()
    suc_nums.append(suc_num)

    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]

    result_suffix = _result_suffix_from_task_config(task_config)
    file_path = os.path.join(save_dir, f"_result_{result_suffix}.txt")
    with open(file_path, "w") as file:
        file.write(f"Timestamp: {eval_ts}\n\n")
        file.write(f"Instruction Type: {instruction_type}\n\n")
        if evaluation_profile == "main_observed":
            metric = "SR@1" if attempt_protocol == "single_attempt" else f"CSR@{max_attempts}"
            file.write(f"Protocol: {attempt_protocol}\nMetric: {metric}\n")
            file.write("CSR is any success within the attempt budget; post-success stability is reported separately.\n\n")
        # file.write(str(task_reward) + '\n')
        file.write("\n".join(map(str, np.array(suc_nums) / test_num)))

    print(f"Data has been saved to {file_path}")
    # return task_reward


def eval_policy(task_name,
                TASK_ENV,
                args,
                model,
                st_seed,
                test_num=100,
                video_size=None,
                instruction_type=None,
                skip_get_obs_within_replan=False,
                fixed_seed=None,
                max_attempts=1,
                manifest_episodes=None,
                fixed_instruction=None,
                attempt_log_path=None,
                record_context=None,
                attempt_protocol="stop_on_success",
                evaluation_profile=None):
    if evaluation_profile == "main_observed":
        from fastwam.zeva_main_evaluation import ATTEMPT_SCHEMA, append_attempt
    else:
        from experiments.robotwin.multiattempt import ATTEMPT_SCHEMA, append_attempt
    if attempt_protocol not in {"single_attempt", "stop_on_success", "fixed_attempts"}:
        raise ValueError("Unknown attempt_protocol")
    if attempt_protocol == "single_attempt" and max_attempts != 1:
        raise ValueError("single_attempt requires exactly one attempt")
    if evaluation_profile != "main_observed" and attempt_protocol != "stop_on_success":
        raise ValueError("New attempt protocols must use main_observed")
    if manifest_episodes is not None and len(manifest_episodes) != test_num:
        raise ValueError("Manifest task size differs from test_num")
    if manifest_episodes is not None and fixed_seed is not None:
        raise ValueError("Manifest and fixed_seed cannot both select a scene")
    print(f"\033[34mTask Name: {args['task_name']}\033[0m")
    print(f"\033[34mPolicy Name: {args['policy_name']}\033[0m")

    expert_check = True
    TASK_ENV.suc = 0
    TASK_ENV.test_num = 0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []

    policy_name = args["policy_name"]
    eval_func = eval_function_decorator(policy_name, "eval")
    reset_func = eval_function_decorator(policy_name, "reset_model")

    now_seed = st_seed if fixed_seed is None else int(fixed_seed)
    task_total_reward = 0
    clear_cache_freq = args["clear_cache_freq"]

    args["eval_mode"] = True
    strict_seed = fixed_seed is not None or manifest_episodes is not None

    while succ_seed < test_num:
        episode_spec = manifest_episodes[succ_seed] if manifest_episodes is not None else None
        if episode_spec is not None:
            now_seed = episode_spec["seed"]
        render_freq = args["render_freq"]
        args["render_freq"] = 0

        if expert_check:
            try:
                # Expert validation is not the policy rollout and has no
                # ffmpeg writer attached yet. Disable video output here so a
                # successful plan check cannot dereference a missing writer.
                expert_args = dict(args)
                expert_args["eval_video_save_dir"] = None
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **expert_args)
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except UnStableError as e:
                # print(" -------------")
                # print("Error: ", e)
                # print(" -------------")
                TASK_ENV.close_env()
                if strict_seed:
                    raise RuntimeError(
                        f"fixed_seed={fixed_seed} expert setup failed; refusing to change the seed (selected={now_seed})"
                    ) from e
                now_seed += 1
                args["render_freq"] = render_freq
                continue
            except Exception as e:
                print(" -------------")
                print("Error: ", e)
                print("Stack Trace: ", traceback.format_exc())
                print(" -------------")
                TASK_ENV.close_env()
                if strict_seed:
                    raise RuntimeError(
                        f"fixed_seed={fixed_seed} expert check failed; refusing to change the seed (selected={now_seed})"
                    ) from e
                now_seed += 1
                args["render_freq"] = render_freq
                print("error occurs !")
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
            suc_test_seed_list.append(now_seed)
        else:
            if strict_seed:
                raise RuntimeError(
                    f"Selected scene seed={now_seed} failed expert validation; refusing to substitute a seed"
                )
            now_seed += 1
            args["render_freq"] = render_freq
            continue

        args["render_freq"] = render_freq

        try:
            TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        except UnStableError as e:
            # This seed passed expert_check but failed during rollout env init.
            # Roll back the accepted-seed counter and skip to next seed.
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            TASK_ENV.close_env()
            if strict_seed:
                raise RuntimeError(
                    f"fixed_seed={fixed_seed} rollout setup failed; refusing to change the seed (selected={now_seed})"
                ) from e
            now_seed += 1
            continue
        except Exception as e:
            succ_seed -= 1
            if len(suc_test_seed_list) > 0 and suc_test_seed_list[-1] == now_seed:
                suc_test_seed_list.pop()
            print(" -------------")
            print("Error: ", e)
            print("Stack Trace: ", traceback.format_exc())
            print(" -------------")
            TASK_ENV.close_env()
            if strict_seed:
                raise RuntimeError(
                    f"fixed_seed={fixed_seed} rollout setup failed; refusing to change the seed (selected={now_seed})"
                ) from e
            now_seed += 1
            print("error occurs !")
            continue
        if episode_spec is not None:
            instruction = episode_spec["instruction"]
        elif fixed_instruction is not None:
            instruction = str(fixed_instruction)
        else:
            episode_info_list = [episode_info["info"]]
            results = generate_episode_descriptions(args["task_name"], episode_info_list, test_num)
            instruction = np.random.choice(results[0][instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)  # set language instruction

        current_video_path = None
        episode_idx = TASK_ENV.test_num
        video_enabled = TASK_ENV.eval_video_path is not None

        def start_attempt_video(attempt_id):
            if not video_enabled:
                return None
            path = Path(TASK_ENV.eval_video_path) / f"episode{episode_idx}_attempt{attempt_id}.mp4"
            ffmpeg = subprocess.Popen(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "rgb24",
                    "-video_size",
                    video_size,
                    "-framerate",
                    "10",
                    "-i",
                    "-",
                    "-pix_fmt",
                    "yuv420p",
                    "-vcodec",
                    "libx264",
                    "-crf",
                    "23",
                    str(path),
                ],
                stdin=subprocess.PIPE,
            )
            TASK_ENV._set_eval_video_ffmpeg(ffmpeg)
            return path

        def stop_attempt_video():
            if video_enabled and getattr(TASK_ENV, "eval_video_ffmpeg", None):
                TASK_ENV._del_eval_video_ffmpeg()

        succ = False
        ever_succ = False
        first_success_attempt = None
        reset_func(model)  # Episode reset exactly once; retries preserve PIM.
        for attempt_id in range(max_attempts):
            # fixed_attempts continues after success, so per-attempt success
            # must not inherit the preceding attempt's outcome.
            succ = False
            first_success_before = first_success_attempt
            if attempt_id > 0:
                # Recreate the same fixed-seed episode for a genuine retry.
                stop_attempt_video()
                TASK_ENV.close_env()
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                TASK_ENV.set_instruction(instruction=instruction)
            current_video_path = start_attempt_video(attempt_id)
            reset_scope = "episode" if attempt_id == 0 else "attempt"
            if hasattr(model, "begin_attempt"):
                model.begin_attempt(attempt_id)
            elif attempt_id:
                reset_scope = "episode"  # Explicitly record incompatible fallback behavior.
                reset_func(model)
            before = model.evaluation_metadata() if hasattr(model, "evaluation_metadata") else {}
            if evaluation_profile == "main_observed":
                if before.get("evaluation_profile") != "main_observed" or before.get("memory_semantics") != "completed_effect_trace":
                    raise ValueError("Main evaluator requires the actual observed-effect runtime, not an Ego checkpoint/config")
                if reset_scope != ("episode" if attempt_id == 0 else "attempt"):
                    raise ValueError("Main retries require an attempt reset that preserves experience")
            started = time.perf_counter()
            status, error = "timeout", None
            try:
                while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
                    need_obs = True
                    if skip_get_obs_within_replan and hasattr(model, "should_request_observation"):
                        need_obs = bool(model.should_request_observation())

                    observation = None
                    if need_obs:
                        observation = TASK_ENV.get_obs()
                    eval_func(TASK_ENV, model, observation)
                    if TASK_ENV.eval_success:
                        succ = True
                        status = "success"
                        break
                if hasattr(model, "finalize_attempt"):
                    model.finalize_attempt(TASK_ENV, success=succ)
                if succ:
                    ever_succ = True
                    if first_success_attempt is None:
                        first_success_attempt = attempt_id + 1
            except Exception as exc:
                status, error = "error", repr(exc)
                raise
            finally:
                after = model.evaluation_metadata() if hasattr(model, "evaluation_metadata") else {}
                if attempt_log_path:
                    record = dict(record_context or {})
                    record.update(after)
                    record.update({
                        "schema": ATTEMPT_SCHEMA,
                        "task": task_name,
                        "episode_id": episode_spec["episode_id"] if episode_spec else f"{task_name}:{now_seed}",
                        "scene_seed": int(now_seed),
                        "instruction": str(instruction),
                        "attempt": attempt_id + 1,
                        "reset_scope": reset_scope,
                        "max_attempts": max_attempts,
                        "success": bool(succ) if status != "error" else False,
                        "status": status,
                        "error": error,
                        "step_count": int(getattr(model, "step_count", TASK_ENV.take_action_cnt)),
                        "step_limit": int(TASK_ENV.step_lim),
                        "wall_time_seconds": time.perf_counter() - started,
                        "pim_entries_before": int(before.get("pim_entries", 0)),
                        "pim_entries_after": int(after.get("pim_entries", 0)),
                        "seed_stream_before": before.get("seed_stream"),
                        "video_path": str(current_video_path) if current_video_path else None,
                    })
                    if evaluation_profile == "main_observed":
                        record.update({
                            "attempt_protocol": attempt_protocol,
                            "scene_reset": True,
                            "memory_reset": attempt_id == 0,
                            "model_rng_reset": False,
                            "first_success_attempt_before": first_success_before,
                            "first_success_attempt": first_success_attempt,
                            "post_success_attempt": first_success_before is not None,
                        })
                    append_attempt(attempt_log_path, record)
            if succ and attempt_protocol != "fixed_attempts":
                break
        # task_total_reward += TASK_ENV.episode_score
        if video_enabled:
            stop_attempt_video()
            if current_video_path is None or not current_video_path.exists():
                raise FileNotFoundError(f"Expected eval video file not found: {current_video_path}")
            # Keep immutable per-attempt paths referenced by structured logs.

        if ever_succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        metric_label = ("SR@1" if attempt_protocol == "single_attempt" else f"CSR@{max_attempts}") if evaluation_profile == "main_observed" else "Success rate"
        print(
            f"\033[93m{task_name}\033[0m | \033[94m{args['policy_name']}\033[0m | \033[92m{args['task_config']}\033[0m | \033[91m{args['ckpt_setting']}\033[0m\n"
            f"{metric_label}: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => \033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n"
        )
        # TASK_ENV._take_picture()
        if not strict_seed:
            now_seed += 1

    return now_seed, TASK_ENV.suc


def parse_args_and_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overrides", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Parse overrides
    def parse_override_pairs(pairs):
        override_dict = {}
        for i in range(0, len(pairs), 2):
            key = pairs[i].lstrip("--")
            value = pairs[i + 1]
            try:
                value = eval(value)
            except:
                pass
            override_dict[key] = value
        return override_dict

    if args.overrides:
        overrides = parse_override_pairs(args.overrides)
        config.update(overrides)

    return config


if __name__ == "__main__":
    from test_render import Sapien_TEST
    Sapien_TEST()

    usr_args = parse_args_and_config()

    main(usr_args)
