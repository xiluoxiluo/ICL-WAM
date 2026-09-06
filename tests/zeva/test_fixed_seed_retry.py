from pathlib import Path


def test_fixed_seed_retry_path_refuses_seed_drift():
    source = Path("third_party/RoboTwin/script/eval_policy.py").read_text()
    policy_source = Path("experiments/robotwin/fastwam_policy/deploy_policy.py").read_text()
    assert "fixed_seed={fixed_seed} expert setup failed" in source
    assert "fixed_seed={fixed_seed} rollout setup failed" in source
    assert "episode{episode_idx}_attempt{attempt_id}.mp4" in source
    assert "finalize_attempt" in source
    assert "model.finalize_attempt(TASK_ENV, success=succ)" in source
    assert "replan_steps % transition_steps" in policy_source
    assert "def finalize_attempt(self, task_env, success: bool = False)" in policy_source
    assert "self._attempt_finalized" in policy_source
