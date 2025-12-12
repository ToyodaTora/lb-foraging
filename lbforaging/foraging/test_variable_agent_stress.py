import numpy as np
import traceback
from environment import ForagingEnv, Action


# ============================================================
#  Utility helpers
# ============================================================

def safe_assert(cond, msg):
    if not cond:
        print(f"[FAIL] {msg}")
        return False
    return True


def print_header(title):
    print("\n" + "="*70)
    print("TEST:", title)
    print("="*70)


# ============================================================
#  Test 1: Observation validity under random transitions
# ============================================================

def test_observation_stress(env, steps=120):
    print_header("Observation Space & Stability Stress Test")

    obs, _ = env.reset(seed=0)

    # _valid_actions 更新のためのダミー step
    dummy_actions = [0] * env.n_agent
    env.step(dummy_actions)

    ok = True

    for t in range(steps):
        # ランダム行動
        acts = []
        for i in range(len(env.players)):
            if env.is_possible_agents[i]:
                acts.append(np.random.randint(0, 5))
            else:
                acts.append(0)  # 無効エージェントは常に NONE

        # step
        next_obs, rewards, done, trunc, info = env.step(acts)

        # ---- Obs 全体の整合性 ----
        for i, o in enumerate(next_obs):
            if not env.observation_space[i].contains(o):
                print(f"[FAIL] obs[{i}] out of observation_space at step {t}: {o}")
                ok = False

            if np.isnan(o).any() or np.isinf(o).any():
                print(f"[FAIL] obs[{i}] contains NaN/Inf at step {t}")
                ok = False

        # ---- reward の sanity ----
        if np.isnan(rewards).any() or np.isinf(rewards).any():
            print(f"[FAIL] reward has NaN/Inf at step {t}")
            ok = False

        # update
        obs = next_obs

    if ok:
        print("[OK] Observation stability test passed.")
    return ok


# ============================================================
#  Test 2: Random remove/spawn & field consistency
# ============================================================

def test_remove_spawn_stress(env, cycles=50):
    print_header("Remove / Spawn Stress + Field Consistency")

    obs, _ = env.reset(seed=1)
    dummy_actions = [0] * env.n_agent
    env.step(dummy_actions)

    ok = True

    for c in range(cycles):
        # ランダムで remove / spawn
        if np.random.rand() < 0.4:
            # remove
            env.remove_one_agent()
        else:
            # spawn
            env.spawn_one_agent(env.min_player_level, env.max_player_level)

        # ---- field 整合性チェック ----
        field = env.field.copy()

        # active agents の位置リスト取得
        active_positions = []
        for i, p in enumerate(env.players):
            if env.is_possible_agents[i]:
                pos = p.position
                active_positions.append(tuple(pos))

                # Field に agent が置かれているか
                fy, fx = pos
                if fy < 0 or fx < 0:
                    print(f"[FAIL] active agent {i} has invalid negative pos after spawn/remove.")
                    ok = False


    if ok:
        print("[OK] Remove/Spawn & Field consistency passed.")

    return ok


# ============================================================
#  Test 3: Action availability sanity
# ============================================================

def test_action_availability(env, steps=50):
    print_header("Available Action Consistency")

    obs, _ = env.reset(seed=3)
    dummy_actions = [0] * env.n_agent
    env.step(dummy_actions)

    ok = True

    for t in range(steps):
        # 各 agent の _valid_actions を調べる
        for i, p in enumerate(env.players):
            av = env._valid_actions[p]  # ここが環境実装依存のキー
            if env.is_possible_agents[i]:
                if len(av) == 0:
                    print(f"[FAIL] enabled agent {i} has empty _valid_actions at step {t}")
                    ok = False
            else:
                # disabled agent の available action が NONE のみか
                if av != [Action.NONE]:
                    print(f"[FAIL] disabled agent {i} should have only NONE action, got {av}")
                    ok = False

        # ランダム行動
        acts = []
        for i in range(len(env.players)):
            if not env.is_possible_agents[i]:
                acts.append(0)
            else:
                acts.append(np.random.choice(env._valid_actions[env.players[i]]))

        obs, r, done, trunc, info = env.step(acts)

    if ok:
        print("[OK] Action availability stable.")

    return ok


# ============================================================
#  Test 4: Episode termination & truncation
# ============================================================

def test_episode_end(env):
    print_header("Episode Termination / Truncation")

    obs, _ = env.reset(seed=4)
    dummy_actions = [0] * env.n_agent
    env.step(dummy_actions)

    ok = True

    # max_episode_steps 分ループ
    for t in range(env._max_episode_steps + 5):
        acts = [0] * env.n_agent
        obs, r, done, trunc, info = env.step(acts)
        a = False
        # print("t=",t)
        # for y in range(env.field_size[0]):
        #     print("  ", end="")
        #     for x in range(env.field_size[0]):
        #         for p in env.players:
        #             if p.position == (y, x):
        #                 print("x ", end="")
        #                 a = True
        #                 break
        #         if a == False:
        #             print(str(env.field[y][x])+" ", end="")
        #         else:
        #             a = False
        #     print("")
        # print("")
        if t < env._max_episode_steps:
            # 途中でいきなり done=True になるとおかしい
            if done and not trunc:
                print(f"[FAIL] done=True unexpectedly at t={t}")
                ok = False
                break
        else:
            # 上限を超えたら必ず trunc=True
            if not trunc:
                print(f"[FAIL] trunc should be True when t > max_episode_steps, got {trunc}")
                ok = False
            break

    if ok:
        print("[OK] Episode termination logic correct.")

    return ok


# ============================================================
#  Main
# ============================================================

def main():
    print("=== Variable-Agent LBF Stress Test ===")

    # 必要に応じてパラメータは調整してください
    env = ForagingEnv(
        players=4,
        min_players=1,
        min_player_level=2,
        max_player_level=2,
        min_food_level=2,
        max_food_level=2,
        field_size=(5,5),
        max_num_food=10,
        sight=1,
        max_episode_steps=50,
        _max_episode_steps=50,
        force_coop=False,
        normalize_reward=True,
        grid_observation=False,
        observe_agent_levels=False,
        penalty=0.0,
        render_mode=None,
        is_variableN=True,
        remove_agent_prov=0.0,
        create_agent_prov=0.0,
    )

    results = []

    try:
        results.append(test_observation_stress(env))
        results.append(test_remove_spawn_stress(env))
        results.append(test_action_availability(env))
        results.append(test_episode_end(env))
    except Exception as e:
        print("[ERROR] Exception occurred:", e)
        traceback.print_exc()
        results.append(False)

    print("\n=== Summary ===")
    print(results)
    if all(results):
        print("[ALL OK] Environment is stable.")
    else:
        print("[SOME FAILURES] See log above.")

if __name__ == "__main__":
    main()
