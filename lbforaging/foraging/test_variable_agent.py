#%%

# test_variable_agents.py
# Run this in the same dir as environment.py
import traceback
import numpy as np
from pprint import pprint

# import the environment module you provided
import environment as env_mod  # file: environment.py (from user). :contentReference[oaicite:4]{index=4}

def simple_assert(cond, msg):
    if not cond:
        print("  [FAIL]", msg)
        return False
    else:
        print("  [OK]  ", msg)
        return True

def main():
    print("=== Variable-agent LBF environment basic tests ===")
    # quick checks for expected class and methods
    has_env = hasattr(env_mod, "ForagingEnv")
    if not has_env:
        print("ERROR: environment.py does not define ForagingEnv.")
        return
    ForagingEnv = env_mod.ForagingEnv

    # small deterministic settings
    players = 4
    min_players = 2
    min_player_level = 1
    max_player_level = 2
    min_food_level = 2
    max_food_level = 2
    field_size = (5, 5)
    max_num_food = 3
    sight = 1
    max_episode_steps = 50
    _max_episode_steps = 50
    force_coop = False

    # instantiate env
    try:
        env = ForagingEnv(
            players,
            min_players,
            min_player_level,
            max_player_level,
            min_food_level,
            max_food_level,
            field_size,
            max_num_food,
            sight,
            max_episode_steps,
            _max_episode_steps,
            force_coop,
            normalize_reward=True,
            grid_observation=False,
            observe_agent_levels=False,
            penalty=0.0,
            render_mode=None,
            is_variableN=True,           # use the ctor arg name that exists
            remove_agent_prov=0.0,        # disable random remove/create during step
            create_agent_prov=0.0,
        )
        print("Instantiated ForagingEnv.")
    except Exception as e:
        print("Failed to instantiate ForagingEnv:", e)
        traceback.print_exc()
        return

    # detect and report obvious API/name mismatches
    print("\n-- Checking for naming mismatches that will break runtime --")
    mismatch_count = 0
    if not hasattr(env, "is_variable_n") and hasattr(env, "is_variable_N"):
        print(" - Only is_variable_N exists (unexpected).")
    if hasattr(env, "is_variable_n") and not hasattr(env, "is_variable_N"):
        print(" - Warning: constructor uses is_variable_n but reset() may expect is_variable_N.")
        print("   Creating alias env.is_variable_N = env.is_variable_n to continue tests.")
        env.is_variable_N = env.is_variable_n  # alias to avoid AttributeError
        mismatch_count += 1

    # create_agent vs spawn_one_agent
    if not hasattr(env, "create_agent") and hasattr(env, "spawn_one_agent"):
        print(" - create_agent() missing; creating alias env.create_agent = env.spawn_one_agent() so tests can proceed.")
        env.create_agent = lambda : env.spawn_one_agent(env.min_player_level, env.max_player_level)
        mismatch_count += 1

    if mismatch_count > 0:
        print(f" Made {mismatch_count} temporary aliases to allow tests to run. Note: fix these in source file.")

    # seed RNG for deterministic behavior
    try:
        env.seed(0)
    except Exception:
        pass

    # run reset and check initial observation / internal flags
    try:
        obs, info = env.reset(seed=0)
        print("\nreset() succeeded.")
    except Exception as e:
        print("\nreset() failed. Printing exception and aborting further tests.")
        traceback.print_exc()
        return

    # Basic post-reset checks
    print("\n-- Post-reset checks --")
    # check is_possible_agents existence and length
    simple_assert(hasattr(env, "is_possible_agents"), "env has attribute is_possible_agents")
    if hasattr(env, "is_possible_agents"):
        simple_assert(len(env.is_possible_agents) == players, f"is_possible_agents length == {players}")

    # check n_agent is within bounds
    ok = simple_assert(hasattr(env, "n_agent"), "env has attribute n_agent")
    if ok:
        simple_assert(env.min_agents <= env.n_agent <= env.max_agents, f"n_agent ({env.n_agent}) between min/max ({env.min_agents}/{env.max_agents})")

    # OBS sanity: each obs should match observation_space
    print("\n-- Observation checks --")
    try:
        assert isinstance(obs, tuple) or isinstance(obs, list)
        obs_tuple = tuple(obs)
        for i, ob in enumerate(obs_tuple):
            # check shape/in-bounds by calling observation_space.contains where possible
            contains = env.observation_space[i].contains(ob)
            simple_assert(contains, f"obs[{i}] in observation_space")
        print("Observation generation OK.")
    except AssertionError as e:
        print("Observation shape/contents mismatch:", e)
    except Exception:
        print("Error while checking observations:")
        traceback.print_exc()

    # Save baseline snapshots
    before_positions = [p.position for p in env.players]
    before_levels = [p.level for p in env.players]
    before_possible = list(env.is_possible_agents)
    print("\nplayers positions:", before_positions)
    print("players levels:   ", before_levels)
    print("is_possible_agents:", before_possible)
    print("field sum (food):", env.field.sum())

    # Test remove_one_agent()
    print("\n-- remove_one_agent() test --")
    try:
        result = env.remove_one_agent()
        print("remove_one_agent() returned:", result)
        # find which agent became impossible
        after_possible = env.is_possible_agents
        simple_assert(sum(1 for x in after_possible if x) <= sum(1 for x in before_possible if x), "number of possible agents decreased or equal after remove")
        # find removed id(s)
        removed_ids = [i for i, (b, a) in enumerate(zip(before_possible, after_possible)) if b and not a]
        if removed_ids:
            rid = removed_ids[0]
            p = env.players[rid]
            simple_assert(p.position == (-1, -1), f"removed player {rid} position is (-1, -1)")
            simple_assert(p.level == -1, f"removed player {rid} level == -1")
            simple_assert(p.reward == 0, f"removed player {rid} reward == 0")
            simple_assert(p.history == [], f"removed player {rid} history reset")
        else:
            print(" - No agent removed (maybe already at minimum count). This may be acceptable depending on implementation.")
    except Exception:
        print("Exception during remove_one_agent():")
        traceback.print_exc()

    # Test spawn_one_agent() (called spawn_one_agent in file; alias create_agent exists if missing)
    print("\n-- spawn_one_agent() test --")
    try:
        # record possible agents before spawn
        before_possible2 = list(env.is_possible_agents)
        # Call spawn via the available API: prefer spawn_one_agent if exists
        if hasattr(env, "spawn_one_agent"):
            result2 = env.spawn_one_agent(env.min_player_level, env.max_player_level)
            print("spawn_one_agent() returned:", result2)
        elif hasattr(env, "create_agent"):
            result2 = env.create_agent()
            print("create_agent() returned:", result2)
        else:
            print("No spawn/create method available to test.")
            result2 = None

        # check if any previously impossible agent became possible
        after_possible2 = env.is_possible_agents
        revived_ids = [i for i, (b, a) in enumerate(zip(before_possible2, after_possible2)) if (not b) and a]
        if revived_ids:
            rid = revived_ids[0]
            p = env.players[rid]
            simple_assert(p.position != (-1, -1), f"revived player {rid} placed on board (not -1,-1)")
            simple_assert(p.level >= 0, f"revived player {rid} has non-negative level")
            simple_assert(env.is_possible_agents[rid] is True, f"is_possible_agents[{rid}] == True")
        else:
            print(" - No agent revived (maybe none available).")

    except Exception:
        print("Exception during spawn_one_agent/create_agent:")
        traceback.print_exc()

    # Check observation after spawn/remove sequence
    print("\n-- Observations after remove/spawn sequence --")
    try:
        nobs = env.test_make_gym_obs()
        for i, ob in enumerate(nobs):
            # For disabled agents, many values should be default -1; we test trivially for an all-negative check
            if not env.is_possible_agents[i]:
                all_neg = (ob <= 0).all() or (ob <= -1).all()
                print(f" obs[{i}] (agent disabled) - quick negative check: {all_neg}")
            else:
                print(f" obs[{i}] (agent enabled) - sum: {ob.sum()}")
    except Exception:
        print("Exception while regenerating observations:")
        traceback.print_exc()

    # Test step() with actions: ensure disabled agents' actions are forced to NONE
    print("\n-- step() action enforcement test --")
    try:
        # Build actions (use 0 for NONE) — supply some nonzero for disabled agents to check enforcement
        actions = [0] * len(env.players)
        # set a disabled agent to propose a non-zero action to see if env overwrites
        for i, p in enumerate(env.players):
            if not env.is_possible_agents[i]:
                actions[i] = 1  # attempt a NORTH
                break
        print("Input actions:", actions)
        nobs2, rewards, done, trunc, info = env.step(actions)
        print("Step returned rewards:", rewards)
        # check that for disabled agents reward == 0
        for i, p in enumerate(env.players):
            if not env.is_possible_agents[i]:
                simple_assert(p.reward == 0, f"disabled player {i} reward == 0 after step")
    except Exception:
        print("Exception during step():")
        traceback.print_exc()

    # Repeated remove/create cycles
    print("\n-- Repeated remove/create cycles --")
    try:
        for i in range(6):
            env.remove_one_agent()
            env.spawn_one_agent(env.min_player_level, env.max_player_level)
        print("Repeated cycles done; sample player states:")
        sample = [(i, p.position, p.level, env.is_possible_agents[i]) for i, p in enumerate(env.players)]
        pprint(sample)
    except Exception:
        print("Exception during repeated cycles:")
        traceback.print_exc()

    print("\n=== Tests complete. ===")
    print("Notes / next steps:")
    print(" - Fix typos in source: unify is_variable_n vs is_variable_N; unify create_agent vs spawn_one_agent (or add wrapper).")
    print(" - Consider ensuring reset() uses existing attribute names and step() calls implemented spawn/create API.")
    print(" - The test script made instance-level aliases to continue tests; those are NOT fixes for production code.")

if __name__ == "__main__":
    main()

# %%
