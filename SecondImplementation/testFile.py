from ai_interface import run_headless_battle

def my_policy(state):
    # state has everything the AI needs: turn number, every unit's hp/
    # status effects/available skills, and the boss's hp/status/chosen
    # skills this turn. See build_state() below for the exact shape.
    print(state)
    commands = []
    for unit in state["units"]:
        if not unit["available_skills"]:
            continue  # dead or staggered - needs no command at all
        commands.append([0, 0])  # bottom skill, attack unopposed
    return commands

result = run_headless_battle(my_policy, seed=42)
print(result["outcome"], result["turns"])