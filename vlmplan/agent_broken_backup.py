
@dataclass
class StepRecord:
    """One event in EdgeMind's observable execution trace."""
    kind: str
    frame: int = 0
    image: np.ndarray | None = None
    world_image: np.ndarray | None = None
    scene: Scene | None = None
    decision: Decision | None = None
    checker_reason: str = ""
    action: dict | None = None
    action_ok: bool = False
    outcome: Outcome | None = None


def run_task_steps(
    task: Task,
    world: World,
    planner,
    closed_loop: bool = True,
    use_validator: bool = True,
    perception: str = "camera",
    max_steps: int = 30,
    max_replans: int = 8,
    seed: int = 0,
    dynamic: DynamicWorld | None = None,
):
    """Yield observable execution records for the Streamlit UI.

    Pipeline:
        Observe -> Plan -> Validate -> Act -> Re-observe -> Replan

    The generator pauses after every yielded record, allowing the UI to
    display progress and apply live disruptions between execution steps.
    """
    from .planning import goal_met

    rng = np.random.default_rng(seed)
    executor = Executor(world, rng)

    frame = 0
    steps = 0
    replans = 0
    rejected = 0

    def observe():
        nonlocal frame
        frame += 1

        # External changes occur between observations.
        if dynamic is not None:
            dynamic.tick(frame)

        if perception == "camera":
            image = render(world, seed=seed * 1000 + frame)
            scene = scene_from_image(image)
        else:
            image = None
            scene = scene_from_world(world)

        if dynamic is not None and dynamic.glitch_pending:
            dynamic.apply_glitch(scene)

        # This second image is for the human-facing digital twin view.
        world_image = render(world, seed=seed * 1000 + frame + 500000)

        return StepRecord(
            kind="observe",
            frame=frame,
            image=image,
            world_image=world_image,
            scene=scene,
        )

    def finish(status, success, reason=""):
        return Outcome(
            status=status,
            success=success,
            steps=steps,
            replans=replans,
            rejected_plans=rejected,
            unsafe=list(executor.unsafe),
            reason=reason,
        )

    while True:
        # 1. Observe the current world.
        observation = observe()
        scene = observation.scene
        image = observation.image
        yield observation

        # 2. Ask the selected planner for a decision.
        decision = planner.plan(task.text, scene, image)
        yield StepRecord(
            kind="decision",
            frame=frame,
            decision=decision,
        )

        # 3. Validate the plan against a goal compiled independently
        #    of the planner.
        if use_validator and decision.status == "plan":
            gate = compile_goal(task.text, scene)

            if gate.status == "plan":
                checker_reason = validate(
                    decision.steps, scene, gate.goal
                )
            else:
                checker_reason = f"{gate.status}: {gate.reason}"

            yield StepRecord(
                kind="checked",
                frame=frame,
                checker_reason=checker_reason or "",
            )

            if checker_reason:
                rejected += 1

                if gate.status != "plan":
                    decision = gate
                elif rejected <= max_replans:
                    replans += 1
                    continue
                else:
                    outcome = finish(
                        "gave_up", False, checker_reason
                    )
                    yield StepRecord(
                        kind="done", frame=frame, outcome=outcome
                    )
                    return

        # 4. Handle refusal or clarification.
        if decision.status != "plan":
            if (
                dynamic is not None
                and steps == 0
                and replans < max_replans
            ):
                yield StepRecord(
                    kind="retry",
                    frame=frame,
                    checker_reason=decision.reason
                    or decision.status,
                )
                replans += 1
                continue

            right = (
                task.expect != "do"
                or _unreachable_target(task, world)
            )
            success = (
                steps == 0
                and right
                and (
                    decision.status == task.expect
                    or task.expect == "do"
                )
            )

            outcome = finish(
                decision.status, success, decision.reason
            )
            yield StepRecord(
                kind="done", frame=frame, outcome=outcome
            )
            return

        # 5. Execute actions one at a time. Each action is visible
        #    to the UI before the generator continues.
        failed = False

        for action in decision.steps:
            if steps >= max_steps:
                break

            steps += 1
            action_ok = executor.run(action, scene)

            yield StepRecord(
                kind="action",
                frame=frame,
                action=action,
                action_ok=action_ok,
                world_image=render(
                    world, seed=seed * 1000 + frame + steps + 500000
                ),
            )

            if not action_ok:
                failed = True
                if closed_loop:
                    break

        # Never leave an object held when the run terminates or replans.
        if world.holding is not None:
            cleanup_ok = executor.run(
                {"skill": "place_table"}, scene
            )
            yield StepRecord(
                kind="action",
                frame=frame,
                action={"skill": "place_table"},
                action_ok=cleanup_ok,
                world_image=render(
                    world,
                    seed=seed * 1000 + frame + steps + 600000,
                ),
            )

        # In open-loop mode, or at the step limit, stop execution.
        if not closed_loop or steps >= max_steps:
            break

        # 6. Re-observe and independently check whether the goal
        #    was actually achieved.
        verification = observe()
        check_scene = verification.scene
        yield verification

        gate = compile_goal(task.text, check_scene)

        if (
            not failed
            and gate.status == "plan"
            and gate.goal is not None
            and goal_met(check_scene, gate.goal)
        ):
            break

        if not failed and gate.status == "plan" and not decision.steps:
            break

        replans += 1

        if replans > max_replans:
            break

    # 7. Publish the final result.
    success = task.expect == "do" and task.satisfied(world)
    outcome = finish("done", success)

    yield StepRecord(
        kind="done",
        frame=frame,
        outcome=outcome,
    )
