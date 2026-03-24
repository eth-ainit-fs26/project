"""
Monkey-patching helpers for the Project 1 notebook exercises.

Each function applies a student-defined method to the appropriate class.
Import and call these at the end of each exercise cell instead of writing
the patching and confirmation prints inline.
"""


def patch_hook_evaluator(cosine_fn, quality_fn):
    """Patch HookEvaluator with student-implemented similarity methods."""
    from agents.hooks import HookEvaluator
    HookEvaluator.cosine_similarity = cosine_fn
    HookEvaluator.evaluate_hook_quality_with_embedding = quality_fn
    print("HookEvaluator patched: cosine_similarity + evaluate_hook_quality_with_embedding")


def patch_marketing_session(run_session_fn, run_step_fn):
    """Patch MarketingSession with student-implemented session methods."""
    from agents.marketing_session import MarketingSession
    MarketingSession.run_session = run_session_fn
    MarketingSession.run_step = run_step_fn
    print("MarketingSession patched: run_session + run_step")


def patch_world_simulator(accuracy_fn):
    """Patch WorldSimulatorP1 with the student-implemented accuracy method."""
    from app.world_sim_p1 import WorldSimulatorP1
    WorldSimulatorP1.calculate_segment_hook_accuracy = accuracy_fn
    print("WorldSimulatorP1 patched: calculate_segment_hook_accuracy")


def patch_opro(meta_prompt_fn):
    """Patch Opro2 with the student-implemented meta-prompt method."""
    from agents.opro import Opro2
    Opro2.create_meta_prompt = meta_prompt_fn
    print("Opro2 patched: create_meta_prompt")


def show_meta_prompt_example(create_meta_prompt_fn):
    """
    Render an example meta-prompt using the student's create_meta_prompt
    implementation, so they can verify it looks reasonable before running
    the full simulation.
    """
    from models.models import CustomerSegment
    from agents.opro import OptimizerEntry

    example_segment = CustomerSegment.HIGH_VALUE_CUSTOMERS.value
    example_performers = [
        OptimizerEntry(accuracy_score=0.75, hook="Exclusive deals for our high-value customers!"),
        OptimizerEntry(accuracy_score=0.72, hook="Special offers just for you, our valued customer!"),
        OptimizerEntry(accuracy_score=0.70, hook="Unlock premium savings today with our exclusive deals!"),
        OptimizerEntry(accuracy_score=0.68, hook="Don't miss out on special offers tailored for you!"),
        OptimizerEntry(accuracy_score=0.65, hook="Join our loyalty program for exclusive discounts!"),
    ]

    try:
        print("Example meta-prompt (segment: High-Value, 5 past hooks):\n")
        print("=" * 25 + " BEGIN META-PROMPT " + "=" * 25)
        result = create_meta_prompt_fn(None, example_segment, example_performers)
        print(result)
        print("=" * 25 + "  END META-PROMPT  " + "=" * 25)
    except NotImplementedError:
        print("create_meta_prompt not yet implemented — implement it, then re-run this cell.")
    except Exception as e:
        print(f"Error rendering example meta-prompt: {e}")


def confirm_world_simulator(world_simulator, use_opro2_mode: bool):
    """Print a one-line confirmation after WorldSimulatorP1 is initialised."""
    mode = "OPRO2 (Direct Hook Optimization)" if use_opro2_mode else "Original OPRO (Prompt Optimization)"
    print(f"WorldSimulatorP1 ready — {world_simulator.simulation_duration} days, "
          f"{world_simulator.opro_cycle_days}-day OPRO cycles, mode: {mode}")
