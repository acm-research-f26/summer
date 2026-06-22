"""
main.py
-------
CLI entry point. Example:

    python main.py --case case30 --scenario light --provider anthropic --model claude-sonnet-4-6
    python main.py --case cigre_mv --scenario severe --provider openai --model gpt-4.1
    python main.py --case case30 --scenario disconnected --provider gemini --model gemini-2.5-flash

Requires the relevant API key in the environment:
  ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
"""

import argparse
import json

from grid_agent import load_network, make_scenario, GridAgentOrchestrator, LLMClient


def main():
    p = argparse.ArgumentParser(description="Run Grid-Agent on a benchmark scenario.")
    p.add_argument("--case", choices=["case30", "cigre_mv"], default="case30")
    p.add_argument("--scenario", choices=["light", "medium", "severe", "disconnected"], default="light")
    p.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default="anthropic")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--max-iterations", type=int, default=5)
    p.add_argument("--save-training-record", default=None,
                   help="Optional path to dump the Summarizer's training record as JSON.")
    args = p.parse_args()

    net, controllable = load_network(args.case)
    net = make_scenario(net, controllable, args.scenario)

    llm = LLMClient(provider=args.provider, model=args.model)
    orchestrator = GridAgentOrchestrator(llm=llm, max_iterations=args.max_iterations)

    result = orchestrator.run(net, controllable, verbose=True)

    print("\n================ RESULT ================")
    print(f"Success: {result['success']}")
    print(f"Iterations used: {result['iterations']}")
    print(f"Remaining violations: {len(result['final_violations'])}")

    if args.save_training_record:
        with open(args.save_training_record, "w") as f:
            json.dump(result["training_record"], f, indent=2)
        print(f"Training record saved to {args.save_training_record}")


if __name__ == "__main__":
    main()
