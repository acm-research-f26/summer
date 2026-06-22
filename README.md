![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

# Fall 2026 Paper Implementations

# Grid-Agent: An LLM-Powered Multi-Agent System for Power Grid Control

## 📌 Project Summary
This paper proposes Grid-Agent, a multi-agent framework that uses large language models to detect and fix power grid violations such as voltage problems, overloaded lines, and disconnected buses. Instead of relying purely on classical optimization solvers, the system pairs an LLM acting as a planner with a real power flow engine, so the LLM's reasoning always gets checked against the actual physics of the grid before anything is accepted. I reimplemented the core architecture in Python using pandapower for the power flow side and a swappable LLM client (Claude, GPT, or Gemini) for the planning side.

## 🎯 Motivation
Power outages cost the US economy over $150 billion a year, and grids are only getting harder to manage as more renewables, EVs, and distributed devices get added. Traditional optimization methods (like OPF solvers) are mathematically solid but rigid. They struggle to incorporate operator judgment or adapt quickly to new situations, and they treat each problem in isolation rather than reasoning about the grid holistically. The paper's bet is that LLMs can add a layer of contextual, semantic reasoning on top of these solvers, without giving up the numerical guarantees that make grid control safe.

## 🧩 Novelty
- **Agent decomposition**: Instead of one model trying to do everything, the work is split across five specialized agents (Topology, Planner, Executor, Validator, Summarizer), each handling one part of the violation resolution pipeline.
- **Adaptive multi-scale representation**: Small networks get fully serialized for the LLM, while larger ones get compressed into a semantic graph that focuses detail near the actual violations. This keeps the prompt size manageable as networks scale up.
- **Safety-first execution**: Every proposed action runs in a sandboxed copy of the network first. If a plan doesn't actually improve things or introduces new violations, it gets rolled back automatically rather than applied.

## 🧠 Methodology
1. **Dataset**: No traditional dataset here. Instead, the work uses standard power systems benchmark networks: the [IEEE 30-bus](https://labs.ece.uw.edu/pstca/pf30/pg_tca30bus.htm) transmission system and the [CIGRE Medium Voltage](https://www.cigre.org/) distribution network, both loaded directly through pandapower's built-in test case library. Violation scenarios (voltage issues, thermal overloads, disconnected buses) are injected on top of these networks by scaling loads and tripping lines.
2. **Architecture**: Five agents built around a shared pandapower network state:
   - **Topology Agent** runs the power flow and detects violations against voltage and thermal limits.
   - **Planner Agent** is the LLM. It receives the current violations plus the available actions (switch toggles, battery placement, load curtailment) and returns a plan as structured JSON.
   - **Executor Agent** applies that plan to a sandboxed copy of the network and reruns the power flow.
   - **Validator Agent** checks whether violations actually improved and whether any new ones appeared, deciding whether to keep the change or roll it back.
   - **Summarizer Agent** writes a plain-language explanation of what happened and packages the episode as a record for future fine-tuning.
3. **Evaluation**:
   - Ran the full pipeline on both benchmark networks across multiple injected severity levels (light, medium, severe, fully disconnected).
   - Verified the rollback logic specifically by feeding the Executor a deliberately ineffective plan and confirming the Validator rejected it and reverted state, then fed it a plan that actually targeted the right loads and confirmed it was accepted.
   - Tested the LLM client against multiple providers to confirm the planner logic isn't tied to one specific model.
4. **Metrics**:
   - Resolution success (all violations cleared or not)
   - Number of planning iterations needed to converge
   - Whether a given plan was accepted or rolled back by the Validator
   - Number of actions used per successful resolution

#### Additional Methodology:
- **Provider-agnostic LLM client**: Built a small wrapper so the Planner agent can call Anthropic, OpenAI, or Gemini models interchangeably, since the original paper benchmarks across six different LLMs and I wanted the implementation to support that same kind of comparison.

## 🌍 Impact
This shows a practical pattern for using LLMs in safety-critical infrastructure: let the model reason and propose, but never let it act without a numerical solver checking the result first. That combination of semantic reasoning and hard physical validation is probably the more realistic near-term path for AI in grid operations, compared to either fully rule-based systems or letting a model act unchecked. It also produces explanations alongside its actions, which matters a lot for operators who need to trust and audit automated decisions on critical infrastructure.

#### Future Work
- **Scaling to larger networks**: Testing against bigger distribution feeders (like IEEE 69-bus) to see how well the semantic graph representation holds up as networks grow well past CIGRE MV size.
- **Optimality benchmarking**: Comparing the LLM's action plans directly against a traditional OPF solver to see how close the heuristic solutions actually get to optimal, and where the gap matters most.

**Additional Sources:**
- [Grid-Agent paper (arXiv:2508.05702)](https://arxiv.org/abs/2508.05702)
- [pandapower documentation](https://pandapower.readthedocs.io/) — power flow library used for the physics-based validation layer
- [IEEE 30-Bus Test Case (Power Systems Test Case Archive, UW)](https://labs.ece.uw.edu/pstca/pf30/pg_tca30bus.htm) — transmission benchmark network used for testing
- [CIGRE Benchmark Networks](https://e-cigre.org/publication/575-benchmark-systems-for-network-integration-of-renewable-and-distributed-energy-resources) — source of the CIGRE Medium Voltage distribution network
- Alsac, O. and Stott, B., "Optimal load flow with steady-state security," *IEEE Transactions on Power Apparatus and Systems*, 1974 — foundational OPF formulation referenced by the paper as the classical baseline this work moves beyond
