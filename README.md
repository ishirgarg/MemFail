# MemFail: Stress-Testing Failure Modes of LLM Memory Systems

<div align="center">

[🤗 Dataset](https://huggingface.co/datasets/ishirgarg/MemFail) ·
[📄 Paper](https://arxiv.org/abs/2605.26667)

</div>

LLM agents increasingly rely on external memory systems to stay consistent across long-horizon interactions — but the very mechanisms that make these systems efficient (compressing experiences, updating facts, forgetting noise) also introduce new ways to fail. **MemFail** is a diagnostic benchmark that *isolates* these failure modes instead of hiding them behind aggregate accuracy.

We formalize any memory system as the composition of three canonical operations — **summarization**, **storage**, and **retrieval** — and identify the natural failure modes induced by each. We then construct five adversarial datasets across four tasks, where each task is hand-designed to elicit one specific failure mode. This lets us attribute every wrong answer to a concrete architectural cause, rather than treating the memory system as a black box.

We evaluate four state-of-the-art memory systems — **Mem0**, **A-MEM**, **SimpleMem**, and **StructMem** — and surface findings that aggregate benchmarks would otherwise obscure:

- **No single architecture dominates.** Each system has a distinctive failure signature: graph-based StructMem excels at causal reasoning but collapses on coexisting-fact retrieval; Mem0 shows the opposite pattern.
- **Scaling doesn't fix it.** Increasing the number of retrieved memories or upgrading the underlying LLM yields little improvement, and sometimes degrades performance. Current systems are bound by *architectural* constraints, not model intelligence or context budget.
- **More tokens isn't always better.** Verbose memories help on summary-bottlenecked tasks but hurt retrieval-heavy tasks, where large memories pollute the embedding space.

Based on these findings, we propose two under-explored directions: **mixture-of-memories architectures** that route different experiences to different substores, and **task-adaptive token scaling** that sizes memories to the type of incoming information.

## Failure Modes & Tasks

MemFail provides five datasets across four tasks, each targeting a specific failure mode:

| Task | Targets | What it tests |
| --- | --- | --- |
| **Conditional-Facts (Easy)** | Storage / retrieval | Faithful retention of qualifying conditions on causal facts |
| **Conditional-Facts (Hard)** | Summary | Reconstructing a rule decomposed across non-adjacent sentences |
| **Coexisting-Facts** | Storage + retrieval | Storing and retrieving *all* compatible facts (e.g., multiple preferences) without overwriting |
| **Persona-Retrieval** | Storage / summary | Returning idiosyncratic persona details — and abstaining on misleading queries about unknown people |
| **Long-Hop** | Retrieval | Composing strictly transitive chains of facts scattered across separate storage events |

Any memory system that implements three functions — `store_conversation`, `retrieve_memories`, and `get_all_memories` — plugs directly into the evaluation harness.

## Installation

```bash
conda create -n memfail python=3.11
conda activate memfail
pip install uv
uv sync
```

Set up your environment variables (required for OpenAI usage):
```bash
cat > .env << 'EOF'
OPENAI_API_KEY=sk-your-openai-key
EOF
```

## Quick Start

Each MemFail task ships with an end-to-end pipeline script that generates (or loads) the dataset, evaluates the selected memory systems, and grades the results with an LLM-as-a-judge:

```bash
./conditional_facts.sh          # Summary failures
./coexisting_facts.sh            # Storage + retrieval failures
./long_hop.sh                    # Long-range retrieval failures
./personal_retrieval.sh          # Persona storage failures
```

Each script accepts flags to skip individual memory systems, pin a specific test-taker or judge model, and control retrieval depth `k`. See the comment block at the top of any script for the full option list.

To run a single evaluation in Python:

```bash
uv run python examples/evaluation_example.py
```

To explore memory systems interactively in a chat UI:

```bash
uv run streamlit run app/chat_ui.py -- --memory mem0 --llm openai --model gpt-4o-mini
```

## Datasets

All five datasets are released on the [🤗 Hub](https://huggingface.co/datasets/ishirgarg/MemFail) and mirrored under `datasets/`:

- `datasets/conditional_facts/easy/` — Conditional-Facts (Easy)
- `datasets/conditional_facts/hard/` — Conditional-Facts (Hard)
- `datasets/coexisting_facts/` — Coexisting-Facts (N ∈ {2,3,4,5})
- `datasets/custom_persona_retrieval/` — Persona-Retrieval (direct + misleading queries)
- `datasets/long_hop/` — Long-Hop (K ∈ {1,2,3})

## Code Structure

- `src/types.py` — Core protocols: `MemorySystem`, `LLM`, `EvaluationPromptTemplate`.
- `src/memory.py` — Memory system wrappers (Mem0, A-MEM, SimpleMem, StructMem, plus stateless baselines).
- `src/evaluation.py` — Evaluation harness; runs the storage → query → grading loop.
- `src/llm.py` — Backend adapters for OpenAI, Anthropic, and Ollama.
- `src/dataset.py` — Dataset loaders for the `conversations[].queries[]` format used by the harness.
- `src/prompt_templates.py` — Controls how retrieved memories and history are injected into the LLM prompt.
- `playground/<task>/evaluate_*.py` — Per-task evaluation entrypoint.
- `playground/<task>/analyze_errors.py` — LLM-as-a-judge grading that classifies each failure into summary / storage / retrieval / reasoning errors.
- `generate_dataset.py` & `dataset_utils/` — Adversarial dataset generation pipelines.
- `app/chat_ui.py` — Streamlit interface for inspecting prompts and retrieved memories live.

## Adding a New Memory System

Implement the three-function API from the paper:

```python
class MyMemorySystem:
    def store_conversation(self, conversation): ...
    def retrieve_memories(self, query, conversation, k): ...
    def get_all_memories(self): ...
```

Wrap it in the `MemorySystem` protocol in `src/memory.py`, register it in the evaluation scripts, and the harness — including failure-mode attribution — works automatically.

## Questions

For questions or suggestions, please open an issue on GitHub.

## Citation

If you use MemFail in your research, please cite:

```bibtex
@article{garg2026memfail,
  title={MemFail: Stress-Testing Failure Modes of LLM Memory Systems},
  author={Garg, Ishir and Kolhe, Neel and Song, Dawn and Zhao, Xuandong},
  journal={arXiv preprint arXiv:2605.26667},
  year={2026},
  url={https://arxiv.org/abs/2605.26667}
}
```
