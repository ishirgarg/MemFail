"""Run every plotting script in this directory."""
from __future__ import annotations

import plot_success_vs_k
import plot_errors_vs_k
import plot_coexisting_by_n_facts
import plot_memory_errors_vs_k
import plot_all_errors_vs_k
import plot_acc_vs_tokens_per_memory
import plot_report


if __name__ == "__main__":
    plot_success_vs_k.main()
    plot_errors_vs_k.main()
    plot_coexisting_by_n_facts.main()
    plot_memory_errors_vs_k.main()
    plot_all_errors_vs_k.main()
    plot_acc_vs_tokens_per_memory.main()
    plot_report.main()
