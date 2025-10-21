"""
Diagnostic script to analyze AgACI weight dynamics.

This script runs AgACI with detailed logging to understand why weights
are not changing across regime switches.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

HERE = os.path.dirname(__file__)
PROJ = os.path.abspath(os.path.join(HERE, ".."))
for p in [HERE, PROJ]:
    if p not in sys.path:
        sys.path.insert(0, p)

from AdaptiveConformalPredictionsTimeSeries.agaci import pinball_loss, pinball_loss_gradient


def analyze_weight_dynamics():
    """Analyze why weights are not changing."""

    print("\n" + "="*80)
    print("DIAGNOSTIC: AgACI Weight Update Mechanism")
    print("="*80)

    # Simulate a simple scenario
    n_experts = 5
    T = 100

    # Simulate expert predictions (different gamma values give different intervals)
    np.random.seed(42)
    expert_preds = np.random.randn(n_experts, T) + np.arange(n_experts).reshape(-1, 1) * 0.1
    y_true = np.random.randn(T)

    # BOA parameters
    eta = 0.5
    tau = 0.05  # Lower quantile

    # Initialize weights
    weights = np.ones(n_experts) / n_experts
    cumulative_losses = np.zeros(n_experts)

    print(f"\nSimulation setup:")
    print(f"  n_experts: {n_experts}")
    print(f"  T: {T}")
    print(f"  eta (base learning rate): {eta}")
    print(f"  tau (quantile level): {tau}")
    print(f"  Initial weights: {weights}")

    # Track weight evolution
    weight_history = []
    eta_t_history = []
    gradient_history = []

    for t in range(T):
        # Current predictions
        preds = expert_preds[:, t]

        # Compute losses
        losses = np.array([pinball_loss(y_true[t], pred, tau) for pred in preds])
        gradients = np.array([pinball_loss_gradient(y_true[t], pred, tau) for pred in preds])

        # Update cumulative losses
        cumulative_losses += losses

        # Adaptive learning rate
        eta_t = eta / np.sqrt(t + 1)

        # Update log-weights (gradient-based)
        log_weights = np.log(weights) - eta_t * gradients
        log_weights = log_weights - np.max(log_weights)  # numerical stability
        weights = np.exp(log_weights)
        weights = weights / np.sum(weights)

        # Store history
        weight_history.append(weights.copy())
        eta_t_history.append(eta_t)
        gradient_history.append(gradients.copy())

        # Print details for first few and select steps
        if t < 5 or t % 25 == 0:
            print(f"\n  t={t}")
            print(f"    y_true: {y_true[t]:.4f}")
            print(f"    expert_preds: {preds}")
            print(f"    losses: {losses}")
            print(f"    gradients: {gradients}")
            print(f"    eta_t: {eta_t:.6f}")
            print(f"    weights (after update): {weights}")
            print(f"    max gradient magnitude: {np.max(np.abs(gradients)):.6f}")
            print(f"    weight change magnitude: {eta_t * np.max(np.abs(gradients)):.6f}")

    # Analysis
    weight_history = np.array(weight_history)
    eta_t_history = np.array(eta_t_history)
    gradient_history = np.array(gradient_history)

    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Learning rate decay
    print("\nLearning rate decay:")
    print(f"  eta_t[0]:   {eta_t_history[0]:.6f}")
    print(f"  eta_t[10]:  {eta_t_history[10]:.6f}")
    print(f"  eta_t[50]:  {eta_t_history[50]:.6f}")
    print(f"  eta_t[99]:  {eta_t_history[99]:.6f}")
    print(f"  Ratio (t=99 / t=0): {eta_t_history[99] / eta_t_history[0]:.6f}")

    # Weight changes
    weight_changes = np.diff(weight_history, axis=0)
    print("\nWeight change statistics:")
    print(f"  Max absolute change: {np.max(np.abs(weight_changes)):.6f}")
    print(f"  Mean absolute change: {np.mean(np.abs(weight_changes)):.6f}")
    print(f"  Std absolute change: {np.std(np.abs(weight_changes)):.6f}")

    # Gradient statistics
    print("\nGradient statistics:")
    print(f"  Max gradient magnitude: {np.max(np.abs(gradient_history)):.6f}")
    print(f"  Mean gradient magnitude: {np.mean(np.abs(gradient_history)):.6f}")
    print(f"  Gradient range: [{np.min(gradient_history):.6f}, {np.max(gradient_history):.6f}]")

    # Expected weight change
    max_gradient = np.max(np.abs(gradient_history))
    expected_change_early = eta_t_history[10] * max_gradient
    expected_change_late = eta_t_history[99] * max_gradient
    print("\nExpected max weight change (gradient-based):")
    print(f"  At t=10:  eta_t * max_gradient = {expected_change_early:.6f}")
    print(f"  At t=99:  eta_t * max_gradient = {expected_change_late:.6f}")

    # Plot weight evolution
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Weight evolution
    ax = axes[0]
    for i in range(n_experts):
        ax.plot(weight_history[:, i], label=f'Expert {i}', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Weight')
    ax.set_title('Weight Evolution Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Learning rate decay
    ax = axes[1]
    ax.plot(eta_t_history, linewidth=2, color='red')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Learning rate (eta_t)')
    ax.set_title('Adaptive Learning Rate Decay: eta_t = eta / sqrt(t)')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    # Plot 3: Weight change magnitude
    ax = axes[2]
    weight_change_mag = np.linalg.norm(weight_changes, axis=1)
    ax.plot(weight_change_mag, linewidth=2, color='green')
    ax.set_xlabel('Time step')
    ax.set_ylabel('||Δw||')
    ax.set_title('Magnitude of Weight Changes')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    os.makedirs('figures/diagnostics', exist_ok=True)
    plt.savefig('figures/diagnostics/weight_dynamics_diagnostic.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: figures/diagnostics/weight_dynamics_diagnostic.png")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    print("\n1. LEARNING RATE DECAY:")
    print(f"   - The learning rate decreases by {100*(1 - eta_t_history[99]/eta_t_history[0]):.1f}%")
    print(f"   - At t=99, eta_t is only {eta_t_history[99]:.6f} (started at {eta_t_history[0]:.6f})")
    print("   - This sqrt(t) decay is TOO AGGRESSIVE for regime-switching data!")

    print("\n2. WEIGHT CHANGE MAGNITUDE:")
    print(f"   - Max weight change per step: {np.max(np.abs(weight_changes)):.6f}")
    print(f"   - This is very small - weights barely move!")

    print("\n3. RECOMMENDATION:")
    print("   - Use a SLOWER learning rate decay, e.g., eta_t = eta / log(t + 2)")
    print("   - Or use CONSTANT learning rate: eta_t = eta")
    print("   - Or use SLIDING WINDOW: reset weights periodically")

    print("\n" + "="*80)


def compare_learning_rate_schedules():
    """Compare different learning rate schedules."""

    print("\n" + "="*80)
    print("COMPARISON: Learning Rate Schedules")
    print("="*80)

    T = 500
    eta = 0.5

    schedules = {
        'Constant': lambda t: eta,
        '1/sqrt(t)': lambda t: eta / np.sqrt(t + 1),
        '1/log(t)': lambda t: eta / np.log(t + 2),
        '1/(t^0.25)': lambda t: eta / ((t + 1) ** 0.25),
        '1/(t^0.6)': lambda t: eta / ((t + 1) ** 0.6),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot learning rate schedules
    for name, schedule in schedules.items():
        eta_t = [schedule(t) for t in range(T)]
        ax1.plot(eta_t, label=name, linewidth=2)

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Learning rate')
    ax1.set_title('Learning Rate Schedules')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')

    # Plot relative decay (normalized to initial value)
    for name, schedule in schedules.items():
        eta_t = [schedule(t) for t in range(T)]
        eta_t_normalized = np.array(eta_t) / eta_t[0]
        ax2.plot(eta_t_normalized, label=name, linewidth=2)

    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Relative learning rate (eta_t / eta_0)')
    ax2.set_title('Learning Rate Decay (Normalized)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.axhline(0.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='10% of initial')

    plt.tight_layout()
    plt.savefig('figures/diagnostics/learning_rate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n  Saved: figures/diagnostics/learning_rate_comparison.png")

    # Print values at key points
    print("\nLearning rate values at key time points:")
    print(f"{'Schedule':<15} {'t=10':<12} {'t=50':<12} {'t=100':<12} {'t=500':<12}")
    print("-" * 65)
    for name, schedule in schedules.items():
        vals = [schedule(t) for t in [10, 50, 100, 500]]
        print(f"{name:<15} {vals[0]:<12.6f} {vals[1]:<12.6f} {vals[2]:<12.6f} {vals[3]:<12.6f}")


if __name__ == "__main__":
    analyze_weight_dynamics()
    compare_learning_rate_schedules()
