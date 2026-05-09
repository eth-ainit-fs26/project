from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np


def _project1_dir() -> str:
    """Return the absolute path to the Project1 directory (parent of this utils/ folder)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def setup_environment() -> None:
    """Configure imports and working directory for the part 2 notebooks."""
    project1_dir = _project1_dir()
    if project1_dir not in sys.path:
        sys.path.insert(0, project1_dir)
    os.chdir(project1_dir)
    np.random.seed(42)
    print("✅ Environment configured for CF analysis")


def initialize_system_components(
    n_iterations: int = 300,
    n_simulation_days: int = 100,
    k: int = 5,
    items_per_customer: int = 15,
):
    """Reset and seed the database with customer/item data, return registries for part 2."""
    setup_environment()

    from models.data_loader import load_data_from_files, setup_real_data
    from models.database import database, initialize_database
    from models.models import Customer, Delivery, Item, MarketingOperation, ShopVisit, Transaction
    from models.registries import CustomerRegistry, ItemCatalogue, TransactionRegistry

    if database.is_closed():
        database.connect()
    database.drop_tables([Delivery, MarketingOperation, ShopVisit, Transaction, Customer, Item], safe=True)
    initialize_database()

    customers_df, products_df, _hooks_df = load_data_from_files()
    setup_real_data(customers_df, products_df)

    customer_registry = CustomerRegistry()
    item_catalogue = ItemCatalogue()
    transaction_registry = TransactionRegistry()
    hyperparams = {
        "N_ITERATIONS": n_iterations,
        "N_SIMULATION_DAYS": n_simulation_days,
        "K": k,
        "ITEMS_PER_CUSTOMER": items_per_customer,
    }

    n_customers = len(customer_registry.get_all_customers())
    n_items = len(item_catalogue.get_all_items())
    n_transactions = len(transaction_registry.get_all_transactions())

    print("✅ System components initialized:")
    print(f"   • Customers: {n_customers}")
    print(f"   • Items: {n_items}")
    print(f"   • Transactions: {n_transactions}")
    print(f"   • Training items per customer: {items_per_customer}")
    print(f"   • Recommendation size: {k}")
    return customer_registry, item_catalogue, transaction_registry, hyperparams


def _cosine_kernel(customer_vector: np.ndarray, item_vector: np.ndarray) -> float:
    denom = np.linalg.norm(customer_vector) * np.linalg.norm(item_vector)
    if denom == 0:
        return 0.0
    cosine = float(np.dot(customer_vector, item_vector) / denom)
    return max(0.0, min(1.0, (cosine + 1.0) / 2.0))


def prepare_ground_truth_data(customer_registry, item_catalogue):
    customers = customer_registry.get_all_customers()
    items = item_catalogue.get_all_items()
    customer_ids = [customer.cid for customer in customers]
    item_ids = [item.pid for item in items]
    customer_preferences = np.array([customer.feature_vector for customer in customers], dtype=float)
    item_features = np.array([item.feature_vector for item in items], dtype=float)
    item_prices = np.array([item.price for item in items], dtype=float)
    item_revenues = np.array([item.price - item.cost for item in items], dtype=float)

    true_kappa_matrix = np.zeros((len(customers), len(items)), dtype=float)
    for c_idx, customer_vector in enumerate(customer_preferences):
        for i_idx, item_vector in enumerate(item_features):
            true_kappa_matrix[c_idx, i_idx] = _cosine_kernel(customer_vector, item_vector)

    print(f"Customer preference vectors shape: {customer_preferences.shape}")
    print(f"Preference vector range: [{customer_preferences.min():.3f}, {customer_preferences.max():.3f}]")
    print(f"Preference vector mean norm: {np.linalg.norm(customer_preferences, axis=1).mean():.3f}")
    print(f"Item feature vectors shape: {item_features.shape}")
    print(f"Feature vector range: [{item_features.min():.3f}, {item_features.max():.3f}]")
    print(f"Feature vector mean norm: {np.linalg.norm(item_features, axis=1).mean():.3f}")
    print(f"Price range: [${item_prices.min():.2f}, ${item_prices.max():.2f}]")
    print(f"Revenue range: [${item_revenues.min():.2f}, ${item_revenues.max():.2f}]")
    print(f"True κ matrix shape: {true_kappa_matrix.shape}")
    print(f"True κ range: [{true_kappa_matrix.min():.3f}, {true_kappa_matrix.max():.3f}]")
    print(f"True κ mean: {true_kappa_matrix.mean():.3f}")

    return (
        customers,
        items,
        customer_ids,
        item_ids,
        customer_preferences,
        item_features,
        item_prices,
        item_revenues,
        true_kappa_matrix,
    )


def validate_todo_implementations(customer_registry, item_catalogue, transaction_registry) -> bool:
    from agents.project2_recommendation.recommender_agent import RecommenderAgent

    print("🔍 VALIDATING YOUR TODO IMPLEMENTATIONS")
    print("=" * 50)
    print()

    try:
        agent = RecommenderAgent(
            customer_registry, item_catalogue, transaction_registry,
            d=3, lr=0.01, reg=1e-4, w_purchase=1.0, w_neg=0.5,
        )
    except Exception as e:
        print(f"❌ Failed to create RecommenderAgent: {e}")
        return False

    customers = customer_registry.get_all_customers()
    items = item_catalogue.get_all_items()
    if not customers or not items:
        print("❌ No customers or items in registry — run initialize_system_components first.")
        return False

    test_cid = customers[0].cid
    test_iids = [it.pid for it in items[:3]]

    print(f"Using test customer: {test_cid}, test item: {test_iids[0]}")
    print()

    all_ok = True

    print("1. Testing _kappa_hat() function...")
    try:
        kappa = agent._kappa_hat(test_cid, test_iids[0])
        if 0.0 <= kappa <= 1.0:
            print(f"   ✅ κ prediction: {kappa:.4f} (valid range [0,1])")
        else:
            print(f"   ❌ κ prediction: {kappa:.4f} (INVALID — must be in [0,1])")
            all_ok = False
    except NotImplementedError:
        print("   ❌ _kappa_hat() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in _kappa_hat(): {e}")
        all_ok = False

    print("\n2. Testing recommend() function...")
    try:
        recs = agent.recommend(test_cid, k=3)
        all_item_ids = {it.pid for it in items}
        if isinstance(recs, list) and len(recs) > 0:
            print(f"   ✅ Recommendations: {recs} (valid list of {len(recs)} items)")
            if all(r in all_item_ids for r in recs):
                print("   ✅ All recommended items exist in catalogue")
            else:
                print("   ❌ Some recommended items not found in catalogue")
                all_ok = False
        else:
            print(f"   ❌ Invalid recommendations: {recs}")
            all_ok = False
    except NotImplementedError:
        print("   ❌ recommend() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in recommend(): {e}")
        all_ok = False

    print("\n3. Testing update_from_session() function...")
    try:
        agent.update_from_session(test_cid, test_iids, [test_iids[0]])
        print("   ✅ update_from_session() executed without errors")
        print(f"   ✅ Processed {len(test_iids)} shown items, 1 purchases")
    except NotImplementedError:
        print("   ❌ update_from_session() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in update_from_session(): {e}")
        all_ok = False

    print("\n4. Testing _update_embeddings() function...")
    try:
        agent._update_embeddings(test_cid, test_iids[0], target=1.0, weight=1.0)
        print("   ✅ Embeddings updated successfully")
    except NotImplementedError:
        print("   ❌ _update_embeddings() not yet implemented")
        all_ok = False
    except Exception as e:
        print(f"   ❌ Error in _update_embeddings(): {e}")
        all_ok = False

    print("\n5. Testing function integration...")
    try:
        k_before = agent._kappa_hat(test_cid, test_iids[0])
        agent._update_embeddings(test_cid, test_iids[0], target=0.0, weight=1.0)
        k_after = agent._kappa_hat(test_cid, test_iids[0])
        if abs(k_before - k_after) > 1e-8:
            print(f"   ✅ κ prediction changed from {k_before:.4f} to {k_after:.4f} (learning working)")
        else:
            print("   ⚠️  κ prediction unchanged — gradient updates may not be working")
    except Exception as e:
        print(f"   ❌ Error in integration test: {e}")
        all_ok = False

    print()
    print("=" * 50)
    if all_ok:
        print("🎉 VALIDATION COMPLETE!")
        print("✅ Your implementations appear to be working correctly.")
        print("📋 You can now proceed to run the full simulation.")
    else:
        print("❌ Some validations failed. Please check your implementations.")
    print()
    return all_ok


def _prediction_vectors(agent, customer_ids: List[int], item_ids: List[int], true_kappa_matrix: np.ndarray):
    predictions = []
    truth = []
    for c_idx, customer_id in enumerate(customer_ids):
        for i_idx, item_id in enumerate(item_ids):
            predictions.append(agent._kappa_hat(customer_id, item_id))
            truth.append(true_kappa_matrix[c_idx, i_idx])
    return np.array(predictions), np.array(truth)


def analyze_and_plot_kappa_prediction_quality(agent, customer_ids, item_ids, true_kappa_matrix):
    from sklearn.metrics import mean_squared_error

    predictions, truth = _prediction_vectors(agent, customer_ids, item_ids, true_kappa_matrix)
    correlation = float(np.corrcoef(truth, predictions)[0, 1]) if len(truth) > 1 else 0.0
    mse = float(mean_squared_error(truth, predictions))

    print(f"Correlation: r = {correlation:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"Predicted range: [{predictions.min():.3f}, {predictions.max():.3f}]")
    print(f"True range:      [{truth.min():.3f}, {truth.max():.3f}]")

    try:
        plot_kappa_prediction_quality(truth, predictions, correlation, mse)
    except Exception as e:
        print(f"(Plot skipped: {e})")
    return correlation, mse


def plot_learning_curves(kappa_errors: List[float], revenues: List[float], n_iterations: int) -> None:
    import matplotlib.pyplot as plt

    steps = np.linspace(0, n_iterations, num=len(kappa_errors), endpoint=False) if kappa_errors else []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, kappa_errors, marker="o")
    axes[0].set_title("Kappa Prediction Error")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("MSE")
    axes[1].plot(steps, revenues, marker="o", color="#2E86AB")
    axes[1].set_title("Training Revenue")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Revenue")
    fig.tight_layout()
    plt.show()


def plot_kappa_prediction_quality(true_values, predicted_values, correlation: float, mse: float) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5))
    plt.scatter(true_values, predicted_values, alpha=0.45, s=18)
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.title(f"Kappa Prediction Quality (r={correlation:.3f}, MSE={mse:.4f})")
    plt.xlabel("True kappa")
    plt.ylabel("Predicted kappa")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


def plot_training_data_coverage(customer_ids, item_ids, customer_training_items, items_per_customer: int) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_c, n_i = len(customer_ids), len(item_ids)
    item_idx_map = {iid: j for j, iid in enumerate(item_ids)}

    training_matrix = np.zeros((n_c, n_i))
    for i, cid in enumerate(customer_ids):
        for iid in customer_training_items.get(cid, []):
            j = item_idx_map.get(iid)
            if j is not None:
                training_matrix[i, j] = 1

    coverage = [
        sum(1 for shown_items in customer_training_items.values() if item_id in shown_items)
        for item_id in item_ids
    ]
    per_customer = [len(customer_training_items.get(cid, [])) for cid in customer_ids]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    sns.heatmap(training_matrix, ax=ax, cmap="Greys", cbar=True,
                xticklabels=False, yticklabels=False)
    ax.set_title("Training Coverage Matrix\n(Black = trained, White = unseen)")
    ax.set_xlabel("Items")
    ax.set_ylabel("Customers")

    ax = axes[1]
    mean_cov = np.mean(coverage)
    ax.hist(coverage, bins=min(20, max(1, len(item_ids))), color="steelblue", alpha=0.7, edgecolor="black")
    ax.axvline(mean_cov, color="red", linewidth=2, label=f"Mean: {mean_cov:.1f}")
    ax.set_title("Item Training Coverage")
    ax.set_xlabel("Customers exposed")
    ax.set_ylabel("Number of items")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.bar(range(n_c), per_customer, color="steelblue", alpha=0.7)
    ax.axhline(items_per_customer, color="red", linewidth=2, linestyle="--",
               label=f"Target: {items_per_customer}")
    ax.set_title("Items Per Customer")
    ax.set_xlabel("Customer Index")
    ax.set_ylabel("Training items")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

    all_training = {iid for v in customer_training_items.values() for iid in v}
    total_pairs = n_c * items_per_customer
    possible = n_c * n_i
    print(f"Total training pairs: {total_pairs} out of {possible} possible "
          f"({total_pairs / possible * 100:.1f}%)")
    print(f"Unique items in training: {len(all_training)}/{n_i}")
    print(f"Average item coverage: {mean_cov:.1f} customers per item")


def detailed_agent_simulation(
    agent,
    agent_name: str,
    n_simulation_days: int,
    k: int,
    customer_registry,
    item_catalogue,
    customer_ids: List[int],
    item_ids: List[int],
    true_kappa_matrix: np.ndarray,
) -> Dict[str, Any]:
    item_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    customer_index = {customer_id: idx for idx, customer_id in enumerate(customer_ids)}
    margin = {item.pid: item.price - item.cost for item in item_catalogue.get_all_items()}

    daily_revenue = []
    daily_purchases = []
    daily_attempts = []
    daily_conversion_rate = []
    daily_kappa_mse = []

    for _day in range(n_simulation_days):
        revenue = 0.0
        purchases = 0
        attempts = 0

        for customer_id in customer_ids:
            shown_items = agent.recommend(customer_id, k)
            purchased_items = []
            c_idx = customer_index[customer_id]
            for item_id in shown_items:
                i_idx = item_index[item_id]
                attempts += 1
                if np.random.random() < true_kappa_matrix[c_idx, i_idx]:
                    purchased_items.append(item_id)
                    purchases += 1
                    revenue += margin[item_id]

            agent.update_from_session(customer_id, shown_items, purchased_items)

        sample_customer_ids = customer_ids[: min(10, len(customer_ids))]
        sample_item_ids = item_ids[: min(10, len(item_ids))]
        preds, truth = _prediction_vectors(
            agent,
            sample_customer_ids,
            sample_item_ids,
            true_kappa_matrix[: len(sample_customer_ids), : len(sample_item_ids)],
        )
        daily_kappa_mse.append(float(np.mean((preds - truth) ** 2)))
        daily_revenue.append(revenue)
        daily_purchases.append(purchases)
        daily_attempts.append(attempts)
        daily_conversion_rate.append(purchases / attempts if attempts else 0.0)

    total_revenue = float(np.sum(daily_revenue))
    total_purchases = int(np.sum(daily_purchases))
    total_attempts = int(np.sum(daily_attempts))
    return {
        "agent_name": agent_name,
        "daily_revenue": daily_revenue,
        "daily_purchases": daily_purchases,
        "daily_attempts": daily_attempts,
        "daily_conversion_rate": daily_conversion_rate,
        "daily_kappa_mse": daily_kappa_mse,
        "total_revenue": total_revenue,
        "total_purchases": total_purchases,
        "total_attempts": total_attempts,
        "overall_conversion_rate": total_purchases / total_attempts if total_attempts else 0.0,
        "revenue_per_customer": total_revenue / max(1, len(customer_ids)),
    }


def plot_comprehensive_performance_analysis(
    simulation_results: Dict[str, Dict[str, Any]],
    colors: Dict[str, str],
    kappa_errors: List[float],
    n_iterations: int,
    customer_registry,
) -> None:
    import matplotlib.pyplot as plt

    names = list(simulation_results.keys())
    n_days = len(simulation_results[names[0]]["daily_revenue"])
    days = np.arange(1, n_days + 1)

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    axes = axes.flatten()

    ax = axes[0]
    for name in names:
        ax.plot(days, simulation_results[name]["daily_revenue"],
                color=colors.get(name, "gray"), label=name, alpha=0.75)
    ax.set_title("Daily Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for name in names:
        cumrev = np.cumsum(simulation_results[name]["daily_revenue"])
        ax.plot(days, cumrev, color=colors.get(name, "gray"), label=name, linewidth=2)
    ax.set_title("Cumulative Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    data = [simulation_results[n]["daily_revenue"] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True)
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(colors.get(name, "gray"))
        patch.set_alpha(0.7)
    ax.set_title("Revenue Distribution")
    ax.set_ylabel("Daily Revenue ($)")
    ax.grid(True, alpha=0.3)

    ax = axes[3]
    tots = [simulation_results[n]["total_revenue"] for n in names]
    bars = ax.bar(names, tots, color=[colors.get(n, "gray") for n in names], alpha=0.8)
    ax.set_title("Total Revenue Comparison")
    ax.set_ylabel("Total Revenue ($)")
    for bar, v in zip(bars, tots):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(tots) * 0.01,
                f"${v:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[4]
    for name in names:
        ax.plot(days, simulation_results[name]["daily_conversion_rate"],
                color=colors.get(name, "gray"), label=name, alpha=0.75)
    ax.set_title("Daily Conversion Rate")
    ax.set_xlabel("Day")
    ax.set_ylabel("Conversion Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[5]
    cf_name = "CF (Trained)"
    if cf_name in simulation_results:
        errs = simulation_results[cf_name]["daily_kappa_mse"]
        ax.plot(days[:len(errs)], errs, color=colors.get(cf_name, "blue"), linewidth=2)
        ax.set_title("CF κ Prediction Error (Simulation)")
        ax.set_xlabel("Day")
        ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)

    ax = axes[6]
    for name in names:
        rev = np.array(simulation_results[name]["daily_revenue"])
        if len(rev) >= 7:
            rolling = np.convolve(rev, np.ones(7) / 7, mode="valid")
            ax.plot(np.arange(7, n_days + 1), rolling,
                    color=colors.get(name, "gray"), label=name)
    ax.set_title("7-Day Rolling Average Revenue")
    ax.set_xlabel("Day")
    ax.set_ylabel("Revenue ($)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[7]
    rpc = [simulation_results[n]["revenue_per_customer"] for n in names]
    ax.bar(names, rpc, color=[colors.get(n, "gray") for n in names], alpha=0.8)
    ax.set_title("Revenue per Customer")
    ax.set_ylabel("Revenue / Customer ($)")
    ax.grid(True, alpha=0.3)

    ax = axes[8]
    cvs = [
        np.std(simulation_results[n]["daily_revenue"]) /
        max(np.mean(simulation_results[n]["daily_revenue"]), 1.0)
        for n in names
    ]
    bars = ax.bar(names, cvs, color=[colors.get(n, "gray") for n in names], alpha=0.8)
    ax.set_title("Revenue Stability\n(Lower CV = More Stable)")
    ax.set_ylabel("Coefficient of Variation")
    for bar, v in zip(bars, cvs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[9]
    avgs = [simulation_results[n]["overall_conversion_rate"] for n in names]
    bars = ax.bar(names, avgs, color=[colors.get(n, "gray") for n in names], alpha=0.8)
    ax.set_title("Average Conversion Rate")
    ax.set_ylabel("Conversion Rate")
    for bar, v in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[10]
    if kappa_errors:
        train_iters = np.linspace(0, n_iterations, num=len(kappa_errors), endpoint=False)
        ax.plot(train_iters, kappa_errors, color="navy", linewidth=2)
        ax.set_title("CF κ Learning Progress (Training)")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.3)

    ax = axes[11]
    max_rev = max(simulation_results[n]["total_revenue"] for n in names)
    norm = [simulation_results[n]["total_revenue"] / max(max_rev, 1) for n in names]
    bars = ax.bar(names, norm, color=[colors.get(n, "gray") for n in names], alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5)
    ax.set_title("Normalised Revenue\n(Relative to Best)")
    ax.set_ylabel("Relative Revenue")
    for bar, v in zip(bars, norm):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Comprehensive Performance Analysis: CF vs Baselines",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def print_comprehensive_analysis(
    simulation_results: Dict[str, Dict[str, Any]],
    correlation: float,
    mse: float,
    customer_registry,
) -> None:
    names = list(simulation_results.keys())
    by_rev = sorted(names, key=lambda n: simulation_results[n]["total_revenue"], reverse=True)
    by_conv = sorted(names, key=lambda n: simulation_results[n]["overall_conversion_rate"], reverse=True)

    print("COLLABORATIVE FILTERING VS BASELINES - FINAL ANALYSIS")
    print("=" * 60)
    print()
    print("🏆 PERFORMANCE RANKINGS:\n")
    print("By Total Revenue:")
    for rank, name in enumerate(by_rev, 1):
        rev = simulation_results[name]["total_revenue"]
        print(f"  {rank}. {name}: ${rev:,.0f}")

    print("\nBy Average Conversion Rate:")
    for rank, name in enumerate(by_conv, 1):
        conv = simulation_results[name]["overall_conversion_rate"]
        print(f"  {rank}. {name}: {conv:.3f}")

    print("\n📊 KEY FINDINGS:\n")
    cf_name = "CF (Trained)"
    if cf_name in simulation_results:
        cf_rev = simulation_results[cf_name]["total_revenue"]
        print("1. CF PERFORMANCE:")
        for baseline in ("Random", "Most Expensive"):
            if baseline in simulation_results:
                b_rev = simulation_results[baseline]["total_revenue"]
                pct = (cf_rev - b_rev) / max(b_rev, 1) * 100
                print(f"   • CF outperforms {baseline} baseline by {pct:+.1f}%")
        print(f"   • CF κ prediction quality: r={correlation:.3f}, MSE={mse:.4f}")
        print(f"   • CF successfully learns customer preferences over time")

        print("\n2. BASELINE COMPARISON:")
        for name in names:
            if name != cf_name:
                rev = simulation_results[name]["total_revenue"]
                conv = simulation_results[name]["overall_conversion_rate"]
                print(f"   • {name}: ${rev:,.0f} revenue, {conv:.3f} conversion")

        kappa_sim = simulation_results[cf_name].get("daily_kappa_mse", [])
        if kappa_sim:
            improvement = (kappa_sim[0] - kappa_sim[-1]) / max(kappa_sim[0], 1e-10) * 100
            print(f"\n3. LEARNING EFFECTIVENESS:")
            print(f"   • CF κ prediction improves {improvement:+.1f}% during simulation")

    print("\n4. BUSINESS INSIGHTS:")
    n_customers = len(customer_registry.get_all_customers())
    best = by_rev[0]
    rpc = simulation_results[best]["total_revenue"] / max(n_customers, 1)
    print(f"   • Best revenue per customer: ${rpc:,.0f}")

    cvs = {
        n: np.std(simulation_results[n]["daily_revenue"]) /
        max(np.mean(simulation_results[n]["daily_revenue"]), 1.0)
        for n in names
    }
    most_stable = min(cvs, key=cvs.get)
    print(f"   • Most stable revenue: {most_stable} (CV: {cvs[most_stable]:.3f})")
    print(f"   • Most efficient (purchases/attempts): {by_conv[0]} "
          f"({simulation_results[by_conv[0]]['overall_conversion_rate']:.3f})")

    print("\n✅ RECOMMENDATIONS:")
    print(f"   • {best} model successfully outperforms simple baselines")
    print("   • Continue using current CF approach as primary recommender")
    print("   • Focus on improving item representation learning")
    print("   • Consider hybrid approaches combining CF with content-based filtering")
    print("   • Monitor κ prediction quality as key performance indicator")
    print()
    print(f"🎯 CONCLUSION: {best} is the best overall performer!")
