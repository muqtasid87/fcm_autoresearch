"""
Comprehensive Results Analysis for FCM Quadrature Weight Neural Networks

Analyzes trained models from the hyperparameter search:
- Per-weight accuracy (w1, w2, w3, w4)
- Error distributions and outlier analysis
- Hyperparameter impact on performance
- Model comparison plots
- Basis function moment verification

Usage:
    python analyze_results.py <search_directory>

"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
import sys


class FCMResultsAnalyzer:
    """
    Comprehensive analyzer for FCM quadrature weight prediction models.
    """

    def __init__(self, search_dir):
        self.search_dir = Path(search_dir)
        self.results = self._load_results()
        self.output_dir = self.search_dir / 'analysis'
        self.output_dir.mkdir(exist_ok=True)

        if not self.results:
            raise ValueError(f"No results found in {search_dir}")

        print(f"Loaded {len(self.results)} model results from {search_dir}")
        print(f"Analysis output: {self.output_dir}")

    def _load_results(self):
        results_file = self.search_dir / 'all_results.json'
        if not results_file.exists():
            print(f"Error: {results_file} not found!")
            return []
        with open(results_file, 'r') as f:
            return json.load(f)

    def _load_model_predictions(self, model_result):
        """Load saved predictions and ground truth for a model."""
        model_dir = Path(model_result['output_dir'])
        y_pred_path = model_dir / 'y_pred_valid.npy'
        y_true_path = model_dir / 'y_true_valid.npy'

        data = {}
        if y_pred_path.exists():
            data['y_pred'] = np.load(y_pred_path)
        if y_true_path.exists():
            data['y_true'] = np.load(y_true_path)
        return data

    def _load_history(self, model_result):
        """Load training history for a model."""
        history_path = Path(model_result['output_dir']) / 'history.pickle'
        if history_path.exists():
            with open(history_path, 'rb') as f:
                return pickle.load(f)
        return None

    def create_dataframe(self):
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            row = {
                'model_name': r['model_name'],
                'model_width': r['config']['model_width'],
                'model_depth': r['config']['model_depth'],
                'learning_rate': r['config']['learning_rate'],
                'batch_size': r['config']['batch_size_train'],
                'activation': r['config']['activation'],
                'weight_initializer': r['config'].get('weight_initializer', 'glorot_uniform'),
                'num_epochs_config': r['config']['num_epochs'],
                'actual_epochs': r.get('actual_epochs', r['config']['num_epochs']),
                'stopped_early': r.get('stopped_early', False),
                'total_params': r['total_params'],
                'best_val_loss': r['best_val_loss'],
                'final_val_loss': r['final_val_loss'],
                'mean_abs_rel_error': r['mean_abs_rel_error'],
                'median_abs_rel_error': r['median_abs_rel_error'],
                'max_abs_rel_error': r['max_abs_rel_error'],
                'mean_abs_error': r.get('mean_abs_error', np.nan),
                'median_abs_error': r.get('median_abs_error', np.nan),
                'training_time': r['training_time_seconds'],
                'inference_time': r['inference_time_mean'],
                'best_epoch': r['best_epoch'],
                'gpu_id': r.get('gpu_id', None),
            }
            data.append(row)
        return pd.DataFrame(data)

    def run_full_analysis(self):
        """Run all analyses and generate all plots."""
        df = self.create_dataframe()

        print("\n" + "=" * 80)
        print("  COMPREHENSIVE FCM QUADRATURE WEIGHT NN ANALYSIS")
        print("=" * 80)

        # 1. Summary statistics
        self._print_summary_statistics(df)

        # 2. Hyperparameter impact
        self._print_hyperparameter_impact(df)

        # 3. Model ranking table
        self._print_model_ranking(df)

        # 4. Generate all plots
        print("\nGenerating plots...")
        self._plot_loss_curves_grid()
        self._plot_efficiency_analysis(df)

        # 5. Per-model detailed analysis (per-weight errors, moments)
        self._analyze_all_models_predictions()

        # 6. Best model detailed report
        self._best_model_detailed_analysis()

        # 7. Export CSV
        self._export_csv(df)

        # 8. Recommendations
        self._print_recommendations(df)

        print("\n" + "=" * 80)
        print(f"  ANALYSIS COMPLETE - All outputs in: {self.output_dir}")
        print("=" * 80 + "\n")

    # -------------------------------------------------------------------------
    # Summary & Tables
    # -------------------------------------------------------------------------

    def _print_summary_statistics(self, df):
        print("\n" + "-" * 60)
        print("SUMMARY STATISTICS")
        print("-" * 60)

        metrics = {
            'Val Loss': 'best_val_loss',
            'Mean Abs Rel Error': 'mean_abs_rel_error',
            'Median Abs Rel Error': 'median_abs_rel_error',
            'Training Time (s)': 'training_time',
            'Parameters': 'total_params',
        }
        for label, col in metrics.items():
            if col in df.columns and not df[col].isna().all():
                print(f"\n  {label}:")
                print(f"    Best   : {df[col].min():.6g}")
                print(f"    Worst  : {df[col].max():.6g}")
                print(f"    Mean   : {df[col].mean():.6g}")
                print(f"    Median : {df[col].median():.6g}")
                print(f"    Std    : {df[col].std():.6g}")

        early_stopped = df['stopped_early'].sum()
        print(f"\n  Early Stopped: {early_stopped}/{len(df)}")
        if early_stopped > 0:
            avg_early_epoch = df.loc[df['stopped_early'], 'actual_epochs'].mean()
            print(f"  Avg early stop epoch: {avg_early_epoch:.0f}")

    def _print_hyperparameter_impact(self, df):
        print("\n" + "-" * 60)
        print("HYPERPARAMETER IMPACT (mean val loss per group)")
        print("-" * 60)

        for param in ['model_width', 'model_depth', 'learning_rate',
                      'batch_size', 'activation', 'weight_initializer']:
            if param in df.columns:
                grouped = df.groupby(param)['best_val_loss'].agg(['mean', 'min', 'count'])
                print(f"\n  {param}:")
                for idx, row in grouped.iterrows():
                    print(f"    {idx:>15s} : mean={row['mean']:.6e}, "
                          f"best={row['min']:.6e}, n={int(row['count'])}"
                          if isinstance(idx, str) else
                          f"    {idx:>15g} : mean={row['mean']:.6e}, "
                          f"best={row['min']:.6e}, n={int(row['count'])}")

    def _print_model_ranking(self, df):
        print("\n" + "-" * 60)
        print("MODEL RANKING (by validation loss)")
        print("-" * 60)

        df_sorted = df.sort_values('best_val_loss')
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            early = " [ES]" if row['stopped_early'] else ""
            print(f"  {i + 1:2d}. {row['model_name']}{early}")
            print(f"      ValLoss={row['best_val_loss']:.4e}  "
                  f"RelErr={row['mean_abs_rel_error']:.4f}  "
                  f"Epochs={row['actual_epochs']}  "
                  f"Params={row['total_params']:,}")

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def _plot_loss_curves_grid(self):
        """Grid of loss curves for all models."""
        n = len(self.results)
        cols = 6
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        fig.patch.set_facecolor('white')
        if rows == 1:
            axes = axes.reshape(1, -1)

        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])

        for i, r in enumerate(sorted_results):
            ax = axes[i // cols, i % cols]
            history = self._load_history(r)
            if history is not None:
                ax.semilogy(history['loss'], label='Train', alpha=0.7, linewidth=0.8)
                ax.semilogy(history['val_loss'], label='Val', alpha=0.7, linewidth=0.8)
                best_ep = np.argmin(history['val_loss'])
                ax.axvline(x=best_ep, color='green', linestyle='--', alpha=0.4, linewidth=0.5)
                if r.get('stopped_early', False):
                    ax.axvline(x=len(history['loss']) - 1, color='red',
                             linestyle='--', alpha=0.4, linewidth=0.5)
            ax.set_title(f"#{i+1} {r['model_name'].split('_', 2)[-1]}",
                        fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.2)
            if i == 0:
                ax.legend(fontsize=6)

        # Hide unused axes
        for j in range(n, rows * cols):
            axes[j // cols, j % cols].set_visible(False)

        fig.suptitle('Loss Curves for All Models (ranked by val loss)', fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'loss_curves_grid.png', dpi=150,
                   bbox_inches='tight')
        plt.close(fig)
        print("  Saved: loss_curves_grid.png")

    def _plot_efficiency_analysis(self, df):
        """Training time vs performance, inference time vs performance."""
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.patch.set_facecolor('white')

        # 1. Training time vs val loss
        ax = axes[0]
        scatter = ax.scatter(df['training_time'] / 60, df['best_val_loss'],
                           c=df['total_params'], s=80, alpha=0.7, cmap='viridis')
        ax.set_xlabel('Training Time (minutes)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Training Efficiency')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Parameters')

        # 2. Inference time vs val loss
        ax = axes[1]
        scatter = ax.scatter(df['inference_time'] * 1000, df['best_val_loss'],
                           c=df['total_params'], s=80, alpha=0.7, cmap='viridis')
        ax.set_xlabel('Inference Time (ms)')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Inference Efficiency')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Parameters')

        # 3. Parameters vs val loss
        ax = axes[2]
        ax.scatter(df['total_params'], df['best_val_loss'],
                  c='#e74c3c', s=80, alpha=0.7)
        ax.set_xlabel('Total Parameters')
        ax.set_ylabel('Validation Loss')
        ax.set_title('Model Size vs Performance')
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'efficiency_analysis.png', dpi=150,
                   bbox_inches='tight')
        plt.close(fig)
        print("  Saved: efficiency_analysis.png")

    # -------------------------------------------------------------------------
    # Per-Model Prediction Analysis
    # -------------------------------------------------------------------------

    def _analyze_all_models_predictions(self):
        """Analyze predictions for all models."""
        sorted_results = sorted(self.results, key=lambda x: x['best_val_loss'])

        # Generate basis function moment comparison
        self._plot_basis_function_moments(sorted_results)

        # Generate detailed plots for top 5 models
        self._plot_top_models_detailed(sorted_results[:5])

    def _plot_basis_function_moments(self, sorted_results):
        """
        Compare integrals of basis functions {1, x, y, xy} using true vs predicted weights.

        2x2 Gauss quadrature on [-1,1]^2:
          Point ordering (from moment_fitting.py tensor product loop):
            w1 -> (-1/sqrt(3), -1/sqrt(3))
            w2 -> (-1/sqrt(3),  1/sqrt(3))
            w3 -> ( 1/sqrt(3), -1/sqrt(3))
            w4 -> ( 1/sqrt(3),  1/sqrt(3))

        Moment of f = sum_i f(x_i, y_i) * w_i
        """
        best = sorted_results[0]
        data = self._load_model_predictions(best)
        if 'y_pred' not in data or 'y_true' not in data:
            print("  Skipping basis function moments: no predictions found.")
            return

        y_pred = data['y_pred']  # (N, 4)
        y_true = data['y_true']  # (N, 4)

        xi = 1.0 / np.sqrt(3.0)
        # Gauss point coordinates: (x_i, y_i) for w1, w2, w3, w4
        gp_x = np.array([-xi, -xi, xi, xi])
        gp_y = np.array([-xi, xi, -xi, xi])

        # Basis function values at the 4 Gauss points: shape (4,) each
        basis = {
            '1':  np.array([1.0, 1.0, 1.0, 1.0]),
            'x':  gp_x,
            'y':  gp_y,
            'xy': gp_x * gp_y,
            '2.5+1.5x+0.5y+2xy': 2.5 + 1.5*gp_x + 0.5*gp_y + 2.0*gp_x*gp_y,
        }

        # Compute moments: for each basis, dot with weights -> (N,)
        true_moments = {}
        pred_moments = {}
        for name, phi in basis.items():
            true_moments[name] = y_true @ phi  # (N,4) @ (4,) -> (N,)
            pred_moments[name] = y_pred @ phi

        n_basis = len(basis)
        # --- Figure: 2 rows x n_basis cols ---
        # Row 1: Predicted vs True moment scatter
        # Row 2: Absolute error distribution (histogram)
        fig, axes = plt.subplots(2, n_basis, figsize=(6 * n_basis, 11))
        fig.patch.set_facecolor('white')
        colors_scatter = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']

        for col, (name, color) in enumerate(zip(basis.keys(), colors_scatter)):
            m_true = true_moments[name]
            m_pred = pred_moments[name]
            err = m_pred - m_true
            abs_err = np.abs(err)

            # Row 1: scatter pred vs true
            ax = axes[0, col]
            ax.scatter(m_true, m_pred, s=1, alpha=0.1, c=color, rasterized=True)
            lims = [min(m_true.min(), m_pred.min()),
                    max(m_true.max(), m_pred.max())]
            ax.plot(lims, lims, 'r-', linewidth=1.5)
            ax.set_xlabel(f'True moment', fontsize=10)
            ax.set_ylabel(f'Predicted moment', fontsize=10)
            label = name if name != '1' else '1 (= area)'
            ax.set_title(f'Moment of f={label}', fontsize=12)
            ax.grid(True, alpha=0.3)
            # R^2
            ss_res = np.sum(err ** 2)
            ss_tot = np.sum((m_true - np.mean(m_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            ax.text(0.05, 0.95,
                    f'R² = {r2:.6f}\nmean |err| = {abs_err.mean():.4e}',
                    transform=ax.transAxes, fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

            # Row 2: error histogram
            ax = axes[1, col]
            ax.hist(err, bins=200, color=color, alpha=0.7, density=True)
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Error (pred - true)', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'f={name}: mean={err.mean():.3e}, std={err.std():.3e}',
                         fontsize=10)
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f'Basis Function Moment Comparison — {best["model_name"]}\n'
            f'Gauss points at (±1/√3, ±1/√3) on [-1,1]²  |  '
            f'Basis: {{1, x, y, xy, 2.5+1.5x+0.5y+2xy}}\n'
            f'Moment of f = Σ f(xᵢ,yᵢ) · wᵢ   →   comparing true vs predicted weights',
            fontsize=13, y=1.02)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'basis_function_moments.png', dpi=150,
                    bbox_inches='tight')
        plt.close(fig)
        print("  Saved: basis_function_moments.png")

        # Print summary table
        print("\n" + "-" * 60)
        print("BASIS FUNCTION MOMENT COMPARISON")
        print("-" * 60)
        print(f"  {'Basis':>6s}  {'Mean|Err|':>12s}  {'Median|Err|':>12s}  "
              f"{'Max|Err|':>12s}  {'R²':>10s}")
        for name in basis.keys():
            err = np.abs(pred_moments[name] - true_moments[name])
            m_true = true_moments[name]
            ss_res = np.sum((pred_moments[name] - m_true) ** 2)
            ss_tot = np.sum((m_true - np.mean(m_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            print(f"  {name:>6s}  {err.mean():12.6e}  {np.median(err):12.6e}  "
                  f"{err.max():12.6e}  {r2:10.8f}")

    def _plot_top_models_detailed(self, top_results):
        """Generate detailed per-model analysis plots for top N models."""
        for rank, r in enumerate(top_results):
            data = self._load_model_predictions(r)
            if 'y_pred' not in data or 'y_true' not in data:
                continue

            y_pred = data['y_pred']
            y_true = data['y_true']
            model_name = r['model_name']

            fig = plt.figure(figsize=(24, 18))
            fig.patch.set_facecolor('white')
            gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)

            weight_names = ['w1', 'w2', 'w3', 'w4']
            area_frac = np.sum(y_true, axis=1) / 4.0

            # Row 1: Per-weight scatter plots (predicted vs true)
            for wi in range(4):
                ax = fig.add_subplot(gs[0, wi])
                ax.scatter(y_true[:, wi], y_pred[:, wi], s=1, alpha=0.2, c='#3498db')
                lims = [min(y_true[:, wi].min(), y_pred[:, wi].min()),
                        max(y_true[:, wi].max(), y_pred[:, wi].max())]
                ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
                ax.set_xlabel(f'True {weight_names[wi]}')
                ax.set_ylabel(f'Predicted {weight_names[wi]}')
                ax.set_title(f'{weight_names[wi]}: Predicted vs True')
                ax.grid(True, alpha=0.3)
                ss_res = np.sum((y_true[:, wi] - y_pred[:, wi]) ** 2)
                ss_tot = np.sum((y_true[:, wi] - np.mean(y_true[:, wi])) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                ax.text(0.05, 0.95, f'R² = {r2:.6f}', transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Row 2: Error distributions per weight
            for wi in range(4):
                ax = fig.add_subplot(gs[1, wi])
                errors = y_pred[:, wi] - y_true[:, wi]
                ax.hist(errors, bins=200, color='#2ecc71', alpha=0.7, density=True)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.set_xlabel(f'Error (pred - true)')
                ax.set_ylabel('Density')
                ax.set_title(f'{weight_names[wi]}: Error Distribution\n'
                            f'mean={np.mean(errors):.4e}, std={np.std(errors):.4e}')
                ax.grid(True, alpha=0.3)

            # Row 3: Absolute error vs area fraction (sum(w_true)/4)
            for wi in range(4):
                ax = fig.add_subplot(gs[2, wi])
                abs_err = np.abs(y_pred[:, wi] - y_true[:, wi])
                ax.scatter(area_frac, abs_err, s=1, alpha=0.1, c='#e74c3c',
                           rasterized=True)
                ax.set_xlabel('Area fraction (sum(w)/4)')
                ax.set_ylabel(f'|error| for {weight_names[wi]}')
                ax.set_title(f'{weight_names[wi]}: Error vs Area Fraction')
                ax.grid(True, alpha=0.3)
                # Binned mean line
                n_bins = 20
                bin_edges = np.linspace(area_frac.min(), area_frac.max(), n_bins + 1)
                bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
                bin_idx = np.clip(np.digitize(area_frac, bin_edges) - 1, 0, n_bins - 1)
                bin_means = np.array([abs_err[bin_idx == b].mean()
                                      if (bin_idx == b).sum() > 0 else 0
                                      for b in range(n_bins)])
                ax.plot(bin_centers, bin_means, 'k-o', markersize=3, linewidth=1.5,
                        label='Bin mean')
                ax.legend(fontsize=8)

            fig.suptitle(f'Rank #{rank + 1}: {model_name}\n'
                        f'Val Loss: {r["best_val_loss"]:.4e} | '
                        f'Mean Rel Err: {r["mean_abs_rel_error"]:.4f}',
                        fontsize=13)

            filename = f'model_rank{rank + 1:02d}_detail.png'
            plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {filename}")

    def _best_model_detailed_analysis(self):
        """Extra detailed report for the best model."""
        best = sorted(self.results, key=lambda x: x['best_val_loss'])[0]
        data = self._load_model_predictions(best)

        if 'y_pred' not in data or 'y_true' not in data:
            return

        y_pred = data['y_pred']
        y_true = data['y_true']

        report = []
        report.append("=" * 70)
        report.append("BEST MODEL DETAILED REPORT")
        report.append("=" * 70)
        report.append(f"\nModel: {best['model_name']}")
        report.append(f"Validation Loss: {best['best_val_loss']:.6e}")
        report.append(f"Parameters: {best['total_params']:,}")
        report.append(f"Best Epoch: {best['best_epoch']}")
        report.append(f"Early Stopped: {best.get('stopped_early', False)}")

        report.append("\n--- Configuration ---")
        for k, v in best['config'].items():
            report.append(f"  {k}: {v}")

        report.append("\n--- Per-Weight Metrics ---")
        weight_names = ['w1', 'w2', 'w3', 'w4']
        for wi in range(4):
            errors = y_pred[:, wi] - y_true[:, wi]
            abs_errors = np.abs(errors)
            eps = 1e-10
            rel_errors = np.abs(errors) / (np.abs(y_true[:, wi]) + eps) * 100

            report.append(f"\n  {weight_names[wi]}:")
            report.append(f"    Mean Abs Error     : {np.mean(abs_errors):.6e}")
            report.append(f"    Median Abs Error   : {np.median(abs_errors):.6e}")
            report.append(f"    Max Abs Error      : {np.max(abs_errors):.6e}")
            report.append(f"    Std Error          : {np.std(errors):.6e}")
            report.append(f"    Mean % Error       : {np.mean(rel_errors):.4f}%")
            report.append(f"    Median % Error     : {np.median(rel_errors):.4f}%")
            report.append(f"    95th %ile % Error  : {np.percentile(rel_errors, 95):.4f}%")
            report.append(f"    99th %ile % Error  : {np.percentile(rel_errors, 99):.4f}%")

            # R^2
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y_true[:, wi] - np.mean(y_true[:, wi])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            report.append(f"    R²                 : {r2:.8f}")

        report_text = "\n".join(report)
        report_path = self.output_dir / 'best_model_report.txt'
        with open(report_path, 'w') as f:
            f.write(report_text)

        print(f"\n{report_text}")
        print(f"\n  Saved: best_model_report.txt")

    # -------------------------------------------------------------------------
    # Export & Recommendations
    # -------------------------------------------------------------------------

    def _export_csv(self, df):
        csv_path = self.output_dir / 'all_models_results.csv'
        df_sorted = df.sort_values('best_val_loss')
        df_sorted.to_csv(csv_path, index=False)
        print(f"  Saved: all_models_results.csv")

    def _print_recommendations(self, df):
        print("\n" + "-" * 60)
        print("RECOMMENDATIONS")
        print("-" * 60)

        # Best accuracy
        best = df.loc[df['best_val_loss'].idxmin()]
        print(f"\n  1. BEST ACCURACY:")
        print(f"     {best['model_name']}")
        print(f"     Val Loss: {best['best_val_loss']:.6e}, "
              f"Rel Err: {best['mean_abs_rel_error']:.6f}, "
              f"Params: {best['total_params']:,}")

        # Fastest inference
        fastest = df.loc[df['inference_time'].idxmin()]
        print(f"\n  2. FASTEST INFERENCE:")
        print(f"     {fastest['model_name']}")
        print(f"     Inference: {fastest['inference_time'] * 1000:.2f}ms, "
              f"Val Loss: {fastest['best_val_loss']:.6e}")

        # Best balance
        vl_norm = (df['best_val_loss'] - df['best_val_loss'].min()) / \
                  (df['best_val_loss'].max() - df['best_val_loss'].min() + 1e-12)
        inf_norm = (df['inference_time'] - df['inference_time'].min()) / \
                   (df['inference_time'].max() - df['inference_time'].min() + 1e-12)
        balance = vl_norm + inf_norm
        best_bal = df.loc[balance.idxmin()]
        print(f"\n  3. BEST BALANCE (Accuracy + Speed):")
        print(f"     {best_bal['model_name']}")
        print(f"     Val Loss: {best_bal['best_val_loss']:.6e}, "
              f"Inference: {best_bal['inference_time'] * 1000:.2f}ms")

        # Smallest model with good performance (top 50%)
        median_loss = df['best_val_loss'].median()
        good_models = df[df['best_val_loss'] <= median_loss]
        if len(good_models) > 0:
            smallest = good_models.loc[good_models['total_params'].idxmin()]
            print(f"\n  4. SMALLEST GOOD MODEL (top 50% accuracy):")
            print(f"     {smallest['model_name']}")
            print(f"     Params: {smallest['total_params']:,}, "
                  f"Val Loss: {smallest['best_val_loss']:.6e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <search_directory>")
        print("\nExample:")
        print("  python analyze_results.py gpu_search/search_20240127_143000")
        return

    search_dir = sys.argv[1]

    try:
        analyzer = FCMResultsAnalyzer(search_dir)
    except Exception as e:
        print(f"Error: {e}")
        return

    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()