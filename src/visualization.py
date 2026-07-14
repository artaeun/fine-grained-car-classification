# visualization functions for model analysis

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# visualization style constants

# figure dimensions
FIGURE_WIDTH = 10
FIGURE_HEIGHT = 6

# sample grid layout
GRID_COLS = 4
GRID_ROWS = 2

# font sizes
FONT_TITLE = 12
FONT_IMAGE_LABEL = 9
FONT_AXIS = 12
FONT_LEGEND = 9

# line plot styling
LINE_WIDTH = 2

# color scheme
COLOR_CORRECT = '#4CAF50'
COLOR_INCORRECT = '#F44336'
COLOR_WORST = 'salmon'
COLOR_BEST = 'lightgreen'


def save_figure(save_path):
    # save and display figure
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"saved: {path}")
    plt.show()
    plt.close()


def load_metadata(results_dir):
    # load visualization metadata from json

    metadata_path = Path(results_dir) / 'visualization_metadata.json'

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata not found: {metadata_path}\n"
            "run 03_evaluation.ipynb first to generate metadata"
        )

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # convert string keys back to int for class name lookups
    make_idx_to_name = {}
    for k, v in metadata['make_idx_to_name'].items():
        make_idx_to_name[int(k)] = v

    model_idx_to_name = {}
    for k, v in metadata['model_idx_to_name'].items():
        model_idx_to_name[int(k)] = v

    return metadata, make_idx_to_name, model_idx_to_name


def load_evaluation_data(model_name, results_dir):
    # load evaluation data from json

    data_path = Path(results_dir) / model_name / 'evaluation_data.json'

    if not data_path.exists():
        print(f"warning: evaluation data not found for {model_name}")
        return None

    with open(data_path, 'r') as f:
        data = json.load(f)

    # convert lists back to numpy arrays
    data['indices'] = np.array(data['indices'])

    if data['is_hierarchical']:
        data['make_predictions'] = np.array(data['make_predictions'])
        data['make_labels'] = np.array(data['make_labels'])
        data['model_predictions'] = np.array(data['model_predictions'])
        data['model_labels'] = np.array(data['model_labels'])

        # load probs if available
        if 'make_probs' in data:
            data['make_probs'] = np.array(data['make_probs'])
        if 'model_probs' in data:
            data['model_probs'] = np.array(data['model_probs'])

        # for compatibility
        data['predictions'] = data['make_predictions']
        data['labels'] = data['make_labels']
        data['metrics'] = data['make_metrics']

        if 'make_probs' in data:
            data['probs'] = data['make_probs']
    else:
        data['predictions'] = np.array(data['predictions'])
        data['labels'] = np.array(data['labels'])

        # load probs if available
        if 'probs' in data:
            data['probs'] = np.array(data['probs'])

    return data


def load_training_history(model_name, checkpoint_dir):
    # load training history json

    history_path = Path(checkpoint_dir) / f'{model_name}_history.json'

    if not history_path.exists():
        print(f"warning: history not found for {model_name}")
        return None

    with open(history_path, 'r') as f:
        history = json.load(f)

    return history


def get_history_values(history, key):
    # helper to get values from history, handling hierarchical model keys

    if key in history:
        values = history[key]
        return values

    # fallback for hierarchical model
    if key == 'val_acc':
        if 'val_make_acc' in history:
            values = history['val_make_acc']
            return values

    if key == 'train_acc':
        if 'train_make_acc' in history:
            values = history['train_make_acc']
            return values

    result = None
    return result


def get_original_image_path(img_path):
    # convert preprocessed image path to original image path
    original_path = img_path.replace('image_256', 'image')
    return original_path


def truncate_name(name, max_len):
    # truncate name if too long
    is_too_long = len(name) > max_len
    if is_too_long:
        truncated = name[:max_len - 3] + '...'
        result = truncated
    else:
        result = name
    return result


def plot_per_class_accuracy(labels, predictions, class_names, save_path=None,
                           top_n=15, title='Per-Class Accuracy'):
    # plot per-class accuracy bar chart with actual class names

    num_classes = int(max(labels)) + 1
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for i in range(len(labels)):
        label = labels[i]
        pred = predictions[i]
        class_total[label] = class_total[label] + 1
        if pred == label:
            class_correct[label] = class_correct[label] + 1

    class_accuracy = np.zeros(num_classes)
    for i in range(num_classes):
        if class_total[i] > 0:
            class_accuracy[i] = class_correct[i] / class_total[i]
        else:
            class_accuracy[i] = 0

    # only consider classes that have samples
    valid_indices = np.where(class_total > 0)[0]
    valid_accuracies = class_accuracy[valid_indices]

    # sort valid classes by accuracy
    sorted_order = np.argsort(valid_accuracies)
    sorted_valid_indices = valid_indices[sorted_order]

    # get best and worst from valid classes only
    best_indices = sorted_valid_indices[-top_n:][::-1]
    worst_indices = sorted_valid_indices[:top_n]

    fig, axes = plt.subplots(1, 2, figsize=(14, FIGURE_HEIGHT))

    # best classes (left) - longest bars at top
    best_accuracies = []
    best_labels = []
    for idx in best_indices:
        best_accuracies.append(class_accuracy[idx])
        name = class_names.get(idx, f'class_{idx}')
        name = truncate_name(name, 20)
        best_labels.append(name)

    axes[0].barh(range(len(best_accuracies)), best_accuracies, color=COLOR_BEST)
    axes[0].set_yticks(range(len(best_labels)))
    axes[0].set_yticklabels(best_labels, fontsize=8)
    axes[0].set_xlabel('accuracy')
    axes[0].set_title(f'top {top_n} classes')
    axes[0].set_xlim(0, 1)
    axes[0].invert_yaxis()

    # worst classes (right) - shorter bars at top
    worst_accuracies = []
    worst_labels = []
    for idx in worst_indices:
        worst_accuracies.append(class_accuracy[idx])
        name = class_names.get(idx, f'class_{idx}')
        name = truncate_name(name, 20)
        worst_labels.append(name)

    axes[1].barh(range(len(worst_accuracies)), worst_accuracies, color=COLOR_WORST)
    axes[1].set_yticks(range(len(worst_labels)))
    axes[1].set_yticklabels(worst_labels, fontsize=8)
    axes[1].set_xlabel('accuracy')
    axes[1].set_title(f'bottom {top_n} classes')
    axes[1].set_xlim(0, 1)
    axes[1].invert_yaxis()

    plt.suptitle(title, fontsize=FONT_TITLE)
    plt.tight_layout()
    save_figure(save_path)

    return class_accuracy


def plot_sample_grid(images, titles, title_colors, bar_data=None,
                     bar_labels=None, bar_colors=None, main_title='', save_path=None):
    # grid of images with optional bar charts showing top-5 predictions
    # images: list of PIL images
    # titles: list of title strings (e.g. "True: BMW")
    # title_colors: list of title color strings
    # bar_data: optional list of lists with probability values per sample
    # bar_labels: optional list of lists with class names per sample
    # bar_colors: optional list of lists with bar colors per sample

    n_samples = len(images)
    n_rows = GRID_ROWS
    n_cols = GRID_COLS

    # adjust grid if insufficient samples
    if n_samples < n_rows * n_cols:
        n_rows = int(np.ceil(np.sqrt(n_samples)))
        n_cols = int(np.ceil(n_samples / n_rows))

    has_bars = bar_data is not None

    if has_bars:
        # create figure with alternating image/bar rows
        fig = plt.figure(figsize=(3.5 * n_cols, 4.5 * n_rows))
        gs = fig.add_gridspec(n_rows * 2, n_cols, height_ratios=[3, 1] * n_rows,
                              hspace=0.05, wspace=0.3)
    else:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1:
            if n_cols == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.reshape(1, -1)
        else:
            if n_cols == 1:
                axes = axes.reshape(-1, 1)

    for i in range(n_samples):
        row = i // n_cols
        col = i % n_cols

        if has_bars:
            # image subplot
            ax_img = fig.add_subplot(gs[row * 2, col])
            ax_img.imshow(images[i])
            ax_img.axis('off')
            ax_img.set_title(titles[i], fontsize=FONT_IMAGE_LABEL, color=title_colors[i])

            # bar subplot
            ax_bar = fig.add_subplot(gs[row * 2 + 1, col])
            y_pos = np.arange(len(bar_data[i]))
            bars = ax_bar.barh(y_pos, bar_data[i], color=bar_colors[i], height=0.7)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(bar_labels[i], fontsize=7)
            ax_bar.set_xlim(0, 105)
            ax_bar.invert_yaxis()
            ax_bar.tick_params(axis='x', labelsize=6)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
        else:
            ax = axes[row, col]
            ax.imshow(images[i])
            ax.axis('off')
            ax.set_title(titles[i], fontsize=FONT_IMAGE_LABEL, color=title_colors[i])

    # hide unused subplots (only for non-bar case)
    if not has_bars:
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

    plt.suptitle(main_title, fontsize=FONT_TITLE, y=0.99)
    if not has_bars:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(save_path)


def plot_multi_line_overlay(y_data, labels, colors, xlabel, ylabel, title,
                             save_path=None, legend_loc='upper right', hline_y=None):
    # generic multi-line overlay plot

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for i in range(len(y_data)):
        y_values = y_data[i]
        label = labels[i]
        color = colors[i]
        x_values = range(1, len(y_values) + 1)
        ax.plot(x_values, y_values, label=label, color=color, linewidth=LINE_WIDTH)

    if hline_y is not None:
        ax.axhline(y=hline_y, color='black', linestyle='--', alpha=0.5)

    ax.set_xlabel(xlabel, fontsize=FONT_AXIS)
    ax.set_ylabel(ylabel, fontsize=FONT_AXIS)
    ax.set_title(title, fontsize=FONT_TITLE)
    ax.legend(loc=legend_loc, fontsize=FONT_LEGEND)
    ax.grid(True, which='major', alpha=0.3)
    ax.minorticks_on()
    ax.grid(True, which='minor', alpha=0.15, linestyle=':')

    plt.tight_layout()
    save_figure(save_path)


def extract_training_curves(histories, model_names, model_display_names, model_colors, curve_type):
    # extract training curve data from histories
    # curve_type: 'val_acc', 'val_loss', 'gen_gap'
    # returns: y_data, labels, colors (lists for plot_multi_line_overlay)

    y_data = []
    labels = []
    colors = []

    for model_name in model_names:
        history = histories.get(model_name)
        if history is None:
            continue

        if curve_type == 'val_acc':
            values = get_history_values(history, 'val_acc')
            if values is None:
                print(f"warning: no val_acc data for {model_name}, skipping")
                continue

        elif curve_type == 'val_loss':
            if 'val_loss' not in history:
                print(f"warning: no val_loss data for {model_name}, skipping")
                continue
            values = history['val_loss']

        elif curve_type == 'gen_gap':
            train_acc = get_history_values(history, 'train_acc')
            val_acc = get_history_values(history, 'val_acc')
            if train_acc is None:
                print(f"warning: missing train_acc for {model_name}, skipping")
                continue
            if val_acc is None:
                print(f"warning: missing val_acc for {model_name}, skipping")
                continue
            train_arr = np.array(train_acc)
            val_arr = np.array(val_acc)
            values = train_arr - val_arr
            values = values.tolist()

        else:
            print(f"warning: unknown curve_type {curve_type}")
            continue

        display_name = model_display_names.get(model_name, model_name)
        color = model_colors.get(model_name, '#333333')

        y_data.append(values)
        labels.append(display_name)
        colors.append(color)

    return y_data, labels, colors


def compute_generalization_gap_stats(histories, model_names, display_names):
    # compute generalization gap statistics for all models
    # returns gap vectors and dataframe with avg gap for last 10 epochs

    gap_vectors = {}
    stats_data = []

    for model_name in model_names:
        history = histories.get(model_name)
        if history is None:
            continue

        train_acc = get_history_values(history, 'train_acc')
        val_acc = get_history_values(history, 'val_acc')

        if train_acc is None:
            print(f"warning: missing accuracy data for {model_name}, skipping")
            continue
        if val_acc is None:
            print(f"warning: missing accuracy data for {model_name}, skipping")
            continue

        train_acc_arr = np.array(train_acc)
        val_acc_arr = np.array(val_acc)
        gap = train_acc_arr - val_acc_arr

        gap_vectors[model_name] = gap

        # compute statistics
        n_epochs = len(gap)

        # average gap for last 10 epochs
        last_10_start = max(0, n_epochs - 10)
        avg_last_10 = np.mean(gap[last_10_start:])

        display_name = display_names.get(model_name, model_name)

        row = {}
        row['Model'] = display_name
        row['Avg Gap (Last 10)'] = avg_last_10
        stats_data.append(row)

    gap_df = pd.DataFrame(stats_data)

    return gap_vectors, gap_df


def plot_accuracy_comparison(data_dict, model_display_names, save_path=None):
    # plot bar chart comparing model accuracies

    model_names = []
    top1_accs = []
    top5_accs = []

    for model_name in data_dict:
        data = data_dict[model_name]
        display_name = model_display_names.get(model_name, model_name)
        model_names.append(display_name)
        top1_accs.append(data['metrics']['top1_accuracy'] * 100)
        top5_accs.append(data['metrics']['top5_accuracy'] * 100)

    x = np.arange(len(model_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, FIGURE_HEIGHT))

    bars1 = ax.bar(x - bar_width/2, top1_accs, bar_width, label='Top-1 Accuracy', color='steelblue')
    bars2 = ax.bar(x + bar_width/2, top5_accs, bar_width, label='Top-5 Accuracy', color='lightsteelblue')

    ax.set_ylabel('accuracy (%)', fontsize=FONT_AXIS)
    ax.set_title('Model Accuracy Comparison', fontsize=FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_figure(save_path)


def plot_accuracy_vs_parameters(data_dict, model_display_names, model_colors, save_path=None):
    # scatter plot of accuracy vs model parameters

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for model_name in data_dict:
        data = data_dict[model_name]
        display_name = model_display_names.get(model_name, model_name)
        color = model_colors.get(model_name, '#333333')

        acc = data['metrics']['top1_accuracy'] * 100
        params_m = data['num_parameters'] / 1_000_000

        ax.scatter(params_m, acc, s=150, c=color, label=display_name, edgecolors='black')
        ax.annotate(display_name, (params_m, acc),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('parameters (millions)', fontsize=FONT_AXIS)
    ax.set_ylabel('top-1 accuracy (%)', fontsize=FONT_AXIS)
    ax.set_title('Accuracy vs Model Size', fontsize=FONT_TITLE)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(save_path)


def plot_confusion_matrix(labels, predictions, class_names, title='Confusion Matrix',
                         save_path=None, figsize=(14, 12), top_n_errors=10):
    # plot confusion matrix with annotations

    num_classes = len(class_names)
    cm = confusion_matrix(labels, predictions, labels=range(num_classes))

    # normalize by row (true labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)

    # create figure with two subplots
    fig = plt.figure(figsize=(figsize[0] + 6, figsize[1]))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.3)

    # left: full confusion matrix (normalized)
    ax1 = fig.add_subplot(gs[0])

    im = ax1.imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)

    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Count', fontsize=10)

    # set ticks - show every 5th class for readability
    tick_positions = []
    for i in range(0, num_classes, 5):
        tick_positions.append(i)

    tick_labels = []
    for i in tick_positions:
        name = class_names.get(i, str(i))
        tick_labels.append(name[:10])

    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax1.set_yticks(tick_positions)
    ax1.set_yticklabels(tick_labels, fontsize=7)

    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('True', fontsize=11)
    ax1.set_title(f'{title}\n(Normalized by True Class)', fontsize=12)

    # right: top misclassification pairs
    ax2 = fig.add_subplot(gs[1])

    # find top confusion pairs (excluding diagonal)
    error_pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                if cm[i, j] > 0:
                    true_name = class_names.get(i, str(i))
                    pred_name = class_names.get(j, str(j))
                    error_entry = {}
                    error_entry['true'] = true_name
                    error_entry['pred'] = pred_name
                    error_entry['count'] = cm[i, j]
                    error_entry['pct'] = cm_normalized[i, j] * 100
                    error_pairs.append(error_entry)

    # sort by count using bubble sort (no lambda)
    for i in range(len(error_pairs)):
        for j in range(i + 1, len(error_pairs)):
            if error_pairs[j]['count'] > error_pairs[i]['count']:
                temp = error_pairs[i]
                error_pairs[i] = error_pairs[j]
                error_pairs[j] = temp

    top_errors = error_pairs[:top_n_errors]

    if len(top_errors) > 0:
        error_labels = []
        error_counts = []
        for e in top_errors:
            true_short = truncate_name(e['true'], 10)
            pred_short = truncate_name(e['pred'], 10)
            error_labels.append(f"{true_short} -> {pred_short}")
            error_counts.append(e['count'])

        y_pos = np.arange(len(error_labels))
        bars = ax2.barh(y_pos, error_counts, color='salmon')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(error_labels, fontsize=8)
        ax2.invert_yaxis()
        ax2.set_xlabel('Count', fontsize=10)
        ax2.set_title(f'Top {top_n_errors} Confusions', fontsize=11)

        for idx in range(len(bars)):
            bar = bars[idx]
            count = error_counts[idx]
            ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', fontsize=8)

    plt.tight_layout()
    save_figure(save_path)

    return cm, top_errors


def create_summary_dataframe(all_data, all_histories, model_display_names):
    # create summary dataframe for all models

    summary_data = []

    for model_name in all_data:
        data = all_data[model_name]
        history = all_histories.get(model_name, {})

        display_name = model_display_names.get(model_name, model_name)
        params_m = data['num_parameters'] / 1_000_000

        # get best val acc
        val_acc = get_history_values(history, 'val_acc')
        if val_acc is not None:
            best_val_acc = max(val_acc)
        else:
            best_val_acc = data['metrics']['top1_accuracy'] * 100

        row = {}
        row['Model'] = display_name
        row['Params (M)'] = f'{params_m:.2f}'
        row['Top-1 (%)'] = f"{data['metrics']['top1_accuracy'] * 100:.2f}"
        row['Top-5 (%)'] = f"{data['metrics']['top5_accuracy'] * 100:.2f}"
        row['Precision'] = f"{data['metrics']['precision']:.4f}"
        row['Recall'] = f"{data['metrics']['recall']:.4f}"
        row['F1'] = f"{data['metrics']['f1_score']:.4f}"
        row['Best Val (%)'] = f'{best_val_acc:.2f}'

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    return summary_df


def save_summary_files(summary_df, comparison_dir):
    # save summary csv and markdown files

    csv_path = comparison_dir / 'summary.csv'
    summary_df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

    md_path = comparison_dir / 'summary.md'
    with open(md_path, 'w') as f:
        f.write('# Model Comparison Summary\n\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        f.write('## Test Set Results\n\n')
        f.write(summary_df.to_markdown(index=False))
        f.write('\n\n')

        # find best model
        best_idx = 0
        best_acc = 0.0
        for i in range(len(summary_df)):
            acc = float(summary_df.loc[i, 'Top-1 (%)'])
            if acc > best_acc:
                best_acc = acc
                best_idx = i

        best_model = summary_df.loc[best_idx, 'Model']
        best_acc_str = summary_df.loc[best_idx, 'Top-1 (%)']

        f.write('## Best Model\n\n')
        f.write(f'**{best_model}** achieved the highest top-1 accuracy of **{best_acc_str}%**\n')

    print(f"saved: {md_path}")


def save_generalization_gap_files(gap_stats_df, comparison_dir):
    # save generalization gap csv and markdown files

    # format for display
    gap_display_df = gap_stats_df.copy()
    formatted_values = []
    for val in gap_stats_df['Avg Gap (Last 10)']:
        formatted_values.append(f'{val:.2f}%')
    gap_display_df['Avg Gap (Last 10)'] = formatted_values

    # save csv
    csv_path = comparison_dir / 'generalization_gap_stats.csv'
    gap_stats_df.to_csv(csv_path, index=False)
    print(f"saved: {csv_path}")

    # save markdown
    md_path = comparison_dir / 'generalization_gap_stats.md'
    with open(md_path, 'w') as f:
        f.write('# Generalization Gap Analysis\n\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n\n')
        f.write('## Summary Statistics\n\n')
        f.write('The generalization gap measures overfitting as (Train Accuracy - Validation Accuracy).\n')
        f.write('Lower values indicate better generalization.\n\n')
        f.write(gap_display_df.to_markdown(index=False))
        f.write('\n\n')
        f.write('## Interpretation\n\n')
        f.write('- **Avg Gap (Last 10)**: Average gap in final training phase (last 10 epochs)\n')

    print(f"saved: {md_path}")

    return gap_display_df


def sample_indices_by_mask(mask, indices, n_samples, seed):
    # sample indices where mask is true
    # returns (sampled_data_indices, sampled_result_positions)

    matching_data_indices = []
    matching_result_positions = []

    for i in range(len(mask)):
        if mask[i]:
            matching_data_indices.append(indices[i])
            matching_result_positions.append(i)

    np.random.seed(seed)
    n_available = len(matching_data_indices)

    if n_available >= n_samples:
        sample_positions = np.random.choice(n_available, n_samples, replace=False)
    else:
        sample_positions = np.arange(n_available)

    sampled_data_indices = []
    sampled_result_positions = []
    for pos in sample_positions:
        sampled_data_indices.append(matching_data_indices[pos])
        sampled_result_positions.append(matching_result_positions[pos])

    return sampled_data_indices, sampled_result_positions


def save_classification_report(labels, predictions, model_name, head_type, results_dir):
    # save classification report to text file

    report = classification_report(labels, predictions, output_dict=False, zero_division=0)

    if head_type == 'model':
        report_filename = 'classification_report_model.txt'
        header = f"classification report (model head): {model_name}\n"
    else:
        report_filename = 'classification_report.txt'
        header = f"classification report (make head): {model_name}\n"

    report_path = Path(results_dir) / model_name / report_filename

    with open(report_path, 'w') as f:
        f.write(header)
        f.write(report)

    print(f"saved: {report_path}")

    return report_path


def plot_sampled_predictions(data, mask, image_paths, make_class_names,
                              model_class_names, title, save_path, seed, head_type):
    # sample predictions and generate visualization
    # head_type: 'make' or 'model'

    indices = data['indices']
    sampled_indices, sampled_result_idx = sample_indices_by_mask(
        mask, indices, 8, seed
    )

    is_hierarchical = data['is_hierarchical']

    # determine which data to use based on head type
    if head_type == 'model':
        probs = data.get('model_probs', None)
        preds = data['model_predictions']
        labels_arr = data['model_labels']
        class_names = model_class_names
        label_truncate = 12
    else:
        probs = data.get('probs', None)
        preds = data['predictions']
        labels_arr = data['labels']
        class_names = make_class_names
        label_truncate = 10

    has_probs = probs is not None

    # build lists for plot_sample_grid
    images = []
    titles = []
    title_colors = []
    bar_data = []
    bar_labels = []
    bar_colors = []

    for i in range(len(sampled_indices)):
        idx = sampled_indices[i]
        result_idx = sampled_result_idx[i]

        # load image
        img_path = get_original_image_path(image_paths[idx])
        img = Image.open(img_path).convert('RGB')
        images.append(img)

        # get labels
        true_label = labels_arr[result_idx]
        pred_label = preds[result_idx]
        true_name = class_names.get(true_label, str(true_label))

        is_correct = pred_label == true_label

        # determine title color
        if head_type == 'make' and is_hierarchical:
            # hierarchical make head - check both make and model correctness
            make_correct = data['make_predictions'][result_idx] == data['make_labels'][result_idx]
            model_correct = data['model_predictions'][result_idx] == data['model_labels'][result_idx]
            if make_correct and model_correct:
                color = 'green'
            elif make_correct or model_correct:
                color = 'orange'
            else:
                color = 'red'
        else:
            if is_correct:
                color = 'green'
            else:
                color = 'red'

        title_colors.append(color)

        # format title with true class
        true_name_short = truncate_name(true_name, label_truncate + 5)
        titles.append(f'True: {true_name_short}')

        # build bar data if probs available
        if has_probs:
            sample_probs = probs[result_idx]
            top5_indices = np.argsort(sample_probs)[-5:][::-1]
            top5_probs = sample_probs[top5_indices] * 100

            sample_labels = []
            sample_colors = []
            for j in range(len(top5_indices)):
                class_idx = top5_indices[j]
                name = class_names.get(class_idx, str(class_idx))
                name = truncate_name(name, label_truncate)
                sample_labels.append(name)

                if class_idx == true_label:
                    sample_colors.append(COLOR_CORRECT)
                else:
                    sample_colors.append(COLOR_INCORRECT)

            bar_data.append(top5_probs.tolist())
            bar_labels.append(sample_labels)
            bar_colors.append(sample_colors)

    # call plot_sample_grid
    if has_probs:
        plot_sample_grid(
            images, titles, title_colors,
            bar_data=bar_data, bar_labels=bar_labels, bar_colors=bar_colors,
            main_title=title, save_path=save_path
        )
    else:
        plot_sample_grid(
            images, titles, title_colors,
            main_title=title, save_path=save_path
        )


def generate_model_visualizations(model_name, data, image_paths, make_idx_to_name,
                                   model_idx_to_name, model_display_names, results_dir, seed):
    # generate all visualizations for a single model

    print(f"generating visualizations: {model_name}")

    # create figures subdirectory
    figures_dir = Path(results_dir) / model_name / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    predictions = data['predictions']
    labels = data['labels']
    is_hierarchical = data['is_hierarchical']
    display_name = model_display_names.get(model_name, model_name)

    # 1. per-class accuracy (make head)
    print("\n1. per-class accuracy (make head)")
    plot_per_class_accuracy(
        labels, predictions, make_idx_to_name,
        save_path=figures_dir / 'per_class_accuracy_make.png',
        title=f'{display_name} - Per-Class Accuracy (Make)'
    )

    # 1b-1c. per-class accuracy and classification report (model head)
    if is_hierarchical:
        print("\n1b. per-class accuracy (model head)")
        plot_per_class_accuracy(
            data['model_labels'], data['model_predictions'], model_idx_to_name,
            save_path=figures_dir / 'per_class_accuracy_model.png',
            top_n=20,
            title=f'{display_name} - Per-Class Accuracy (Model)'
        )

        print("\n1c. saving classification report (model head)")
        save_classification_report(
            data['model_labels'], data['model_predictions'],
            model_name, 'model', results_dir
        )

    # 2. sample correct predictions (make head)
    print("\n2. sample correct predictions")
    correct_mask = predictions == labels
    correct_title = f'{display_name} - Correct Predictions'
    if is_hierarchical:
        correct_title = f'{display_name} - Correct Predictions (Make)'
    plot_sampled_predictions(
        data, correct_mask, image_paths,
        make_idx_to_name, model_idx_to_name,
        correct_title, figures_dir / 'sample_correct.png',
        seed, 'make'
    )

    # 3. sample incorrect predictions (make head)
    print("\n3. sample incorrect predictions")
    incorrect_mask = predictions != labels
    incorrect_title = f'{display_name} - Misclassifications'
    if is_hierarchical:
        incorrect_title = f'{display_name} - Misclassifications (Make)'
    plot_sampled_predictions(
        data, incorrect_mask, image_paths,
        make_idx_to_name, model_idx_to_name,
        incorrect_title, figures_dir / 'sample_incorrect.png',
        seed, 'make'
    )

    # 3b-3c. hierarchical model head samples
    if is_hierarchical:
        print("\n3b. sample correct predictions (model head)")
        model_correct_mask = data['model_predictions'] == data['model_labels']
        plot_sampled_predictions(
            data, model_correct_mask, image_paths,
            make_idx_to_name, model_idx_to_name,
            f'{display_name} - Correct Predictions (Model)',
            figures_dir / 'sample_correct_model.png',
            seed + 1, 'model'
        )

        print("\n3c. sample incorrect predictions (model head)")
        model_incorrect_mask = data['model_predictions'] != data['model_labels']
        plot_sampled_predictions(
            data, model_incorrect_mask, image_paths,
            make_idx_to_name, model_idx_to_name,
            f'{display_name} - Misclassifications (Model)',
            figures_dir / 'sample_incorrect_model.png',
            seed + 1, 'model'
        )

    # 4. classification report (make head)
    print("\n4. saving classification report (make head)")
    save_classification_report(labels, predictions, model_name, 'make', results_dir)

    return figures_dir
