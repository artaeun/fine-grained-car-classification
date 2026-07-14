from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.dataset import (
    CompCarsDataset,
    get_val_transforms
)

# helper functions for data exploration script for compcars dataset
# analyzes dataset statistics and saves results

def get_count_value(item):
    # extract count value from (class_id, count) tuple for sorting
    count = item[1]
    return count


def analyze_split_file(split_file):
    # analyze a train/test split file

    with open(split_file, 'r') as f:
        lines = f.readlines()
    makes = []
    models = []
    years = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        parts = line.split('/')
        make_id = int(parts[0])
        model_id = int(parts[1])
        year_str = parts[2]

        makes.append(make_id)
        models.append((make_id, model_id))

        # handle unknown years - use 0 as placeholder to filter later
        if year_str == 'unknown':
            year = 0
        else:
            year = int(year_str)
        years.append(year)

    # compute statistics
    make_counts = Counter(makes)
    model_counts = Counter(models)
    year_counts = Counter(years)

    stats = {}
    stats['total_images'] = len(lines)
    stats['unique_makes'] = len(make_counts)
    stats['unique_models'] = len(model_counts)
    stats['unique_years'] = len(year_counts)
    stats['make_counts'] = make_counts
    stats['model_counts'] = model_counts
    stats['year_counts'] = year_counts

    # compute samples per class stats
    make_samples = list(make_counts.values())
    stats['make_min_samples'] = min(make_samples)
    stats['make_max_samples'] = max(make_samples)
    stats['make_mean_samples'] = np.mean(make_samples)
    stats['make_std_samples'] = np.std(make_samples)
    model_samples = list(model_counts.values())
    stats['model_min_samples'] = min(model_samples)
    stats['model_max_samples'] = max(model_samples)
    stats['model_mean_samples'] = np.mean(model_samples)
    stats['model_std_samples'] = np.std(model_samples)

    return stats


def plot_class_distribution(counts, title, output_path, top_n=30):
    # plot class distribution histogram
    values = list(counts.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # histogram of samples per class
    axes[0].hist(values, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('samples per class')
    axes[0].set_ylabel('number of classes')
    axes[0].set_title(f'{title} - distribution')
    axes[0].axvline(
        np.mean(values),
        color='red',
        linestyle='--',
        label=f'mean: {np.mean(values):.1f}'
    )
    axes[0].legend()

    # top n classes bar chart
    sorted_counts = sorted(counts.items(), key=get_count_value, reverse=True)
    top_classes = sorted_counts[:top_n]
    class_labels = []
    class_values = []
    for class_id, count in top_classes:
        if isinstance(class_id, tuple):
            label = f"{class_id[0]}/{class_id[1]}"
        else:
            label = str(class_id)
        class_labels.append(label)
        class_values.append(count)

    axes[1].barh(range(len(class_values)), class_values, color='steelblue')
    axes[1].set_yticks(range(len(class_labels)))
    axes[1].set_yticklabels(class_labels, fontsize=8)
    axes[1].set_xlabel('number of samples')
    axes[1].set_title(f'{title} - top {top_n} classes')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"saved: {output_path}")


def plot_year_distribution(year_counts, output_path):
    # plot year distribution ignoring unknown years p(stored as 0)
    filtered_counts = {}
    for year, count in year_counts.items():
        if year != 0:
            filtered_counts[year] = count
    sorted_years = sorted(filtered_counts.items())
    years = []
    counts = []
    for year, count in sorted_years:
        years.append(year)
        counts.append(count)
    min_year = min(years)
    max_year = max(years)

    plt.figure(figsize=(12, 5))
    plt.bar(years, counts, color='steelblue', edgecolor='black')
    plt.xlabel('year')
    plt.ylabel('number of images')
    plt.title(f'images per year ({min_year}-{max_year})')
    plt.xlim(min_year - 0.5, max_year + 0.5)
    plt.xticks(range(min_year, max_year + 1), rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"saved: {output_path}")


def save_statistics_report(train_stats, test_stats, output_path):
    # save statistics to markdown file so i can put it in the NN obsidian vault
    # and use it later when writing the report
    with open(output_path, 'w') as f:
        f.write("# CompCars Dataset Statistics\n\n")
        f.write(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Overview\n\n")
        f.write("| Split | Images | Makes | Models | Years |\n")
        f.write("|-------|--------|-------|--------|-------|\n")
        f.write(f"| Train | {train_stats['total_images']} | ")
        f.write(f"{train_stats['unique_makes']} | ")
        f.write(f"{train_stats['unique_models']} | ")
        f.write(f"{train_stats['unique_years']} |\n")
        f.write(f"| Test | {test_stats['total_images']} | ")
        f.write(f"{test_stats['unique_makes']} | ")
        f.write(f"{test_stats['unique_models']} | ")
        f.write(f"{test_stats['unique_years']} |\n")
        f.write(f"| Total | {train_stats['total_images'] + test_stats['total_images']} | ")
        f.write(f"- | - | - |\n\n")

        f.write("## Make Classification (Coarse-grained)\n\n")
        f.write("| Metric | Train | Test |\n")
        f.write("|--------|-------|------|\n")
        f.write(f"| Number of classes | {train_stats['unique_makes']} | ")
        f.write(f"{test_stats['unique_makes']} |\n")
        f.write(f"| Min samples/class | {train_stats['make_min_samples']} | ")
        f.write(f"{test_stats['make_min_samples']} |\n")
        f.write(f"| Max samples/class | {train_stats['make_max_samples']} | ")
        f.write(f"{test_stats['make_max_samples']} |\n")
        f.write(f"| Mean samples/class | {train_stats['make_mean_samples']:.1f} | ")
        f.write(f"{test_stats['make_mean_samples']:.1f} |\n")
        f.write(f"| Std samples/class | {train_stats['make_std_samples']:.1f} | ")
        f.write(f"{test_stats['make_std_samples']:.1f} |\n\n")

        f.write("## Model Classification (Fine-grained)\n\n")
        f.write("| Metric | Train | Test |\n")
        f.write("|--------|-------|------|\n")
        f.write(f"| Number of classes | {train_stats['unique_models']} | ")
        f.write(f"{test_stats['unique_models']} |\n")
        f.write(f"| Min samples/class | {train_stats['model_min_samples']} | ")
        f.write(f"{test_stats['model_min_samples']} |\n")
        f.write(f"| Max samples/class | {train_stats['model_max_samples']} | ")
        f.write(f"{test_stats['model_max_samples']} |\n")
        f.write(f"| Mean samples/class | {train_stats['model_mean_samples']:.1f} | ")
        f.write(f"{test_stats['model_mean_samples']:.1f} |\n")
        f.write(f"| Std samples/class | {train_stats['model_std_samples']:.1f} | ")
        f.write(f"{test_stats['model_std_samples']:.1f} |\n\n")

        f.write("## Class Imbalance Analysis\n\n")
        train_imbalance = train_stats['make_max_samples'] / train_stats['make_min_samples']
        f.write(f"- make imbalance ratio (max/min): {train_imbalance:.1f}x\n")
        model_imbalance = train_stats['model_max_samples'] / train_stats['model_min_samples']
        f.write(f"- model imbalance ratio (max/min): {model_imbalance:.1f}x\n")
        f.write("\n")

        f.write("## Figures\n\n")
        f.write("- `make_distribution.png` - samples per make class\n")
        f.write("- `model_distribution.png` - samples per model class\n")
        f.write("- `year_distribution.png` - images per year\n")

    print(f"saved: {output_path}")


def test_dataset_loading(root_dir, split_file):
    # test that dataset loading works correctly
    print("\ntesting dataset loading...")
    transform = get_val_transforms()
    dataset = CompCarsDataset(
        root_dir=root_dir,
        split_file=split_file,
        transform=transform,
        task='make'
    )
    print(f"  dataset size: {len(dataset)}")
    print(f"  num makes: {dataset.num_makes}")
    print(f"  num models: {dataset.num_models}")

    # load first sample
    image, label = dataset[0]
    print(f"  sample image shape: {image.shape}")
    print(f"  sample label: {label}")

    # verify image dimensions and channels
    is_valid = True
    if image.shape[0] != 3:
        print("  error: expected 3 channels")
        is_valid = False
    if image.shape[1] != 224:
        print("  error: expected height 224")
        is_valid = False
    if image.shape[2] != 224:
        print("  error: expected width 224")
        is_valid = False
    if is_valid:
        print("  dataset loading test passed!")
    return is_valid


def main():
    # paths
    data_dir = Path("dataset/data")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    train_split = data_dir / "train_test_split/classification/train.txt"
    test_split = data_dir / "train_test_split/classification/test.txt"

    print("analyzing compcars dataset...")
    print(f"train split: {train_split}")
    print(f"test split: {test_split}")

    # analyze splits

    #train
    print("\nanalyzing train split...")
    train_stats = analyze_split_file(str(train_split))
    print(f"  images: {train_stats['total_images']}")
    print(f"  makes: {train_stats['unique_makes']}")
    print(f"  models: {train_stats['unique_models']}")

    #test
    print("\nanalyzing test split...")
    test_stats = analyze_split_file(str(test_split))
    print(f"  images: {test_stats['total_images']}")
    print(f"  makes: {test_stats['unique_makes']}")
    print(f"  models: {test_stats['unique_models']}")

    # plot distributions
    print("\ngenerating plots...")

    plot_class_distribution(
        train_stats['make_counts'],
        "make classes (train)",
        str(results_dir / "make_distribution.png")
    )

    plot_class_distribution(
        train_stats['model_counts'],
        "model classes (train)",
        str(results_dir / "model_distribution.png"),
        top_n=30
    )

    plot_year_distribution(
        train_stats['year_counts'],
        str(results_dir / "year_distribution.png")
    )

    # save statistics report
    save_statistics_report(
        train_stats,
        test_stats,
        str(results_dir / "dataset_statistics.md")
    )

    # test dataset loading
    test_dataset_loading(str(data_dir), str(train_split))
    print("\ndata exploration complete!")
    print(f"results saved to: {results_dir}")


if __name__ == '__main__':
    main()
