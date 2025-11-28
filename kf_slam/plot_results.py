#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(baseline_file, msf_file, metric, ylabel, title, output):
    """绘制单项指标对比曲线"""
    baseline_df = pd.read_csv(baseline_file)
    msf_df = pd.read_csv(msf_file)

    plt.figure()
    plt.plot(baseline_df['step'], baseline_df[metric],
             '--', label='baseline')
    plt.plot(msf_df['step'], msf_df[metric],
             '-', label='msf_rrt')

    plt.grid(True)
    plt.legend()
    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(output)
    plt.close()


def draw_boxplot(baseline_files, msf_files):
    """绘制 msf vs baseline 所有距离的箱型图"""
    baseline_lengths = []
    msf_lengths = []

    # baseline
    for file in baseline_files:
        df = pd.read_csv(file)
        if 'distance' in df.columns:
            baseline_lengths.extend(df['distance'].values)

    # msf
    for file in msf_files:
        df = pd.read_csv(file)
        if 'distance' in df.columns:
            msf_lengths.extend(df['distance'].values)

    if baseline_lengths and msf_lengths:
        plt.figure()
        plt.boxplot([baseline_lengths, msf_lengths],
                    labels=['baseline', 'msf_rrt'])
        plt.title('Path Length Distribution (Box Plot)')
        plt.ylabel('Distance')
        plt.grid(True)
        plt.savefig('path_length_boxplot.png')
        plt.close()


def main():
    if len(sys.argv) < 3:
        print("Usage: python plot_results.py baseline.csv msf_rrt.csv")
        sys.exit(1)

    baseline_file = sys.argv[1]
    msf_file = sys.argv[2]

    # 三个主曲线：覆盖率/不确定性/距离
    plot_metric(baseline_file, msf_file,
                'coverage', 'Coverage', 'Coverage Comparison',
                'coverage_comparison.png')

    plot_metric(baseline_file, msf_file,
                'uncertainty', 'Uncertainty', 'Uncertainty Comparison',
                'uncertainty_comparison.png')

    plot_metric(baseline_file, msf_file,
                'distance', 'Distance', 'Distance Comparison',
                'distance_comparison.png')

    # 路径长度分布箱型图
    draw_boxplot([baseline_file], [msf_file])

    print("\n图像生成完成：")
    print("  - coverage_comparison.png")
    print("  - uncertainty_comparison.png")
    print("  - distance_comparison.png")
    print("  - path_length_boxplot.png\n")


if __name__ == '__main__':
    main()

