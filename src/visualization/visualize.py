import tensorflow as tf
import tensorflow_data_validation as tfdv
import os
import click

@click.argument('dataset', type=click.Path())
@click.argument('columns', type=click.Path())
def visualize_statistics(dataset, columns=None):
    if columns == None:
        columns =  [col for col in dataset.columns]
    stats_options = tfdv.StatsOptions(feature_whitelist=columns)
    stats = tfdv.generate_statistics_from_dataframe(dataset, stats_options)
    tfdv.visualize_statistics(stats)
    return stats



