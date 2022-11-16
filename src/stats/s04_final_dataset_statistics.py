# TODO - statistics of the final dataset
import argparse
import json
import logging
import os

from stats.utils.s04_final_dataset_statistics_utils import print_some_examples_in_md, stat_target_anchor_lengths, \
    print_latex_subset_distribution_tuples_v3, stat_mentions_per_entity, stat_mentions_per_entity_across_cuts, \
    get_tuples_per_year
from utils import tempel_logger

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S', level=tempel_logger.logger_level)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_file', required=False, type=str,
                        # TODO
                        default='TODO',
                        help='The config file that contains all the parameters')

    args = parser.parse_args()
    logger.info('getting the final dataset with the following parameters: %s' % args)
    config = json.load(open(args.config_file, 'rt'))
    stats_to_run = set(config['stats_to_run'])

    output_plot_path = config['output_plot_path']
    os.makedirs(output_plot_path, exist_ok=True)
    if 'examples_data' in stats_to_run:
        print_some_examples_in_md(config)

    # get the tuples and serializes them!

    train_tuples_per_year, validation_tuples_per_year, test_tuples_per_year = get_tuples_per_year(config)
    #
    if 'stat_target_anchor_lengths' in stats_to_run:
        logger.info('============= stat_target_anchor_lengths')
        stat_target_anchor_lengths(config,
                                   train_tuples_per_year=train_tuples_per_year,
                                   validation_tuples_per_year=validation_tuples_per_year,
                                   test_tuples_per_year=test_tuples_per_year)

    if 'distribution_subsets_latex' in stats_to_run:
        logger.info('============= and now print_latex_subset_distribution_v2')
        print_latex_subset_distribution_tuples_v3(config,
                                                  train_tuples=train_tuples_per_year,
                                                  validation_tuples=validation_tuples_per_year,
                                                  test_tuples=test_tuples_per_year)

    if 'stat_mentions_per_entity' in stats_to_run:
        logger.info('============= stat_mentions_per_entity')
        stat_mentions_per_entity(config,
                                 train_tuples_per_year=train_tuples_per_year,
                                 validation_tuples_per_year=validation_tuples_per_year,
                                 test_tuples_per_year=test_tuples_per_year
                                 )

    if 'stat_mentions_per_entity_across_cuts' in stats_to_run:
        logger.info('============= stat_mentions_per_entity')
        stat_mentions_per_entity_across_cuts(config,
                                             train_tuples_per_year=train_tuples_per_year,
                                             validation_tuples_per_year=validation_tuples_per_year,
                                             test_tuples_per_year=test_tuples_per_year)
