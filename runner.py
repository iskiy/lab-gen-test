from multiprocessing import Pool, Queue
from multiprocessing import Manager
from queue import Empty  # Import Empty exception

import gc
import numpy as np
from config import NR, THREADS
from stats.experiment_stats import ExperimentStats
from evo_algorithm import EvoAlgorithm
from model.population import Population
from selection.selection_method import SelectionMethod
from model.gen_operators import GeneticOperator
from copy import deepcopy
from datetime import datetime
from output.plotting import StatPlotter


# def run_experiment(selection_method: SelectionMethod,
#                    genetic_operator: GeneticOperator,
#                    populations: list[Population],
#                    param_names: tuple[str]):
#     stats = ExperimentStats(param_names)
#
#     # Prepare the parameters for each run based on the number of runs (NR)
#     run_param_list = [
#         (populations[run_i],
#          selection_method,
#          genetic_operator,
#          param_names,
#          run_i
#          )
#         for run_i in range(NR)
#     ]
#
#     # Execute each run sequentially
#     for params in run_param_list:
#         run_i, run_stats = run(*params)  # Assuming `run` function returns a tuple (run_index, run_stats)
#         stats.add_run(run_stats, run_i)
#
#     stats.calculate()
#     print(f'{str(datetime.now())[:-4]} | Experiment ({"|".join(param_names)}) finished')
#
#     # Since we are not using multiprocessing, the manual call to garbage collection might be less necessary,
#     # but keeping it won't harm in case of large objects being processed
#     gc.collect()
#     return stats
# def run_experiment(selection_method: SelectionMethod,
#                    genetic_operator: GeneticOperator,
#                    populations: list[Population],
#                    param_names: tuple[str], queue):
#     stats = ExperimentStats(param_names)
#
#     run_param_list = [
#         (populations[run_i],
#          selection_method,
#          genetic_operator,
#          param_names,
#          run_i,
#          queue
#          )
#         for run_i in range(NR)
#     ]
#
#     # Creating a single Pool outside the loop
#     with Pool(THREADS) as p:
#         # Splitting the total number of runs into batches based on THREADS
#         for i in range((NR + THREADS - 1) // THREADS):  # This ensures we cover all runs even if NR % THREADS != 0
#             batch_start = i * THREADS
#             batch_end = min((i + 1) * THREADS, NR)  # Avoid going beyond the total number of runs
#             batch_results = p.starmap(run, run_param_list[batch_start:batch_end])
#
#             # Processing results from the current batch
#             for run_i, run_stats in batch_results:
#                 stats.add_run(run_stats, run_i)
#
#     # for i in range(NR // THREADS):
#     #     with Pool(THREADS) as p:
#     #         results = p.starmap(run, run_param_list[(i * THREADS):((i + 1) * THREADS)])
#     #         for run_i, run_stats in results:
#     #             stats.add_run(run_stats, run_i)
#     # if NR % THREADS:
#     #     with Pool(NR % THREADS) as p:
#     #         results = p.starmap(run, run_param_list[-(NR % THREADS):])
#     #         for run_i, run_stats in results:
#     #             stats.add_run(run_stats, run_i)
#
#     stats.calculate()
#     print(f'{str(datetime.now())[:-4]} | Experiment ({"|".join(param_names)}) finished')
#     gc.collect()
#     return stats
#
#
# def run(init_population: Population,
#         selection_method: SelectionMethod,
#         genetic_operator: GeneticOperator,
#         param_names: tuple[str],
#         run_i: int, plotter_queue):
#     plotter = plotter_queue.get()
#     current_run = EvoAlgorithm(deepcopy(init_population), selection_method, genetic_operator, param_names, plotter).run(
#         run_i)
#     plotter_queue.put(plotter)
#     return run_i, current_run

def run_experiment(selection_method: SelectionMethod,
                   genetic_operator: GeneticOperator,
                   populations: list[Population],
                   param_names: tuple[str],
                   # queue
                   ):
    stats = ExperimentStats(param_names)

    run_param_list = [
        (populations[run_i],
         selection_method,
         genetic_operator,
         param_names,
         run_i,
         # queue
         )
        for run_i in range(NR)
    ]

    # Creating a single Pool outside the loop
    with Pool(THREADS) as p:
        # Splitting the total number of runs into batches based on THREADS
        for i in range((NR + THREADS - 1) // THREADS):  # This ensures we cover all runs even if NR % THREADS != 0
            batch_start = i * THREADS
            batch_end = min((i + 1) * THREADS, NR)  # Avoid going beyond the total number of runs
            batch_results = p.starmap(run, run_param_list[batch_start:batch_end])

            # Processing results from the current batch
            for run_i, run_stats in batch_results:
                stats.add_run(run_stats, run_i)

    stats.calculate()

    print(f'{str(datetime.now())[:-4]} | Experiment ({"|".join(param_names[:4])}|{str(param_names[4])}) finished')

    # print(f'{str(datetime.now())[:-4]} | Experiment ({"|".join(param_names)}) finished')
    gc.collect()
    return stats


def run(init_population: Population,
        selection_method: SelectionMethod,
        genetic_operator: GeneticOperator,
        param_names: tuple[str],
        run_i: int,
        # plotter_queue
        ):
    # plotter = plotter_queue.get()
    plotter = StatPlotter()
    current_run = EvoAlgorithm(deepcopy(init_population), selection_method, genetic_operator, param_names, plotter).run(
        run_i)
    plotter.Close()
    # plotter_queue.put(plotter)
    return run_i, current_run