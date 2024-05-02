from config import env, NR
from model.fitness_functions import *
from output.plotting import StatPlotter
from selection.rws import *
from selection.tourn import *
from model.encoding import *
from model.gen_operators import *
from output import excel
from runner import run_experiment
from datetime import datetime
import time
from config import THREADS
from multiprocessing import Manager
from queue import Empty
from selection.sus import *

if env == 'test':
    fitness_functions = [
        (FconstALL(100), 'FconstALL'),
        # (FH(100), 'FH'),
        # (Fx2(BinaryEncoderUni(10, 0.0, 10.23, 0.01)), 'Fx2 Binary'),
        # (Fx2(BinaryEncoderUni(3, 1.0, 8.0, 1)), 'Fx2 Binary Try'),
        # (Fx2(GrayEncoderUni(10, 0.0, 10.23, 0.01)), 'Fx2 Gray'),
        # # (Fx2(FloatEncoder(0.0, 10.23, 10)), 'Fx2 Float'),
        # # (Fx2(FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fx2 Float Gray'),
        # (F5122subx2(BinaryEncoderUni(10, -5.12, 5.11, 0.01)), 'F5122subx2 Binary'),
        # (F5122subx2(GrayEncoderUni(10, -5.12, 5.11, 0.01)), 'F5122subx2 Gray'),
        # # (F5122subx2(FloatEncoder(-5.12, 5.11, 10)), 'F5122subx2 Float'),
        # # (F5122subx2(FloatEncoder(-5.12, 5.11, 10, is_gray=True)), 'F5122subx2 Float Gray'),
        # (Fexp(0.25, BinaryEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp0.25 Binary'),
        # (Fexp(0.25, GrayEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp0.25 Gray'),
        # # (Fexp(0.25, FloatEncoder(0.0, 10.23, 10)), 'Fexp0.25 Float'),
        # # (Fexp(0.25, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp0.25 Float Gray'),
        # (Fexp(1, BinaryEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp1 Binary'),
        # (Fexp(1, GrayEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp1 Gray'),
        # # (Fexp(1, FloatEncoder(0.0, 10.23, 10)), 'Fexp1 Float'),
        # # (Fexp(1, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp1 Float Gray'),
        # (Fexp(2, BinaryEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp2 Binary'),
        # (Fexp(2, GrayEncoderUni(10, 0.0, 10.23, 0.01)), 'Fexp2 Gray'),
        # # (Fexp(2, FloatEncoder(0.0, 10.23, 10)), 'Fexp2 Float'),
        # # (Fexp(2, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp2 Float Gray'),
        # (FRastrigina(7, BinaryEncoderUni(10, -5.12, 5.12, 0.01)), 'FRastrigina Binary'),
        # (FRastrigina(7, GrayEncoderUni(10, -5.12, 5.12, 0.01)), 'FRastrigina Gray'),
        # # (FRastrigina(7, FloatEncoder(-5.12, 5.12, 10)), 'FRastrigina Float'),
        # # (FRastrigina(7, FloatEncoder(-5.12, 5.12, 10, is_gray=True)), 'FRastrigina Float Gray'),
        # (FDeba2(BinaryEncoderUni(10, 0, 1.023, 0.001)), 'FDeba2 Binary'),
        # (FDeba2(GrayEncoderUni(10, 0, 1.023, 0.001)), 'FDeba2 Gray'),
        # # (FDeba2(FloatEncoder(0, 1.023, 10)), 'FDeba2 Float'),
        # # (FDeba2(FloatEncoder(0, 1.023, 10, is_gray=True)), 'FDeba2 Float Gray'),
        # (FDeba4(BinaryEncoderUni(10, 0, 1.023, 0.001)), 'FDeba4 Binary'),
        # (FDeba4(GrayEncoderUni(10, 0, 1.023, 0.001)), 'FDeba4 Gray'),
        # # (FDeba4(FloatEncoder(0, 1.023, 10)), 'FDeba2 Float'),
        # # (FDeba4(FloatEncoder(0, 1.023, 10, is_gray=True)), 'FDeba2 Float Gray')
    ]
    selection_methods = [
        # (RWS(), 'RWS',  {'beta': '1.6'}),
        # (SUS(), 'SUS',  {'beta': '1.6'}),
        (LinearRankingSUS(1.6), 'LinearRankingSUS', {'beta': '1.6', 'p': '1'}),
        # (LinearRankingSUS(1.2), 'LinearRankingSUS', {'beta': '1.2','p': '1'}),
        # (LinearRankingSUS(1.4), 'LinearRankingSUS', {'beta': '1.4', 'p': '1'}),
        # (LinearRankingSUS(2), 'LinearRankingSUS', {'beta': '2', 'p': '1'}),
        # (LinearModifiedRankingSUS(1.6), 'LinearModifiedRankingSUS', {'beta': '1.6'}),
        # (LinearModifiedRankingSUS(1.4), 'LinearModifiedRankingSUS', {'beta': '1.4'}),
        (LinearRankingRWS(1.6), 'LinearRankingRWS', {'beta': '1.6', 'p': '1'}),
        # (LinearRankingRWS(2), 'LinearRankingRWS', {'beta': '2.0'}),
        # (LinearModifiedRankingRWS(1.6), 'LinearModifiedRankingRWS', {'beta': '1.6'}),
        # (LinearModifiedRankingRWS(2), 'LinearModifiedRankingRWS', {'beta': '2.0'}),

        # (TS(2, p10, False), 'TournamentWithoutReplacement', {'t': '2', 'p': '1'}),
        # (TS(4, p10, False), 'TournamentWithoutReplacement', {'t': '4', 'p': '1'}),
        # (TS(8, p10, False), 'TournamentWithoutReplacement', {'t': '8', 'p': '1'}),
        # (TS(10, p10, False), 'TournamentWithoutReplacement', {'t': '10', 'p': '1'}),
        # (TS(2, p08, False), 'TournamentProbWithoutReplacement', {'t': '2', 'p': '0.8'}),
        # (TS(2, p075, False), 'TournamentProbWithoutReplacement', {'t': '2', 'p': '0.75'}),
        # (TS(2, p07, False), 'TournamentProbWithoutReplacement', {'t': '2', 'p': '0.7'}),
        # (TS(2, pxi, False), 'TournamentProbWithoutReplacement', {'t': '2', 'p': 'dynamic'})
    ]
    gen_operators = [
        (BlankGenOperator, 'no_operators'),
        # (Crossover, 'crossover'),
        # (Mutation, 'mutation'),
        # (CrossoverAndMutation, 'xover_mut')
    ]
else:
    fitness_functions = [
        # (FconstALL(100), 'FconstALL'),
        # (FHD(100, 100), 'FHD'),
        # (Fx2(FloatEncoder(0.0, 10.23, 10)), 'Fx2'),
        # (Fx2(FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fx2_gray'),
        # (F5122subx2(FloatEncoder(-5.12, 5.11, 10)), 'F5122subx2'),
        # (F5122subx2(FloatEncoder(-5.12, 5.11, 10, is_gray=True)), 'F5122subx2_gray'),
        # (Fexp(0.25, FloatEncoder(0.0, 10.23, 10)), 'Fexp0.25'),
        # (Fexp(0.25, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp0.25_gray'),
        # (Fexp(1, FloatEncoder(0.0, 10.23, 10)), 'Fexp1'),
        # (Fexp(1, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp1_gray'),
        # (Fexp(2, FloatEncoder(0.0, 10.23, 10)), 'Fexp2'),
        # (Fexp(2, FloatEncoder(0.0, 10.23, 10, is_gray=True)), 'Fexp2_gray')
    ]
    selection_methods = [
        # (RWS, 'RWS'),
        # (DisruptiveRWS, 'RWS_disruptive'),
        # (BlendedRWS, 'RWS_blended'),
        # (WindowRWS, 'RWS_window'),
        # (SUS, 'SUS'),
        # (DisruptiveSUS, 'SUS_disruptive'),
        # (BlendedSUS, 'SUS_blended'),
        # (WindowSUS, 'SUS_window')
    ]
    gen_operators = [
        (BlankGenOperator, 'no_operators'),
        # (Crossover, 'crossover'),
        # (Mutation, 'mutation'),
        # (CrossoverAndMutation, 'xover_mut')
    ]


def init_population(fitness_function, ff_name, gen_operator):
    if ff_name != "FconstALL":
        if gen_operator == 'no_operators':
            return [
                ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_per=1, with_operator=False)
                  for run_i in range(NR)], "InitExact1OptimalNoOperator"),
                ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_percent=0.05,
                             with_operator=False)
                  for run_i in range(NR)], "Init5%OptimalNoOperator"),
                ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_percent=0.10,
                             with_operator=False)
                  for run_i in range(NR)], "Init10%OptimalNoOperator")
            ]
        else:
            return [
                ([Population(fitness_function, seed=get_pop_seed(run_i), with_operator=True)
                  for run_i in range(NR)], "InitNoOptimalWithOperators"),
                # ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_per=1, with_operator=True)
                #   for run_i in range(NR)], "InitExact1OptimalWithOperators"),
                # ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_percent=0.05, with_operator=True)
                #   for run_i in range(NR)], "Init5%OptimalWithOperators"),
                # ([Population(fitness_function, seed=get_pop_seed(run_i), optimal_initialise_percent=0.10,
                #              with_operator=True)
                #   for run_i in range(NR)], "Init10%OptimalWithOperators"),
            ]
    else:
        return [([Population(fitness_function, seed=get_pop_seed(run_i), with_operator=False, can_have_optimal=True)
                  for run_i in range(NR)], "Binomial")]


# a list of tuples of parameters for each run that involves a certain fitness function 
# {fitness_func_name: [(tuples with run parameters), (), ..., ()], other_func: [], ...}
experiment_params = {
    ff: [
        (sm, go, ip, (ff_name, sm_name, go_name, ip_name, selection_params))
        for (sm, sm_name, selection_params) in selection_methods
        for (go, go_name) in gen_operators
        for (ip, ip_name) in init_population(ff, ff_name, go_name)
    ] for (ff, ff_name) in fitness_functions
}


# only keeping one list of populations in memory at a time (for one fitness function)
def generate_all_populations_for_fitness_function(ff):
    return [ff.generate_population_for_run(run_i) for run_i in range(NR)]


def log(x):
    datetime_prefix = str(datetime.now())[:-4]
    print(f'{datetime_prefix} | {x}')


def close_all_plotters(q):
    try:
        while True:
            plotter = q.get_nowait()
            plotter.Close()
    except Empty:
        pass


# if __name__ == '__main__':
#     log('Program start')
#     print('----------------------------------------------------------------------')
#     start_time = time.time()
#     results = []
#
#     with Manager() as manager:
#         plotter_queue = manager.Queue()  # Create a managed queue
#         for _ in range(THREADS):
#             plotter_queue.put(StatPlotter())
#
#         for ff in experiment_params:
#             ff_start_time = time.time()
#             # populations = generate_all_populations_for_fitness_function(ff)
#             params = [params for params in experiment_params[ff]]
#             experiment_stats_list = [run_experiment(*p, plotter_queue) for p in params]
#
#             excel.write_ff_stats(experiment_stats_list)
#             for experiment_stats in experiment_stats_list:
#                 del experiment_stats.runs
#                 results.append(experiment_stats)
#
#             ff_end_time = time.time()
#             ff_name = experiment_params[ff][0][3][0]
#             log(f'{ff_name} experiments finished in {(ff_end_time - ff_start_time):.2f}s')
#
#         excel.write_aggregated_stats(results)
#
#         print('----------------------------------------------------------------------')
#         end_time = time.time()
#         log(f'Program end. Total runtime: {end_time - start_time:.2f}s')
#         close_all_plotters(plotter_queue)

if __name__ == '__main__':
    log('Program start')
    print('----------------------------------------------------------------------')
    start_time = time.time()
    results = []

    # with Manager() as manager:
    #     plotter_queue = manager.Queue()  # Create a managed queue
    # for _ in range(THREADS):
    #     plotter_queue.put(StatPlotter())

    for ff in experiment_params:
        ff_start_time = time.time()
        # populations = generate_all_populations_for_fitness_function(ff)
        params = [params for params in experiment_params[ff]]
        # experiment_stats_list = [run_experiment(*p, plotter_queue) for p in params]
        experiment_stats_list = [run_experiment(*p) for p in params]

        excel.write_ff_stats(experiment_stats_list)
        excel.write_ff_aggregated_stats(experiment_stats_list)
        for experiment_stats in experiment_stats_list:
            del experiment_stats.runs
            results.append(experiment_stats)

        ff_end_time = time.time()
        ff_name = experiment_params[ff][0][3][0]
        log(f'{ff_name} experiments finished in {(ff_end_time - ff_start_time):.2f}s')

    excel.write_aggregated_stats(results)

    print('----------------------------------------------------------------------')
    end_time = time.time()
    log(f'Program end. Total runtime: {end_time - start_time:.2f}s')
        # close_all_plotters(plotter_queue)