from config import (N, NR, OUTPUT_FOLDER, RUN_STATS_NAMES, EXP_STATS_NAMES, FCONSTALL_RUN_STATS_NAMES,
                    FCONSTALL_EXP_STATS_NAMES, GEN_STATS_NAMES, FCONSTALL_GEN_STATS_NAMES)
import xlsxwriter
import os
from stats.experiment_stats import ExperimentStats
from stats.generation_stats import GenerationStats
from model.population import Population


def write_population(population: Population, param_names: tuple[str], run_i, gen_i, reason):
    ff_name = param_names[0]
    # print(ff_name)
    path_hierarchy = __get_path_hierarchy(param_names, run_i)
    path = '/'.join(path_hierarchy)
    filename = f'{run_i}-{gen_i}-{reason}.xlsx'

    if not os.path.exists(path):
        os.makedirs(path)

    workbook = xlsxwriter.Workbook(f'{path}/{filename}')
    worksheet = workbook.add_worksheet()
    worksheet.name = ff_name

    set_chromosomes = set(population.chromosomes)
    chr_i = 0
    used = []
    for chr in set_chromosomes:
        row = chr_i
        # print(gen_stats)
        gen = ''.join(geno.decode('utf-8') for geno in chr.genotype)
        if gen in used:
            continue
        used.append(gen)
        worksheet.write(row, 0, gen)
        worksheet.write(row, 1, population.fitness_function.get_phenotype(chr.genotype))
        worksheet.write(row, 2, chr.fitness)
        worksheet.write(row, 3, len([True for c in population.chromosomes if c.genotype.tobytes() == chr.genotype.tobytes()]))

        chr_i += 1

    workbook.close()


def write_run_stats(gen_stats_list: list[GenerationStats],
        param_names: tuple[str],
        run_i, reason):
    r = ['difference', 'intensity', 'reproduction_rate', 'loss_of_diversity', 'Pfet', 'Ptau']
    if gen_stats_list is None:
        return
    ff_name = param_names[0]
    # print(ff_name)
    path_hierarchy = __get_path_hierarchy(param_names, run_i)
    path = '/'.join(path_hierarchy)
    filename = f'{run_i}-{reason}.xlsx'

    if ff_name == 'FconstALL':
        gen_stats_names = FCONSTALL_GEN_STATS_NAMES
    else:
        gen_stats_names = GEN_STATS_NAMES

    if not os.path.exists(path):
        os.makedirs(path)

    workbook = xlsxwriter.Workbook(f'{path}/{filename}')
    worksheet = workbook.add_worksheet()
    worksheet.name = ff_name
    worksheet.freeze_panes(1, 1)

    n = len([True for i in gen_stats_list])
    for gen_i, gen_stats in enumerate(gen_stats_list):
        row = gen_i + 1
        # print(gen_stats)
        worksheet.write(row, 0, gen_i)

        for stat_i, stat_name in enumerate(gen_stats_names):
            col = stat_i + 1
            t = getattr(gen_stats, stat_name)
            # print(t)
            if stat_name in r:
                worksheet.write(row + 1, col, t)
            else:
                worksheet.write(row, col, t)
            if gen_i == 0:
                worksheet.write(0, col, stat_name)

    workbook.close()

def write_ff_stats(experiment_stats_list: list[ExperimentStats]):
    ff_name = experiment_stats_list[0].params[0]
    path = f'{OUTPUT_FOLDER}/tables/{ff_name}'
    filename = f'{ff_name}_{N}.xlsx'

    if ff_name == 'FconstALL':
        run_stats_names = FCONSTALL_RUN_STATS_NAMES
        exp_stats_names = FCONSTALL_EXP_STATS_NAMES
    else:
        run_stats_names = RUN_STATS_NAMES
        exp_stats_names = EXP_STATS_NAMES

    if not os.path.exists(path):
        os.makedirs(path)
    
    workbook = xlsxwriter.Workbook(f'{path}/{filename}')
    worksheet = workbook.add_worksheet()
    worksheet.name = ff_name
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'fg_color': 'yellow'})
    worksheet.freeze_panes(3, 3)
    
    for exp_i, experiment_stats in enumerate(experiment_stats_list):
        row = exp_i + 2

        selection_params = experiment_stats.params[4]
        selection_params_keys = list(selection_params.keys())

        worksheet.write(row, 0, experiment_stats.params[1])
        i = 0
        for key in selection_params_keys:
            worksheet.write(row, i + 1, selection_params[key])
            i += 1
        worksheet.write(row, i+1, experiment_stats.params[2])
        worksheet.write(row, i+2, experiment_stats.params[3])

        run_stats_count = len(run_stats_names)
        for run_i, run_stats in enumerate(experiment_stats.runs):
            for stat_i, stat_name in enumerate(run_stats_names):
                col = run_i * run_stats_count + stat_i + 3 + len(selection_params_keys)
                worksheet.write(row, col, getattr(run_stats, stat_name))
                if exp_i == 0:
                    worksheet.write(1, col, stat_name)

            if exp_i == 0:
                start_col = run_i * run_stats_count + 3 + len(selection_params_keys)
                worksheet.merge_range(0, start_col, 0, start_col + run_stats_count - 1, f'Run {run_i}', merge_format)

        for stat_i, stat_name in enumerate(exp_stats_names):
            selection_params = experiment_stats.params[4]
            selection_params_keys = list(selection_params.keys())

            col = run_stats_count * NR + stat_i + 3 + len(selection_params_keys)
            worksheet.write(row, col, getattr(experiment_stats, stat_name))
            if exp_i == 0:
                    worksheet.write(1, col, stat_name)

        if exp_i == 0:
            selection_params = experiment_stats.params[4]
            selection_params_keys = list(selection_params.keys())

            start_col = run_stats_count * NR + 3 + len(selection_params_keys)

            worksheet.merge_range(0, start_col, 0, start_col + len(exp_stats_names) - 1 + len(selection_params_keys), f'Aggregated', merge_format)
            worksheet.merge_range(0, 0, 1, 0, 'Selection Method', merge_format)
            i = 0
            for key in selection_params_keys:
                i += 1
                worksheet.merge_range(0, i, 1, i, key, merge_format)
            worksheet.merge_range(0, i+1, 1, i+1, 'Genetic Operator', merge_format)
            worksheet.merge_range(0, i+2, 1, i+2, 'Init Population', merge_format)
       
    workbook.close()


def write_ff_aggregated_stats(experiment_stats_list: list[ExperimentStats]):
    ff_name = experiment_stats_list[0].params[0]
    path = f'{OUTPUT_FOLDER}/tables/{ff_name}'
    filename = f'aggregated_{ff_name}_{N}.xlsx'

    if ff_name == 'FconstALL':
        run_stats_names = FCONSTALL_RUN_STATS_NAMES
        exp_stats_names = FCONSTALL_EXP_STATS_NAMES
    else:
        run_stats_names = RUN_STATS_NAMES
        exp_stats_names = EXP_STATS_NAMES

    if not os.path.exists(path):
        os.makedirs(path)

    workbook = xlsxwriter.Workbook(f'{path}/{filename}')
    worksheet = workbook.add_worksheet()
    worksheet.name = ff_name
    merge_format = workbook.add_format({
        'bold': 1,
        'border': 1,
        'align': 'center',
        'fg_color': 'yellow'})
    worksheet.freeze_panes(3, 3)

    for exp_i, experiment_stats in enumerate(experiment_stats_list):
        row = exp_i + 2

        selection_params = experiment_stats.params[4]
        selection_params_keys = list(selection_params.keys())

        worksheet.write(row, 0, experiment_stats.params[1])
        i = 0
        for key in selection_params_keys:
            worksheet.write(row, i + 1, selection_params[key])
            i += 1
        worksheet.write(row, i + 1, experiment_stats.params[2])
        worksheet.write(row, i + 2, experiment_stats.params[3])
        run_stats_count = 0
        # run_stats_count = len(run_stats_names)
        # for run_i, run_stats in enumerate(experiment_stats.runs):
        #     for stat_i, stat_name in enumerate(run_stats_names):
        #         col = run_i * run_stats_count + stat_i + 3 + len(selection_params_keys)
        #         worksheet.write(row, col, getattr(run_stats, stat_name))
        #         if exp_i == 0:
        #             worksheet.write(1, col, stat_name)
        #
        #     if exp_i == 0:
        #         start_col = run_i * run_stats_count + 3 + len(selection_params_keys)
        #         worksheet.merge_range(0, start_col, 0, start_col + run_stats_count - 1, f'Run {run_i}', merge_format)

        for stat_i, stat_name in enumerate(exp_stats_names):
            selection_params = experiment_stats.params[4]
            selection_params_keys = list(selection_params.keys())

            col = run_stats_count * NR + stat_i + 3 + len(selection_params_keys)
            worksheet.write(row, col, getattr(experiment_stats, stat_name))
            if exp_i == 0:
                worksheet.write(1, col, stat_name)

        if exp_i == 0:
            selection_params = experiment_stats.params[4]
            selection_params_keys = list(selection_params.keys())

            start_col = run_stats_count * NR + 3 + len(selection_params_keys)

            worksheet.merge_range(0, start_col, 0, start_col + len(exp_stats_names) - 1 + len(selection_params_keys),
                                  f'Aggregated', merge_format)
            worksheet.merge_range(0, 0, 1, 0, 'Selection Method', merge_format)
            i = 0
            for key in selection_params_keys:
                i += 1
                worksheet.merge_range(0, i, 1, i, key, merge_format)
            worksheet.merge_range(0, i + 1, 1, i + 1, 'Genetic Operator', merge_format)
            worksheet.merge_range(0, i + 2, 1, i + 2, 'Init Population', merge_format)

    workbook.close()

def write_aggregated_stats(experiment_stats_list: list[ExperimentStats]):
    path = f'{OUTPUT_FOLDER}/tables'
    filename = f'aggregated_{N}.xlsx'

    if not os.path.exists(path):
        os.makedirs(path)

    workbook = xlsxwriter.Workbook(f'{path}/{filename}')
    worksheet = workbook.add_worksheet()
    worksheet.name = 'aggregated'
    worksheet.freeze_panes(1, 4)

    for exp_i, experiment_stats in enumerate(experiment_stats_list):

        selection_params = experiment_stats.params[4]
        selection_params_keys = list(selection_params.keys())


        if exp_i == 0:
            selection_params_keys = list(experiment_stats.params[4].keys())
            worksheet.write(0, 0, 'Fitness Function')
            worksheet.write(0, 1, 'Selection Method')
            i = 1
            for key in selection_params_keys:
                i += 1
                worksheet.write(0, i, key)
            # insert params
            worksheet.write(0, i + 1, 'Genetic Operator')
            worksheet.write(0, i + 2, 'Initial Population')

        row = exp_i + 1


        worksheet.write(row, 0, experiment_stats.params[0])
        worksheet.write(row, 1, experiment_stats.params[1])
        i = 1
        for key in selection_params_keys:
            worksheet.write(row, i + 1, selection_params[key])
            i += 1
        # insert params
        worksheet.write(row, i + 1, experiment_stats.params[2])
        worksheet.write(row, i + 2 , experiment_stats.params[3])
        
        for stat_i, stat_name in enumerate(EXP_STATS_NAMES):
            col = stat_i + 4 + len(selection_params_keys)
            worksheet.write(row, col, getattr(experiment_stats, stat_name))
            if exp_i == 0:
                worksheet.write(0, col, stat_name)

    workbook.close()


def __get_path_hierarchy(param_names, run_i):
    selection_params = param_names[4]
    param = ''
    for i in selection_params.keys():
        param += i + '=' + selection_params[i]
    return [
        OUTPUT_FOLDER,
        'tables',
        param_names[0], # fitness function
        str(N),
        param_names[1] + param, # selection method
        param_names[2], # genetic operator
        param_names[3], # init method
        str(run_i)
    ]