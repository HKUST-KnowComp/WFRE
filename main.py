import argparse
import collections
import os
from pprint import pprint
import pandas as pd
import torch
from tqdm.std import trange, tqdm

from fol.appfoq import compute_final_loss
from data_helper import TaskManager, BenchmarkFormManager, all_normal_form, BenchmarkWholeManager
from fol import BetaEstimator, BoxEstimator, LogicEstimator, NLKEstimator, BetaEstimator4V, order_bounds
from fol.estimator_fuzzle import FuzzleEstiamtor
from fol.estimator_wasserstein import WassersteinEstimator
from utils.util import (Writer, load_data_with_indexing, load_task_manager, read_from_yaml,
                        set_global_seed)

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/papers/NELL.yaml', type=str)
parser.add_argument('--prefix', default='EFO-1_train', type=str)
parser.add_argument('--checkpoint_path', default= None, type=str)
parser.add_argument('--load_step', default=None, type=int)


#ToDo:new csv to store valid and test scores.
#ToDo:early stop by valid's MRR.

def proj_simplex_tensor(emb_grad, sum_demand=1):
    #  suitable for any dimension tensor
    #  and output is the tensor 's last dimension is a discrete probably distribution
    #  detailed proof for this projection algorithm ,please read https://arxiv.org/pdf/1309.1541.pdf
    dim = emb_grad.shape[-1]

    sort_a = torch.sort(emb_grad, descending=True)[0]
    cssv = torch.cumsum(sort_a, dim=-1) - sum_demand
    index = torch.arange(dim,device=emb_grad.device) + 1
    cssv_d = sort_a - cssv / index > 0
    rho = torch.sum(cssv_d, dim=-1, keepdim=True)
    # select the suitable index
    lam = torch.gather(cssv, -1, rho - 1) / rho
    x = torch.relu(emb_grad - lam)
    return torch.flatten(x, start_dim=-2)


def typestr2benchmarkname(type_str: str, mode=None, step=None):
    if mode:
        return f'eval_{mode}_{step}_{type_str}.csv'
    else:
        return f'eval_{type_str}.csv'


def log_benchmark(folder_path, id_list, percentage=False, mode=None, step=None):
    queries = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in", "inp", "pin", "pni", "2u", "up"]
    task_dist = {}
    for i in range(len(queries)):
        task_dist[i] = queries[i]
    all_log = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(float)))
    for task_id in id_list:
        type_str = f'type{task_id:04d}'
        filename = typestr2benchmarkname(type_str, mode, step)
        # real_index = all_formula.loc[all_formula['formula_id'] == f'type{id_str}'].index[0]
        if os.path.exists(os.path.join(folder_path, filename)):
            single_log = pd.read_csv(os.path.join(folder_path, filename))
            index2metrics = single_log['Unnamed: 0']
            for normal_form in single_log.columns:
                if normal_form != 'Unnamed: 0':
                    for index in range(len(single_log[normal_form])):
                        if percentage and index2metrics[index] != 'num_queries':
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index] * 100
                        else:
                            all_log[index2metrics[index]][normal_form][task_id] = single_log[normal_form][index]
    for metric in all_log:
        data_metric = pd.DataFrame.from_dict(all_log[metric])
        if mode:
            if metric == "MRR":
                data_metric.insert(loc=0, column="task_name",value=pd.Series(["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in", "inp", "pin", "pni", "2u", "up"]))
                epfo_list = [i for i in range(len(queries)) if "n" not in queries[i]]
                n_list = [i for i in range(len(queries)) if "n" in queries[i]]
                data_metric.loc["epfo_avg"] = data_metric.iloc[epfo_list].mean()
                data_metric.loc["n_avg"] = data_metric.iloc[n_list].mean()
                data_metric.to_csv(os.path.join(folder_path, f'all_formula_{mode}_{step}_{metric}.csv'))
        else:
            data_metric.to_csv(os.path.join(folder_path, f'all_formula_{metric}.csv'))
    return all_log


def collect_steps_scores(folder_path, mode, steps, evaluate_steps):
    # output MRR "DNF"
    queries = ["1p", "2p", "3p", "2i", "3i", "ip", "pi", "2in", "3in", "inp", "pin", "pni", "2u", "up"]
    step_log = collections.defaultdict(lambda: collections.defaultdict(float))
    task_dist = {}
    for i in range(len(queries)):
        task_dist[i] = queries[i]
    step_log["query_name"] = task_dist
    for step in range(evaluate_steps, steps + evaluate_steps):
        path_index = os.path.join(folder_path, f"all_formula_{mode}_{step}_MRR.csv")
        if os.path.exists(path_index):
            df_index = pd.read_csv(path_index)
            step_log[f"{step}_DNF"] = df_index["DNF"]
            print(df_index)
    df_save = pd.DataFrame.from_dict(data=step_log)
    queries = df_save["query_name"]
    epfo_list = [i for i in range(len(queries)) if "n" not in queries[i]]
    n_list = [i for i in range(len(queries)) if "n" in queries[i]]
    df_save.loc["epfo_avg"] = df_save.iloc[epfo_list].mean()
    df_save.loc["n_avg"] = df_save.iloc[n_list].mean()
    df_save.to_csv(os.path.join(folder_path, f"all_{mode}.csv"))

    return step_log


def train_step(model, opt, iterator):
    model.train()
    opt.zero_grad()
    data = next(iterator)
    emb_list, answer_list = [], []
    union_emb_list, union_answer_list = [], []
    for formula in data:
        if 'u' in formula or 'U' in formula:  # TODO: consider 'evaluate_union' in the future
            union_emb_list.append(data[formula]['emb'])
            union_answer_list.append(data[formula]['answer_set'])
        else:
            emb_list.append(data[formula]['emb'])
            answer_list.extend(data[formula]['answer_set'])
    for formula in data: #ToDo: why twice?
            emb_list.append(data[formula]['emb'])
            answer_list.extend(data[formula]['answer_set'])
    pred_embedding = torch.cat(emb_list, dim=0)
    all_positive_logit, all_negative_logit, all_subsampling_weight = model.criterion(pred_embedding, answer_list)
    positive_loss, negative_loss = compute_final_loss(all_positive_logit, all_negative_logit, all_subsampling_weight)
    loss = (positive_loss + negative_loss) / 2 
    loss.backward()
    opt.step()
    log = {
        'po': positive_loss.item(),
        'ne': negative_loss.item(),
        'loss': loss.item()
    }
    if model.name == 'logic':
        entity_embedding = model.entity_embeddings.weight.data
        if model.bounded:
            model.entity_embeddings.weight.data = order_bounds(entity_embedding)
        else:
            model.entity_embeddings.weight.data = torch.clamp(entity_embedding, 0, 1)
    elif model.name == "wasserstein_pgd":
        entity_embedding_dis = model.distribute(model.entity_embedding.data)
        model.entity_embedding.data = proj_simplex_tensor(entity_embedding_dis)
        #proj里已经拉平过
    elif model.name == 'wasserstein_uot_transformer':
        model.entity_embedding.data = torch.clamp(model.entity_embedding.data, 0, 1)
        assert torch.isnan(model.entity_embedding.data).sum() == 0
    return log


def eval_step(model, eval_iterator, device, mode, allowed_easy_ans=False):
    model.eval()
    logs = collections.defaultdict(lambda: collections.defaultdict(float))
    with torch.no_grad():
        for data in tqdm(eval_iterator):
            for key in data:
                pred = data[key]['emb']
                all_logit = model.compute_all_entity_logit(pred, union=('u' in key or 'U' in key))  # batch*nentity
                argsort = torch.argsort(all_logit, dim=1, descending=True)
                ranking = argsort.clone().to(torch.float)
                #  create a new torch Tensor for batch_entity_range
                if device != torch.device('cpu'):
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1).to(
                            device))
                else:
                    ranking = ranking.scatter_(
                        1, argsort, torch.arange(model.n_entity).to(torch.float).repeat(argsort.shape[0], 1))
                # achieve the ranking of all entities
                for i in range(all_logit.shape[0]):
                    if mode == 'train':
                        easy_ans = []
                        hard_ans = data[key]['answer_set'][i]
                    else:
                        if allowed_easy_ans:
                            easy_ans = []
                            hard_ans = list(set(data[key]['hard_answer_set'][i]).union
                                            (set(data[key]['easy_answer_set'][i])))
                        else:
                            easy_ans = data[key]['easy_answer_set'][i]
                            hard_ans = data[key]['hard_answer_set'][i]

                    num_hard = len(hard_ans)
                    num_easy = len(easy_ans)
                    assert len(set(hard_ans).intersection(set(easy_ans))) == 0
                    # only take those answers' rank
                    cur_ranking = ranking[i, list(easy_ans) + list(hard_ans)]
                    cur_ranking, indices = torch.sort(cur_ranking)
                    masks = indices >= num_easy
                    if device != torch.device('cpu'):
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float).to(device)
                    else:
                        answer_list = torch.arange(
                            num_hard + num_easy).to(torch.float)

                    cur_ranking = cur_ranking - answer_list + 1
                    # filtered setting: +1 for start at 0, -answer_list for ignore other answers

                    cur_ranking = cur_ranking[masks]
                    # only take indices that belong to the hard answers
                    mrr = torch.mean(1. / cur_ranking).item()
                    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                    h10 = torch.mean(
                        (cur_ranking <= 10).to(torch.float)).item()
                    add_hard_list = torch.arange(num_hard).to(torch.float).to(device)
                    hard_ranking = cur_ranking + add_hard_list  # for all hard answer, consider other hard answer
                    logs[key]['retrieval_accuracy'] += torch.mean(
                        (hard_ranking <= num_hard).to(torch.float)).item()
                    logs[key]['MRR'] += mrr
                    logs[key]['HITS1'] += h1
                    logs[key]['HITS3'] += h3
                    logs[key]['HITS10'] += h10
                num_query = all_logit.shape[0]
                logs[key]['num_queries'] += num_query
        for key in logs.keys():
            for metric in logs[key].keys():
                if metric != 'num_queries':
                    logs[key][metric] /= logs[key]['num_queries']
    # torch.cuda.empty_cache()
    return logs


# def training(model, opt, train_iterator, valid_iterator, test_iterator, writer, **train_cfg):
#     lr = train_cfg['learning_rate']
#     with tqdm.trange(train_cfg['steps']) as t:
#         for step in t:
#             log = train_step(model, opt, train_iterator, writer)
#             t.set_postfix({'loss': log['loss']})
#             if step % train_cfg['evaluate_every_steps'] and step > 0:
#                 eval_step(model, valid_iterator, 'valid', writer, **train_cfg)
#                 eval_step(model, test_iterator, 'test', writer, **train_cfg)

#             if step >= train_cfg['warm_up_steps']:
#                 lr /= 5
#                 # logging
#                 opt = torch.optim.Adam(
#                     filter(lambda p: p.requires_grad, model.parameters()),
#                     lr=lr
#                 )
#                 train_cfg['warm_up_steps'] *= 1.5
#             if step % train_cfg['save_every_steps']:
#                 pass
#             if step % train_cfg['log_every_steps']:
#                 pass

def save_eval(log, mode, step, writer):
    for t in log:
        logt = log[t]
        logt['step'] = step
        writer.append_trace(f'eval_{mode}_{t}', logt)


def save_benchmark(log, writer, step, taskmanger: BenchmarkFormManager):
    form_log = collections.defaultdict(lambda: collections.defaultdict(float))
    for normal_form in all_normal_form:
        formula = taskmanger.form2formula[normal_form]
        if formula in log:
            form_log[normal_form] = log[formula]
    writer.save_dataframe(form_log, f"eval_{taskmanger.mode}_{step}_{taskmanger.query_inform_dict['formula_id']}.csv")


def save_whole_benchmark(log, writer, step, whole_task_manager: BenchmarkWholeManager):
    for type_str in whole_task_manager.query_classes:
        save_benchmark(log, writer, step, whole_task_manager.query_classes[type_str])


def load_beta_model(checkpoint_path, model, optimizer):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        args.checkpoint_path, 'checkpoint'))
    init_step = checkpoint['step']
    model.load_state_dict(checkpoint['model_state_dict'])
    current_learning_rate = checkpoint['current_learning_rate']
    warm_up_steps = checkpoint['warm_up_steps']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return current_learning_rate, warm_up_steps, init_step


def load_model(step, checkpoint_path, model, opt):
    print('Loading checkpoint %s...' % checkpoint_path)
    checkpoint = torch.load(os.path.join(
        checkpoint_path, f'{step}.ckpt'), map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_parameter'])
    opt.load_state_dict(checkpoint['optimizer_parameter'])
    learning_rate = train_config['learning_rate']
    warm_up_steps = checkpoint['warm_up_steps'] # here has fixed.
    return learning_rate, warm_up_steps


if __name__ == "__main__":

    args = parser.parse_args()
    # parse args and load config
    # configure = read_from_yaml('config/default.yaml')
    configure = read_from_yaml(args.config)
    print("[main] config loaded")
    pprint(configure)
    # initialize my log writer
    if configure['data']['type'] == 'beta':
        case_name = f'{args.prefix}/{args.config.split("/")[-1].split(".")[0]}'
        # case_name = 'dev/default'
        writer = Writer(case_name=case_name, config=configure, log_path='log')
        # writer = SummaryWriter('./logs-debug/unused-tb')
    else:
        if 'train' in configure['action']:
            case_name = f'{args.prefix}/{args.config.split("/")[-1].split(".")[0]}'
        else:
            case_name = f'{args.prefix}/{args.checkpoint_path.split("/")[-1]}'
        writer = Writer(case_name=case_name, config=configure, log_path='EFO-1_log')

    # initialize environments
    set_global_seed(configure.get('seed', 0))
    if configure.get('cuda', -1) >=0 and torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(configure['cuda']))
        # logging.info('Device use cuda: %s' % configure['cuda'])
    else:
        device = torch.device('cpu')

    # prepare the procedure configs
    train_config = configure['train']
    train_config['device'] = device
    eval_config = configure['evaluate']
    eval_config['device'] = device

    # load the data
    print("[main] loading the data")
    data_folder = configure['data']['data_folder']
    entity_dict, relation_dict, projection_train, reverse_projection_train, projection_valid, reverse_projection_valid,\
        projection_test, reverse_projection_test = load_data_with_indexing(data_folder)
    n_entity, n_relation = len(entity_dict), len(relation_dict)

    # get model
    model_name = configure['estimator']['embedding']
    model_params = configure['estimator'][model_name]
    model_params['n_entity'], model_params['n_relation'] = n_entity, n_relation
    model_params['negative_sample_size'] = train_config['negative_sample_size']
    model_params['device'] = device
    if model_name == 'beta':
        model = BetaEstimator4V(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'box':
        model = BoxEstimator(**model_params)
        allowed_norm = ['DNF+MultiIU']
    elif model_name == 'logic':
        model = LogicEstimator(**model_params)
        allowed_norm = ['DeMorgan', 'DNF+MultiIU']
    elif model_name == 'NewLook':
        model = NLKEstimator(**model_params)
        model.setup_relation_tensor(projection_train)
        allowed_norm = ['DNF+MultiIUD']
    elif model_name == 'Wasserstein_comp':
        model = WassersteinEstimator4(**model_params)
        allowed_norm = ['DeMorgan','DNF+MultiIU']
    elif model_name == 'Wasserstein':
        model = WassersteinEstimator(**model_params)
        allowed_norm = ['DeMorgan','DNF+MultiIU']
    elif model_name == "fuzzle":
        model = FuzzleEstiamtor(**model_params)
        allowed_norm = ['DeMorgan','DNF+MultiIU']
    else:
        assert False, 'Not valid model name!'
    model.to(device)
    valid_tm_list, test_tm_list = [], []
    train_path_iterator, train_path_tm, train_other_iterator, train_other_tm = None, None, None, None
    train_iterator, train_tm, valid_iterator, valid_tm, test_iterator, test_tm = None, None, None, None, None, None
    if configure['data']['type'] == 'beta':
        if 'train' in configure['action']:
            print("[main] load training data")
            beta_path_tasks, beta_other_tasks = [], []
            for task in train_config['meta_queries']:
                if task in ['1p', '2p', '3p']:
                    beta_path_tasks.append(task)
                else:
                    beta_other_tasks.append(task)

            path_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_path_tasks)
            other_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=beta_other_tasks)
            if len(beta_path_tasks) > 0:
                train_path_tm = TaskManager('train', path_tasks, device)
                train_path_iterator = train_path_tm.build_iterators(model, batch_size=train_config['batch_size'])
            if len(beta_other_tasks) > 0:
                train_other_tm = TaskManager('train', other_tasks, device)
                train_other_iterator = train_other_tm.build_iterators(model, batch_size=train_config['batch_size'])
            all_tasks = load_task_manager(
                configure['data']['data_folder'], 'train', task_names=train_config['meta_queries'])
            train_tm = TaskManager('train', all_tasks, device)
            train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])

        if 'valid' in configure['action']:
            print("[main] load valid data")
            tasks = load_task_manager(configure['data']['data_folder'], 'valid',
                                      task_names=configure['evaluate']['meta_queries'])
            valid_tm = TaskManager('valid', tasks, device)
            valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])

        if 'test' in configure['action']:
            print("[main] load test data")
            tasks = load_task_manager(configure['data']['data_folder'], 'test',
                                      task_names=configure['evaluate']['meta_queries'])
            test_tm = TaskManager('test', tasks, device)
            test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
    elif configure['data']['type'] == 'EFO-1':
        if 'train' in configure['action']:
            train_formula_id_file = configure['train']['formula_id_file']
            train_formula_id_data = pd.read_csv(train_formula_id_file)
            path_formulas_index_list, other_formulas_index_list = [], []
            other_ops = ['i', 'I', 'u', 'U', 'n', 'd', 'D']
#            other_ops = ['u', 'U']
            for index in train_formula_id_data.index:
                original_formula = train_formula_id_data['original'][index]
                if True not in [ops in original_formula for ops in other_ops]:
                    path_formulas_index_list.append(index)
                else:
                    other_formulas_index_list.append(index)
            path_formula_id_data = train_formula_id_data.loc[path_formulas_index_list]
            other_formula_id_data = train_formula_id_data.loc[other_formulas_index_list]
            train_path_tm = BenchmarkWholeManager('train', path_formula_id_data, data_folder,
                                                  configure['train']['interested_normal_forms'], device, model)
            train_path_iterator = train_path_tm.build_iterators(model, configure['train']['batch_size'])
            train_other_tm = BenchmarkWholeManager('train', other_formula_id_data, data_folder,
                                                   configure['train']['interested_normal_forms'], device, model)
            train_other_iterator = train_other_tm.build_iterators(model, configure['train']['batch_size'])

        if 'valid' in configure['action']:
            valid_formula_id_file = configure['evaluate']['formula_id_file']
            valid_formula_id_data = pd.read_csv(valid_formula_id_file)
            for i in valid_formula_id_data.index:
                type_str = valid_formula_id_data['formula_id'][i]
                filename = os.path.join(data_folder, f'valid-{type_str}.csv')
                valid_tm = BenchmarkFormManager('valid', valid_formula_id_data.loc[i], filename, device, model)
                valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                valid_tm_list.append(valid_tm)

        if 'test' in configure['action']:
            test_formula_id_file = configure['evaluate']['formula_id_file']
            test_formula_id_data = pd.read_csv(test_formula_id_file)
            for i in test_formula_id_data.index:
                type_str = test_formula_id_data['formula_id'][i]
                old_filename = os.path.join(data_folder, f'data-{type_str}.csv')
                if os.path.exists(old_filename):
                    test_tm = BenchmarkFormManager('test', test_formula_id_data.loc[i], old_filename, device, model)
                else:
                    filename = os.path.join(data_folder, f'test-{type_str}.csv')
                    test_tm = BenchmarkFormManager('test', test_formula_id_data.loc[i], filename, device, model)
                test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                test_tm_list.append(test_tm)
    else:
        assert False, 'Not valid data type!'

    lr = train_config['learning_rate']
    weight_decay = train_config["weight_decay"]
#    opt = adai_optim.Adai(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, betas=(0.1, 0.99), eps=1e-03, weight_decay=5e-4)
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    init_step = 1
    # exit()
    assert 2 * train_config['warm_up_steps'] == train_config['steps']
    if args.checkpoint_path is not None:
        if args.load_step != 0:
            lr, train_config['warm_up_steps'] = load_model(args.load_step, args.checkpoint_path, model, opt)
            init_step = args.load_step
        else:
            lr, train_config['warm_up_steps'], init_step = load_beta_model(args.checkpoint_path, model, opt)

    training_logs = []
    if configure['data']['type'] == 'EFO-1' and 'train' not in configure['action']:
        assert train_config['steps'] == init_step
    with trange(init_step, train_config['steps'] + 1) as t:
        for step in t:
            # basic training step
            if train_path_iterator:
                if step >= train_config['warm_up_steps']:
                    lr /= 5
                    # logging
                    opt = torch.optim.AdamW(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        lr=lr,
                        weight_decay=weight_decay
                    )
                    train_config['warm_up_steps'] *= 1.5
                try:
                    _log = train_step(model, opt, train_path_iterator)
                except StopIteration:
                    print("new epoch for path meta-query")
                    train_path_iterator = train_path_tm.build_iterators(model, batch_size=train_config['batch_size'])
                    _log = train_step(model, opt, train_path_iterator)
                if train_other_iterator:
                    try:
                        _log_other = train_step(model, opt, train_other_iterator)
                        try:
                            _log_second = train_step(model, opt, train_path_iterator)
                        except StopIteration:
                            print("new epoch for path meta-query")
                            train_path_iterator = train_path_tm.build_iterators(model,
                                                                                batch_size=train_config['batch_size'])
                            _log_second = train_step(model, opt, train_path_iterator)
                    except StopIteration:
                        print("new epoch for other meta-query")
                        train_other_iterator = \
                            train_other_tm.build_iterators(model, batch_size=train_config['batch_size'])
                        _log_other = train_step(model, opt, train_other_iterator)
                        try:
                            _log_second = train_step(model, opt, train_path_iterator)
                        except StopIteration:
                            print("new epoch for path meta-query")
                            train_path_iterator = train_path_tm.build_iterators(model,
                                                                                batch_size=train_config['batch_size'])
                            _log_second = train_step(model, opt, train_path_iterator)
                    _alllog = {}
                    for key in _log:
                        _alllog[f'all_{key}'] = (_log[key] + _log_other[key]) / 2
                        _alllog[key] = _log[key]
                    _log = _alllog
                t.set_postfix(_log)
                training_logs.append(_log)
                if step % train_config['log_every_steps'] == 0:
                    for metric in training_logs[0].keys():
                        _log[metric] = sum(log[metric] for log in training_logs) / len(training_logs)
                    _log['step'] = step
                    training_logs = []
                    writer.append_trace('train', _log)

            if step % train_config['evaluate_every_steps'] == 0 or step == train_config['steps']:
                if configure['data']['type'] == 'beta':
                    '''
                    if train_iterator:
                        train_iterator = train_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, train_iterator, device, mode='train')
                        save_eval(_log, 'train', step, writer)
                    '''

                    if valid_iterator:
                        valid_iterator = valid_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, valid_iterator, device, mode='valid')
                        save_eval(_log, 'valid', step, writer)

                    if test_iterator:
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, test_iterator, device, mode='test')
                        save_eval(_log, 'test', step, writer)
                elif configure['data']['type'] == 'EFO-1':
                    # todo: test_in_train, namely test the train dataset
                    id_list = [i for i in range(14)]
                    log_file = os.path.join("EFO-1_log/", writer.idstr)
                    for valid_tm in valid_tm_list:
                        valid_iterator = valid_tm.build_iterators(model, configure['evaluate']['batch_size'])
                        _log = eval_step(model, valid_iterator, device, 'valid')
                        save_benchmark(_log, writer, step, valid_tm)
                    log_benchmark(log_file, id_list, mode="valid", step=step)
                    # todo:add a function to collect mrr
                    for test_tm in test_tm_list:
                        test_iterator = test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                        _log = eval_step(model, test_iterator, device, mode='test')
                        '''
                            test_iterator = 
                            test_tm.build_iterators(model, batch_size=configure['evaluate']['batch_size'])
                            _log_easy = eval_step(model, test_iterator, device, mode='test', allowed_easy_ans=True)
                            for formula in _log_easy:
                                for metrics in _log_easy[formula]:
                                    _log[formula][f'easy_{metrics}'] = _log_easy[formula][metrics]
                        '''
                        save_benchmark(_log, writer, step, test_tm)
                    log_benchmark(log_file, id_list, mode="test", step=step)

            if step % train_config['save_every_steps'] == 0 and train_path_iterator:
                writer.save_model(model, opt, step, train_config['warm_up_steps'], lr)
