"""Isolated AD -> RAD transfer protocol; no legacy entrypoint imports this module."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

import h5py
import numpy as np
import torch

from baseline_dataset import collection_task_ids, optimal_labels
from compressor_experiment import select_dataset_groups
from model.ad import AD
from model.compressed_ad import RAD
from utils import get_traj_file_name, normalize_compiled_state_dict

PROTOCOL = 'gridworld-ad-memory-transfer-v1'
PROJECT = Path(__file__).resolve().parent
ARCHITECTURE = ('tf_n_embd', 'tf_n_head', 'tf_n_layer', 'tf_dim_feedforward', 'tf_dropout')
AD_PREFIXES = ('ad_transformer.', 'embed_state.', 'embed_action.', 'embed_reward.',
               'pred_action.', 'type_embedding')


def digest(filename):
    with Path(filename).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_checkpoint(filename):
    checkpoint = torch.load(filename, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, dict) or not {'config', 'model', 'step'} <= checkpoint.keys():
        raise ValueError('Expected a checkpoint with config, model, and step')
    checkpoint['model'] = normalize_compiled_state_dict(checkpoint['model'])
    return checkpoint


def load_source(filename):
    checkpoint = read_checkpoint(filename)
    config = deepcopy(checkpoint['config'])
    if config.get('model') != 'AD' or config.get('env') not in ('darkroom', 'dktd'):
        raise ValueError('Source must be a Gridworld SAR AD checkpoint')
    if checkpoint['step'] <= 0:
        raise ValueError('Source AD checkpoint must have at least one trained update')
    config['device'] = 'cpu'
    # A strict load validates tokenization and architecture, including optional legacy defaults.
    source = AD(config)
    source.load_state_dict(checkpoint['model'], strict=True)
    config.setdefault('tf_n_head', 4)
    config.setdefault('tf_n_layer', 4)
    config.setdefault('tf_dim_feedforward', config['tf_n_embd'] * 4)
    config.setdefault('tf_dropout', 0.1)
    checkpoint['config'] = config
    return checkpoint


def resolve_config(settings, source, source_path, dataset_dir, seed):
    """Architecture and task split come from the source, not today's AD YAML."""
    cfg = deepcopy(settings)
    src = source['config']
    if cfg['env'] != src['env']:
        raise ValueError('Experiment and AD checkpoint environments differ')
    for key in ARCHITECTURE:
        cfg[key] = src[key]
    for key in ('grid_size', 'num_actions', 'horizon', 'env_split_seed', 'train_env_ratio',
                'alg', 'alg_seed'):
        cfg[key] = src[key]
    cfg['dataset_task_mapping'] = src.get('dataset_task_mapping', 'legacy')
    if 'collection_env_split_seed' in src:
        if cfg['collection_env_split_seed'] != src['collection_env_split_seed']:
            raise ValueError('Collection seed disagrees with source checkpoint')
    cfg.update(protocol=PROTOCOL, model='RAD', device='cpu', seed=seed,
               traj_dir=str(Path(dataset_dir).resolve()), always_use_latent_prefix=False,
               compressor_type='ae', torch_compile=False, dynamics=False)
    cfg['source_ad'] = dict(path=str(Path(source_path).resolve()), sha256=digest(source_path),
                            step=source['step'], config={k: v for k, v in src.items() if k != 'device'})
    cfg['curriculum_schedule'] = [dict(stage, step=int(stage['fraction'] * cfg['train_timesteps']))
                                  for stage in cfg.pop('transfer_curriculum')]
    validate_config(cfg)
    if cfg['n_transit'] > src['n_transit']:
        raise ValueError('RAD context exceeds trained AD positional table; expansion is unsupported')
    return cfg


def validate_config(cfg):
    if cfg.get('protocol') != PROTOCOL or cfg.get('always_use_latent_prefix') is not False:
        raise ValueError('Invalid transfer protocol or initial latent prefix')
    if cfg.get('compressor_type') != 'ae' or cfg.get('torch_compile'):
        raise ValueError('Transfer currently requires an eager AE compressor')
    for key in ('pretrain_timesteps', 'train_timesteps', 'pretrain_batch_size', 'train_batch_size',
                'n_transit', 'n_compress_tokens', 'train_source_timesteps', 'train_n_stream',
                'eval_interval', 'ckpt_interval', 'eval_episodes', 'eval_batch_size',
                'log_interval', 'compress_n_heads', 'compress_n_layers'):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if cfg['n_compress_tokens'] % 3 or cfg['tf_n_embd'] % cfg['compress_n_heads']:
        raise ValueError('Invalid latent count or compressor head dimension')
    if not 0 < cfg['short_memory_keep'] < cfg['n_transit'] - cfg['n_compress_tokens'] // 3:
        raise ValueError('Recent context must leave room for new transitions after compression')
    if cfg['train_source_timesteps'] <= cfg['n_transit'] or cfg['max_context_length'] <= cfg['n_transit']:
        raise ValueError('Training histories must extend beyond first compression')
    if cfg['max_gradient_rounds'] < 1:
        raise ValueError('Joint training requires gradients through compression')
    if not cfg.get('save_best_model'):
        raise ValueError('This protocol requires reward-based best-model selection')
    if not 0 <= cfg['warmup_fraction'] < 1:
        raise ValueError('warmup_fraction must be in [0, 1)')
    for key in ('pretrain_lr', 'ad_lr', 'compression_lr', 'latent_lr'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if cfg['latent_update_mode'] not in RAD.LATENT_UPDATE_MODES:
        raise ValueError('Unknown memory update mode')
    steps = [item['step'] for item in cfg['curriculum_schedule']]
    if not steps or steps[0] != 0 or steps != sorted(set(steps)) or steps[-1] >= cfg['train_timesteps']:
        raise ValueError('Curriculum stages must be distinct and fit inside the update budget')


def initialize_ad(model, source):
    src_cfg = source['config']
    for key in ARCHITECTURE + ('grid_size', 'num_actions', 'env'):
        if model.config[key] != src_cfg[key]:
            raise ValueError(f'AD transfer mismatch: {key}')
    state = model.state_dict()
    imported = set()
    for key, value in source['model'].items():
        target = 'ad_transformer.' + key[len('transformer.'):] if key.startswith('transformer.') else key
        if target not in state or not target.startswith(AD_PREFIXES):
            raise ValueError(f'Unexpected source parameter: {key}')
        if target == 'ad_transformer.pos_embedding':
            if value.shape[1] < state[target].shape[1]:
                raise ValueError('Cannot expand pretrained positional embeddings')
            value = value[:, :state[target].shape[1], :]
        if value.shape != state[target].shape:
            raise ValueError(f'AD transfer shape mismatch: {key}')
        state[target] = value
        imported.add(target)
    expected = {key for key in state if key.startswith(AD_PREFIXES)}
    if imported != expected:
        raise ValueError(f'Missing AD parameters: {sorted(expected - imported)}')
    model.load_state_dict(state, strict=True)


def configure_stage(model, stage):
    if stage not in ('pretrain', 'finetune'):
        raise ValueError(f'Unknown training stage: {stage}')
    active = {'replace': (), 'residual': ('latent_residual_norm.',),
              'multiplicative_gate': ('latent_multiplicative_gate.',),
              'gru_gate': ('latent_gru_gate.', 'latent_gru_candidate.')}[model.latent_update_mode]
    prefixes = (('compression_transformer.', 'reconstruction_decoder.') if stage == 'pretrain'
                else AD_PREFIXES + ('compression_transformer.', 'latent_type_embedding') + active)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(prefixes))
    model.train()
    if stage == 'pretrain':
        model.ad_transformer.eval()
    else:
        model.reconstruction_decoder.eval()


def audit_dataset(cfg):
    """Audit actual group membership; sample reward identities in every used group."""
    filename = Path(cfg['traj_dir']) / (get_traj_file_name(cfg) + '.hdf5')
    # select_dataset_groups legacy branch seeds global random; preserve it.
    state = random.getstate()
    try:
        groups = select_dataset_groups(cfg, 'train')
        src = dict(cfg['source_ad']['config'], collection_env_split_seed=cfg['collection_env_split_seed'])
        source_groups = select_dataset_groups(src, 'train')
    finally:
        random.setstate(state)
    power = {'darkroom': 2, 'dktd': 4}[cfg['env']]
    order = collection_task_ids(cfg['grid_size'], power, cfg['collection_env_split_seed'])
    train_tasks = sorted({order[g] for g in groups})
    source_tasks = {order[g] for g in source_groups}
    eval_tasks = sorted(set(order) - set(train_tasks) - source_tasks)
    if not eval_tasks:
        raise ValueError('No tasks remain held out from both AD and transfer training')
    with h5py.File(filename, 'r') as history:
        for group_id in sorted(set(groups) | set(source_groups)):
            if str(group_id) not in history:
                raise ValueError(f'Missing source history group {group_id}')
            group = history[str(group_id)]
            t, streams = group['actions'].shape
            if t < cfg['train_source_timesteps'] or streams < cfg['train_n_stream']:
                raise ValueError(f'Group {group_id} has insufficient timesteps/streams')
            end = min(t, 2 * cfg['horizon'])
            states = group['states'][:end, :1].transpose(1, 0, 2)
            actions = group['actions'][:end, :1].T
            rewards = group['rewards'][:end, :1].T
            optimal_labels(cfg, group_id, states, actions, rewards)
    stat = filename.stat()
    return dict(path=str(filename), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                train_groups=list(groups), train_task_ids=train_tasks,
                source_train_task_ids=sorted(source_tasks), eval_task_ids=eval_tasks,
                collection_env_split_seed=cfg['collection_env_split_seed'],
                identity_check='first stream, first two episodes of each training group',
                source_provenance='training membership inferred from AD config; original dataset bytes unverified',
                evaluation_split='complement of actual AD and transfer training tasks; may differ from legacy nominal test split')


def capture_rng(data_rng=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                data=data_rng.getstate() if data_rng is not None else None)


def restore_rng(state, data_rng=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    if data_rng is not None:
        data_rng.setstate(state['data'])


class EvaluationRAD(RAD):
    """Change only the policy-visible memory contents, preserving layout and recurrence."""
    memory_intervention = 'intact'

    def _forward_ad_transformer(self, x, has_latent_prefix=False):
        if has_latent_prefix and self.memory_intervention != 'intact':
            memory, recent = x[:, :self.n_compress_tokens], x[:, self.n_compress_tokens:]
            if self.memory_intervention == 'zero':
                memory = torch.zeros_like(memory)
            elif self.memory_intervention == 'shuffle':
                # Shuffle slots with a separate generator; action RNG stays paired.
                order = torch.randperm(self.n_compress_tokens, generator=self.intervention_rng).to(x.device)
                memory = memory[:, order]
            else:
                raise ValueError('Unknown memory intervention')
            x = torch.cat((memory, recent), dim=1)
        return super()._forward_ad_transformer(x, has_latent_prefix)


@contextmanager
def isolated_evaluation(model, seed):
    state = capture_rng()
    training = model.training
    limit = getattr(model, 'max_compressions', None)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model.eval()
        if isinstance(model, RAD):
            model.set_curriculum(None)
        yield
    finally:
        model.train(training)
        if isinstance(model, RAD):
            model.set_curriculum(limit)
        restore_rng(state)


def evaluate_policy(model, cfg, seed=0, intervention='intact'):
    from env import make_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    if intervention != 'intact' and not isinstance(model, EvaluationRAD):
        raise ValueError('Memory interventions require EvaluationRAD')
    if isinstance(model, EvaluationRAD):
        model.memory_intervention = intervention
        model.intervention_rng = torch.Generator().manual_seed(seed + 9187)
    task_ids = cfg['dataset_audit']['eval_task_ids']
    power = {'darkroom': 2, 'dktd': 4}[cfg['env']]
    rewards, compressions = [], 0
    with isolated_evaluation(model, seed), torch.inference_mode():
        for start in range(0, len(task_ids), cfg['eval_batch_size']):
            tasks = [np.unravel_index(i, (cfg['grid_size'],) * power)
                     for i in task_ids[start:start + cfg['eval_batch_size']]]
            factories = [make_env(cfg, goal=t) if power == 2 else make_env(cfg, key=t[:2], goal=t[2:])
                         for t in tasks]
            envs = DummyVecEnv(factories)
            try:
                envs.seed(seed)
                result = model.evaluate_in_context(envs, cfg['horizon'] * cfg['eval_episodes'])
                rewards.append(result['reward_episode'])
                compressions += result.get('total_compressions', 0)
            finally:
                envs.close()
    return np.concatenate(rewards, axis=0), int(compressions)


def atomic_save(filename, checkpoint):
    filename = Path(filename)
    temporary = filename.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(filename)


def write_json(filename, value):
    Path(filename).write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
