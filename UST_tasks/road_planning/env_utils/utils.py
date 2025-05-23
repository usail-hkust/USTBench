import yaml
import glob
import os
from typing import Text, Dict
import random

def get_file_path(file_path):
    # hostname = socket.gethostname()
    cwd = os.getcwd()
    file_path = os.path.join(cwd, file_path)
    return file_path

def load_yaml(file_path):
    file_path = get_file_path(file_path)
    files = glob.glob(file_path, recursive=True)
    print(file_path)
    assert(len(files) == 1)
    cfg = yaml.safe_load(open(files[0], 'r'))
    return cfg



class LLM_debug:
    def __init__(self, llm_path_or_name, batch_size, task_info):
        """
        初始化调试用的LLM类。

        :param llm_path_or_name: 通常是LLM的路径或名称，这里仅作占位
        :param batch_size: 批量大小，这里仅作占位
        :param task_info: 任务信息，这里仅作占位
        """
        self.llm_path_or_name = llm_path_or_name
        self.batch_size = batch_size
        self.task_info = task_info
        self.memory = None

    def hybrid_decision_making_pipeline(self, observation_texts, answer_form):
        """
        模拟LLM的决策过程，随机选择道路。

        :param observation_texts: 观察文本列表
        :param answer_form: 答案选项格式列表
        :return: 包含决策信息的列表
        """
        decisions = []
        for _ in observation_texts:
            # 假设 answer_form 中包含可用道路的选项
            available_roads = answer_form[0].split('/')
            selected_road = random.choice(available_roads)
            data_analysis = {"info": "随机选择道路用于调试"}
            summary = f"随机选择了道路: {selected_road}"
            decisions.append({
                "answer": selected_road,
                "data_analysis": data_analysis,
                "summary": summary
            })
        return decisions

    def hybrid_self_reflection_pipeline(self, observation_texts, decisions, decision_summaries, env_feedback_texts):

        self_reflections = []

        return self_reflections

class Config:

    def __init__(self, cfg: Text, slum_name: Text, global_seed: int, tmp: bool, root_dir: Text,
                 agent: Text = 'random', cfg_dict: Dict = None):
        self.id = cfg
        self.slum = slum_name
        self.seed = global_seed
        cwd = os.getcwd()
        if cfg_dict is not None:
            cfg = cfg_dict
        else:
            cwd = os.getcwd()
            file_path = os.path.join(cwd, 'UST_tasks/road_planning/Data/demo.yaml')
            cfg = load_yaml(file_path)
        # create dirs
        self.root_dir = os.path.join(cwd, 'tmp') if tmp else root_dir
        self.data_dir = 'data/Epworth_Before'
        self.cfg_dir = os.path.join(self.root_dir, self.slum, agent, str(self.seed))
        self.model_dir = os.path.join(self.cfg_dir, 'models')
        self.log_dir = os.path.join(self.cfg_dir, 'log')
        self.tb_dir = os.path.join(self.cfg_dir, 'tb')
        self.plan_dir = os.path.join(self.cfg_dir, 'plan')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        os.makedirs(self.plan_dir, exist_ok=True)

        self.agent = agent

        # env
        self.objectives_plan = cfg.get('objectives_plan', '')
        self.init_plan = cfg.get('init_plan', '')
        self.env_specs = cfg.get('env_specs', dict())
        self.reward_specs = cfg.get('reward_specs', dict())
        self.obs_specs = cfg.get('obs_specs', dict())

        # agent config
        self.agent_specs = cfg.get('agent_specs', dict())

        # training config
        self.gamma = cfg.get('gamma', 0.99)
        self.tau = cfg.get('tau', 0.95)
        self.state_encoder_specs = cfg.get('state_encoder_specs', dict())
        self.policy_specs = cfg.get('policy_specs', dict())
        self.value_specs = cfg.get('value_specs', dict())
        self.lr = cfg.get('lr', 4e-4)
        self.weightdecay = cfg.get('weightdecay', 0.0)
        self.eps = cfg.get('eps', 1e-5)
        self.value_pred_coef = cfg.get('value_pred_coef', 0.5)
        self.entropy_coef = cfg.get('entropy_coef', 0.01)
        self.clip_epsilon = cfg.get('clip_epsilon', 0.2)
        self.max_num_iterations = cfg.get('max_num_iterations', 1000)
        self.num_episodes_per_iteration = cfg.get('num_episodes_per_iteration', 1000)
        self.max_sequence_length = cfg.get('max_sequence_length', 100)
        self.original_max_sequence_length = cfg.get('max_sequence_length', 100)
        self.num_optim_epoch = cfg.get('num_optim_epoch', 4)
        self.mini_batch_size = cfg.get('mini_batch_size', 1024)
        self.save_model_interval = cfg.get('save_model_interval', 10)

    def train(self) -> None:
        """Train land use only"""
        self.skip_land_use = False
        self.skip_road = True
        self.max_sequence_length = self.original_max_sequence_length // 2

    def finetune(self) -> None:
        """Change to road network only"""
        self.skip_land_use = True
        self.skip_road = False
        self.max_sequence_length = self.original_max_sequence_length // 2

    def log(self, logger, tb_logger):
        """Log cfg to logger and tensorboard."""
        logger.info(f'data_dir:{self.data_dir}')
        logger.info(f'id: {self.id}')
        logger.info(f'seed: {self.seed}')
        logger.info(f'objectives_plan: {self.objectives_plan}')
        logger.info(f'init_plan: {self.init_plan}')
        logger.info(f'env_specs: {self.env_specs}')
        logger.info(f'reward_specs: {self.reward_specs}')
        logger.info(f'obs_specs: {self.obs_specs}')
        logger.info(f'agent_specs: {self.agent_specs}')
        logger.info(f'gamma: {self.gamma}')
        logger.info(f'tau: {self.tau}')
        logger.info(f'state_encoder_specs: {self.state_encoder_specs}')
        logger.info(f'policy_specs: {self.policy_specs}')
        logger.info(f'value_specs: {self.value_specs}')
        logger.info(f'lr: {self.lr}')
        logger.info(f'weightdecay: {self.weightdecay}')
        logger.info(f'eps: {self.eps}')
        logger.info(f'value_pred_coef: {self.value_pred_coef}')
        logger.info(f'entropy_coef: {self.entropy_coef}')
        logger.info(f'clip_epsilon: {self.clip_epsilon}')
        logger.info(f'max_num_iterations: {self.max_num_iterations}')
        logger.info(f'num_episodes_per_iteration: {self.num_episodes_per_iteration}')
        logger.info(f'max_sequence_length: {self.max_sequence_length}')
        logger.info(f'num_optim_epoch: {self.num_optim_epoch}')
        logger.info(f'mini_batch_size: {self.mini_batch_size}')
        logger.info(f'save_model_interval: {self.save_model_interval}')

        if tb_logger is not None:
            tb_logger.add_hparams(
                hparam_dict={
                    'id': self.id,
                    'seed': self.seed,
                    'objectives_plan': self.objectives_plan,
                    'init_plan': self.init_plan,
                    'env_specs': str(self.env_specs),
                    'reward_specs': str(self.reward_specs),
                    'obs_specs': str(self.obs_specs),
                    'agent_specs': str(self.agent_specs),
                    'gamma': self.gamma,
                    'tau': self.tau,
                    'state_encoder_specs': str(self.state_encoder_specs),
                    'policy_specs': str(self.policy_specs),
                    'value_specs': str(self.value_specs),
                    'lr': self.lr,
                    'weightdecay': self.weightdecay,
                    'eps': self.eps,
                    'value_pred_coef': self.value_pred_coef,
                    'entropy_coef': self.entropy_coef,
                    'clip_epsilon': self.clip_epsilon,
                    'max_num_iterations': self.max_num_iterations,
                    'num_episodes_per_iteration': self.num_episodes_per_iteration,
                    'max_sequence_length': self.max_sequence_length,
                    'num_optim_epoch': self.num_optim_epoch,
                    'mini_batch_size': self.mini_batch_size,
                    'save_model_interval': self.save_model_interval},
                metric_dict={'hparam/placeholder': 0.0})
