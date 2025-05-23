from pprint import pprint
# import pygad
import setproctitle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import app, flags

from UST_tasks.urban_planning.utils.config import Config
from UST_tasks.urban_planning.agents.urban_planning_agent import UrbanPlanningAgent

# ------------------ Flag Definitions ------------------ #
flags.DEFINE_string('root_dir', '/data/', 'Root directory for logs, summaries, and checkpoints.')
flags.DEFINE_string('cfg', 'hlg', 'Configuration file for RL training.')
flags.DEFINE_bool('tmp', False, 'Use temporary storage.')
flags.DEFINE_enum('agent', 'rule-centralized', ['rl-sgnn', 'rl-mlp', 'rule-centralized', 'rule-decentralized', 'gsca', 'ga'], 'Agent type.')
flags.DEFINE_bool('mean_action', True, 'Use greedy strategy.')
flags.DEFINE_bool('visualize', False, 'Visualize the planning process.')
flags.DEFINE_bool('only_road', False, 'Visualize only road planning.')
flags.DEFINE_bool('save_video', False, 'Save video of the planning process.')
flags.DEFINE_integer('global_seed', 2025, 'Seed for environment and weight initialization.')
flags.DEFINE_string('iteration', '0', 'Start iteration (number or "best").')

FLAGS = flags.FLAGS


# ------------------ Helper Functions ------------------ #
def set_device_and_seed(cfg):
    """Set the device and random seed for reproducibility."""
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cpu')  # Currently using CPU by default
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    return device


def create_agent(cfg, device):
    """Create and return the UrbanPlanningAgent."""
    checkpoint = int(FLAGS.iteration) if FLAGS.iteration.isnumeric() else FLAGS.iteration
    return UrbanPlanningAgent(cfg=cfg, dtype=torch.float32, device=device, num_threads=1,
                              training=False, checkpoint=checkpoint, restore_best_rewards=True)


def run_inference(agent):
    """Run inference for the agent."""
    agent.infer(num_samples=1, mean_action=FLAGS.mean_action, visualize=FLAGS.visualize,
                save_video=FLAGS.save_video, only_road=FLAGS.only_road)


def run_ga(agent):
    """Run GA-based planning."""
    best_solution, _ = agent.load_ga()
    _, plan, log_eval = agent.fitness_ga(best_solution, num_samples=1, mean_action=FLAGS.mean_action,
                                         visualize=FLAGS.visualize, return_log_eval=True)
    pprint(plan, indent=4, sort_dicts=False)
    agent.save_plan(log_eval)


# ------------------ Main Loop ------------------ #
def main_loop(_):
    """Main loop for the urban planning process."""
    setproctitle.setproctitle('urban_planning')

    # Load the configuration
    cfg = Config(FLAGS.cfg, FLAGS.global_seed, FLAGS.tmp, FLAGS.root_dir, FLAGS.agent)

    # Set device and random seed
    device = set_device_and_seed(cfg)

    # Create the agent
    agent = create_agent(cfg, device)

    # Freeze land use if only road planning is required
    if FLAGS.only_road:
        agent.freeze_land_use()

    # Run either inference or GA-based planning based on the selected agent
    if FLAGS.agent != 'ga':
        run_inference(agent)
    else:
        run_ga(agent)


# ------------------ Entry Point ------------------ #
if __name__ == '__main__':
    flags.mark_flags_as_required(['cfg', 'global_seed'])
    app.run(main_loop)
