import threading
import numpy as np
from util.output_json import OutputJson
import copy


class Worker(object):
    def __init__(self, a3c, env, i, args, logger, OJ):
        self.env = copy.deepcopy(env)
        self.name = a3c.thread_AC[i].name
        self.AC = a3c.thread_AC[i]
        self.COORD = a3c.COORD
        self.A3C = a3c
        self.args = args
        self.logger = logger
        self.OJ = OJ

    def work(self):
        global memory, GLOBAL_EP, GLOBAL_STEP, discount_memory
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.COORD.should_stop() and GLOBAL_EP < self.args['numGames']:
            s = self.env.reset()
            s = np.array(s)
            episode_reward = 0
            episode_discount_reward = 0
            step = 0
            success = 0
            while True:
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                success += info['success']
                s_ = np.array(s_)
                if self.args['reward_normalize']:
                    normalize_reward = r / self.args['done_reward']
                else:
                    normalize_reward = r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(normalize_reward)
                episode_discount_reward = episode_discount_reward + round(r * np.power(self.args['reward_decay'], step), 8)
                episode_reward = episode_reward + r

                if (step != 0 and step % self.args['batch_size'] == 0) or step == self.args['epi_step'] or done:
                    if len(buffer_s) != 0:
                        self.AC.update(buffer_s, buffer_a, buffer_r, done, s_, GLOBAL_STEP, GLOBAL_EP)
                        buffer_s, buffer_a, buffer_r = [], [], []

                if done or step == self.args['epi_step']:  # update global and assign to local net
                    memory[GLOBAL_EP % self.args['reward_memory']] = episode_reward
                    discount_memory[GLOBAL_EP % self.args['reward_memory']] = episode_discount_reward
                    mean_memory = np.mean(memory)
                    discount_mean_memory = np.mean(discount_memory)
                    success = (success > 0).__float__()
                    self.OJ.update([done, step, episode_discount_reward, discount_mean_memory, episode_reward, mean_memory, GLOBAL_EP,success])
                    self.OJ.print_first()

                    self.logger.write_tb_log('discount_reward', episode_discount_reward, GLOBAL_EP)
                    self.logger.write_tb_log('discount_reward_mean', discount_mean_memory, GLOBAL_EP)
                    self.logger.write_tb_log('undiscounted reward', episode_reward, GLOBAL_EP)
                    self.logger.write_tb_log('reward_mean', mean_memory, GLOBAL_EP)
                    self.logger.write_tb_log('success', success, GLOBAL_EP)
                    success = 0

                    GLOBAL_EP += 1
                    break

                s = s_
                GLOBAL_STEP += 1
                step += 1

            if self.args['save_model'] and (GLOBAL_EP % self.args['save_per_episodes'] == 1):
                self.A3C.save_model(
                     self.args['results_path'] + self.args['SAVE_PATH'] + "/model" + "_" + str(GLOBAL_EP - 1))
                self.OJ.save(self.args['results_path'] + self.args['reward_output'], self.args['output_filename'])


def run(args, env, a3c, logger):
    global memory, GLOBAL_EP, GLOBAL_STEP, discount_memory
    memory = np.zeros(args['reward_memory'])
    discount_memory = np.zeros(args['reward_memory'])
    GLOBAL_EP = 0
    GLOBAL_STEP = 0
    field = ['win', 'step', 'discounted_reward', 'discount_reward_mean', 'undiscounted_reward', 'reward_mean', 'episode','success']
    OJ = OutputJson(field)

    workers = []
    for i in range(a3c.N_WORKERS):
        workers.append(Worker(a3c, env, i, args, logger, OJ))
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)  # 创建一个线程，并分配其工作
        t.start()  # 开启线程
        worker_threads.append(t)
    a3c.COORD.join(worker_threads)  # 把开启的线程加入主线程，等待threads结束

    OJ.save(args['results_path'] + args['reward_output'], args['output_filename'])
    if args['save_model']:
        a3c.save_model(args['results_path'] + args['SAVE_PATH'] + "/model")
