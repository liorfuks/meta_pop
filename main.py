from src.optimizers import OpenAIOptimizer, CanonicalESOptimizer, CanonicalESMeanOptimizer
from src.policy import Policy
from src.logger import Logger

from argparse import ArgumentParser
from mpi4py import MPI
import numpy as np
import time
import json
import gym


# This will allow us to create optimizer based on the string value from the configuration file.
# Add you optimizers to this dictionary.
optimizer_dict = {
    'OpenAIOptimizer': OpenAIOptimizer,
    'CanonicalESOptimizer': CanonicalESOptimizer,
    'CanonicalESMeanOptimizer': CanonicalESMeanOptimizer
}


# Main function that executes training loop.
# Population size is derived from the number of CPUs
# and the number of episodes per CPU.
# One CPU (id: 0) is used to evaluate currently proposed
# solution in each iteration.
# run_name comes useful when the same hyperparameters
# are evaluated multiple times.
def main(ep_per_cpu, game, configuration_file, run_name):
    start_time = time.time()

    with open(configuration_file, 'r') as f:
        configuration = json.loads(f.read())

    env_name = '%sNoFrameskip-v4' % game

    # MPI stuff
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    cpus = comm.Get_size()

    # Meta Population
    meta_pop_size = 5
    meta_pop_active = list(range(meta_pop_size))
    next_havling_time = 40 # minutes for next


    mu_list = [5, 10, 20, 50, 100]

    # One cpu (rank 0) will evaluate results
    train_cpus = cpus - meta_pop_size

    # Deduce population size
    lam = train_cpus * ep_per_cpu

    # Create environment
    env = gym.make(env_name)

    # Create policy (Deep Neural Network)
    # Internally it applies preprocessing to the environment state
    policy = Policy(env, network=configuration['network'], nonlin_name=configuration['nonlin_name'])

    # Create reference batch used for normalization
    # It will be overwritten with vb from worker with rank 0
    vb = policy.get_vb()

    # Extract vector with current parameters.
    parameters_list = [policy.get_parameters() for count in range(meta_pop_size)]

    # Send parameters from worker 0 to all workers (MPI stuff)
    # to ensure that every worker starts in the same position
    for i in range(meta_pop_size):
        comm.Bcast([parameters_list[i], MPI.FLOAT], root=0)
    comm.Bcast([vb, MPI.FLOAT], root=0)

    # Set the same virtual batch for each worker
    if rank != 0:
        policy.set_vb(vb)

    if rank < meta_pop_size:
        parent_id = rank
        eval_moving_avg = 0
        forget_factor = 0.9

    if rank >= meta_pop_size:
        parent_id = int((rank-meta_pop_size)//(train_cpus/meta_pop_size))

    # Create optimizer with user defined settings (hyperparameters)
    OptimizerClass = optimizer_dict[configuration['optimizer']]
    optimizer = OptimizerClass(parameters_list, lam, rank, meta_pop_size, parent_id, mu_list[parent_id], configuration["settings"])

    # Only rank 0 worker will log information from the training
    logger = None
    if rank < meta_pop_size:    # TODO: Improve logger for meta pop
        # Initialize logger, save virtual batch and save some basic stuff at the beginning
        logger = Logger(optimizer.log_path(game, configuration['network'], run_name))
        if rank == 0:
            logger.save_vb(vb)

        # Log basic stuff
        logger.log('Game'.ljust(25) + '%s' % game, rank)
        logger.log('Network'.ljust(25) + '%s' % configuration['network'], rank)
        logger.log('Optimizer'.ljust(25) + '%s' % configuration['optimizer'], rank)
        logger.log('Number of CPUs'.ljust(25) + '%d' % cpus, rank)
        logger.log('Population'.ljust(25) + '%d' % lam, rank)
        logger.log('Dimensionality'.ljust(25) + '%d' % len(parameters_list[0]), rank)

        # Log basic info from the optimizer
        #optimizer.log_basic(logger)

    # We will count number of steps
    # frames = 4 * steps (3 * steps for SpaceInvaders)
    steps_passed = 0
    while True:
        # Iteration start time
        iter_start_time = time.time()
        # Workers that run train episodes
        if rank >= meta_pop_size:
            # Empty arrays for each episode. We save: length, reward, noise index
            lens = [0] * ep_per_cpu
            rews = [0] * ep_per_cpu
            inds = [0] * ep_per_cpu
            parent_id_arr = [0] * ep_per_cpu

            # For each episode in this CPU we get new parameters,
            # update policy network and perform policy rollout
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)
                e_rew, e_len = policy.rollout()
                lens[i] = e_len
                rews[i] = e_rew
                inds[i] = ind
                parent_id_arr[i] = parent_id


            # Aggregate information, will later send it to each worker using MPI
            msg = np.array(rews + lens + inds + parent_id_arr, dtype=np.int32)

        # Worker rank 0 that runs evaluation episodes
        else:
            rews = [0] * ep_per_cpu
            lens = [0] * ep_per_cpu
            for i in range(ep_per_cpu):
                ind, p = optimizer.get_parameters()
                policy.set_parameters(p)
                e_rew, e_len = policy.rollout()
                rews[i] = e_rew
                lens[i] = e_len

            eval_mean_rew = np.mean(rews)
            eval_max_rew = np.max(rews)
            print("real mean {}".format(eval_mean_rew))
            eval_moving_avg = eval_mean_rew + forget_factor*(eval_moving_avg-eval_mean_rew)
            print("mean eval for rank {} is {}".format(rank, eval_moving_avg))

            # Empty array, evaluation results are not used for the update
            msg = np.array(eval_moving_avg, dtype=np.int32)
            #msg = np.zeros(3 * ep_per_cpu, dtype=np.int32)

        # MPI stuff
        # Initialize array which will be updated with information from all workers using MPI
        results = np.empty((cpus, 4 * ep_per_cpu), dtype=np.int32)
        comm.Allgather([msg, MPI.INT], [results, MPI.INT])

        eval_results = results[:meta_pop_size, 0]

        # Skip empty evaluation results from worker with id 0
        results = results[meta_pop_size:, :]

        # Extract IDs and rewards
        rews = results[:, :ep_per_cpu].flatten()
        lens = results[:, ep_per_cpu:(2*ep_per_cpu)].flatten()
        ids = results[:, (2 * ep_per_cpu):(3 * ep_per_cpu)].flatten()
        par_id = results[:, (3 * ep_per_cpu):].flatten()

        rews_list = [0] * meta_pop_size
        ids_list = [0] * meta_pop_size
        train_mean_reward = [0] * meta_pop_size
        train_max_reward = [0] * meta_pop_size
        for id in meta_pop_active:
            rewards_id = [i for i, x in enumerate(par_id) if x == id]
            if not rewards_id:
                print("shittttttttttt {}".format(rewards_id))
            rews_list[id] = ([rews[i] for i in rewards_id])
            train_mean_reward[id] = (np.mean(rews_list[id]))
            train_max_reward[id] = (np.max(rews_list[id]))
            ids_list[id] = ([ids[i] for i in rewards_id])


        # Update parameters
        for i in meta_pop_active:
            optimizer.update(ids=ids_list[i], rewards=rews_list[i])

        #===============Sucssesive Halving==================
        if next_havling_time <= ((time.time()-start_time)/60):
            print("Assigning good weights to bad {}".format(((time.time() - start_time) / 60)))
            print("Eval rewards list {}".format(eval_results))
            ranking = sorted(range(len(eval_results)), key=lambda k: eval_results[k], reverse=True)
            print("ranking {}".format(ranking))
            bottom = ranking[int(0.6*meta_pop_size):]
            print("bottom {}".format(bottom))
            if parent_id in bottom:
                optimizer.assign_weights(ranking[int(len(ranking) - ranking.index(parent_id) - 1)])
                print("rank {} switch from {} to {}".format(rank,parent_id, ranking[int(len(ranking) - ranking.index(parent_id) - 1)]))
            next_havling_time += 40

        #         print("Halving now time passed {}".format(((time.time()-start_time)/60)))
        #         eval_mean = []
        #         for rank_i in range(meta_population):
        #             # print(eval_results[rank_i, :ep_per_cpu])
        #             eval_mean.append(np.mean(eval_results[rank_i, :ep_per_cpu]))
        #         print("halving rewards list {}".format(eval_mean))
        #         ranking = sorted(range(len(eval_mean)), key=lambda k: eval_mean[k], reverse=True)
        #         print("ranking {}".format(ranking))
        #         bottom = ranking[int(half_pop // 2):]
        #         print("bottom {}".format(bottom))
        #         if parent_id in bottom:
        #             old = parent_id
        #             parent_id = int(ranking.index(parent_id)-len(ranking)//2)
        #             print("switch from {} to {}".format(old, parent_id))
        #         next_havling_time *= 2
        #         half_pop /= 2
        #         ep_per_cpu //=2



        # Steps passed = Sum of episode steps from all offsprings
        steps = np.sum(lens)
        steps_passed += steps

        # Write some logs for this iteration
        # Using logs we are able to recover solution saved
        # after 1 hour of training or after 1 billion frames
        if rank < meta_pop_size:
            iteration_time = (time.time() - iter_start_time)
            time_elapsed = (time.time() - start_time) / 60
            train_mean_rew = np.mean(rews)
            train_max_rew = np.max(rews)
            logger.log('------------------------------------', rank)
            logger.log('Iteration'.ljust(25) + '%f' % (optimizer.iteration//meta_pop_size), rank)
            logger.log('EvalMeanReward'.ljust(25) + '%f' % eval_moving_avg, rank)
            logger.log('EvalMaxReward'.ljust(25) + '%f' % eval_max_rew, rank)
            logger.log('TrainMeanReward'.ljust(25) + '%f' % train_mean_rew, rank)
            logger.log('TrainMaxReward'.ljust(25) + '%f' % train_max_rew, rank)
            logger.log('StepsSinceStart'.ljust(25) + '%f' % steps_passed, rank)
            logger.log('StepsThisIter'.ljust(25) + '%f' % steps, rank)
            logger.log('IterationTime'.ljust(25) + '%f' % iteration_time, rank)
            logger.log('TimeSinceStart'.ljust(25) + '%f' % time_elapsed, rank)

            # Give optimizer a chance to log its own stuff
            # optimizer.log(logger)
            logger.log('------------------------------------', rank)

            # Write stuff for training curve plot
            stat_string = "{},\t{},\t{},\t{},\t{},\t{}\n". \
                format(steps_passed, (time.time() - start_time),
                       eval_moving_avg, eval_max_rew, train_mean_rew, train_max_rew)
            logger.write_general_stat(stat_string, rank)
            # logger.write_optimizer_stat(optimizer.stat_string())

            # Save currently proposed solution every 20 iterations
            if optimizer.iteration % 20 == 1:
                logger.save_parameters(optimizer.parameters, optimizer.iteration, rank)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-e', '--episodes_per_cpu',
                        help="Number of episode evaluations for each CPU, "
                             "population_size = episodes_per_cpu * Number of CPUs",
                        default=1, type=int)
    parser.add_argument('-g', '--game', help="Atari Game used to train an agent")
    parser.add_argument('-c', '--configuration_file', help='Path to configuration file')
    parser.add_argument('-r', '--run_name', help='Name of the run, used to create log folder name', type=str)
    args = parser.parse_args()
    return args.episodes_per_cpu, args.game, args.configuration_file, args.run_name


if __name__ == '__main__':
    ep_per_cpu, game, configuration_file, run_name = parse_arguments()
    main(ep_per_cpu, game, configuration_file, run_name)
