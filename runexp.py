from datetime import datetime       # for date for result dir
from multiprocessing import Pool    # non-distributed parallellism
from time import time               # timing
from progress.bar import Bar        # progress indicators
import os                           # for mkdir
import os.path                      # for path manipulation
import yaml                         # for loading experiment files
import sys                          # for path, exit
import shutil                       # file copying
import logging                      # for logging, duh
import argparse                     # CLI
import itertools                    # imap and count
import scoop                        # for distributed parallellism

from boolnet.exptools.learn_boolnet import learn_bool_net
import boolnet.exptools.config_tools as cfg


def initialise_logging(settings, result_dir):
    ''' sets up logging with user defined level. '''
    level_dict = {'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG,
                  'none': logging.ERROR}
    log_filename = os.path.join(result_dir, 'log')
    # default to 'none'
    log_level = level_dict[settings.get('debug_level', 'warning')]
    logging.basicConfig(filename=log_filename,
                        level=log_level)


def initialise_notifications(args):
    if args.notify:
        try:
            import requests
            import instapush
            try:
                with open(args.ip_config) as f:
                    ip_settings = yaml.load(f, Loader=yaml.CSafeLoader)
                appid = ip_settings['appid']
                secret = ip_settings['secret']
                return instapush.App(appid=appid, secret=secret)
            except FileNotFoundError:
                print('Instapush config not found: {}'.format(args.ip_config))
            except yaml.YAMLError as err:
                print('Invalid instapush config: {}'.format(err))
            except requests.exceptions.RequestException as err:
                print('Failed to initialise notifications: {}'.format(err))
        except ImportError as err:
            print('Failed to import notification APIs: {}'.format(err))
        # disable notifications in the event of any errors
        print('Notifications disabled.\n')
        return None
    else:
        return None


def notify(notifier, settings, total_time, notes='none'):
    if notifier is not None:
        import requests
        try:
            message = {'name': str(settings['name']),
                       'time': str(total_time),
                       'num warnings': str(0),  # not implemented
                       'notes': str(notes)}
            ret = notifier.notify(event_name='experiment_complete',
                                  trackers=message)
            if ret.get('error', False):
                print('Notification error: {}'.format(ret))
        except requests.exceptions.RequestException as err:
            print('Failed to send notification: {}.'.format(err))


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('experiment',
                        type=argparse.FileType('r'),
                        help='experiment config filename.')
    parser.add_argument('-n', '--numprocs', metavar='N', type=int,
                        default=8, choices=range(0, 17),
                        help='how many parallel processes to use (give 0 for '
                             'scoop).')
    parser.add_argument('-p', '--notify', action='store_true',
                        help='enable push notifications.')
    parser.add_argument('-c', '--ip-config', metavar='file', type=str,
                        default='instapush.cfg',
                        help='instapush config file path (for notifications).')
    parser.add_argument('-r', '--result-dir', type=str, metavar='dir',
                        default='HMRI/experiments/results',
                        help='directory to store results in (in own subdir).')
    parser.add_argument('-b', '--batch-mode', action='store_true',
                        help='suppress progress bars.')

    return parser.parse_args()


def create_result_dir(base_dir, exp_name):
    ''' Generates a new timestamped directory for the results and copies
        the experiment script, experiment file and git hash into it.'''
    # make directory for the results
    result_dir = os.path.join(base_dir, '{}_{}_'.format(
        exp_name, datetime.now().strftime('%y-%m-%d')))

    # find the first number i such that 'Results_datetime_i' is not
    # already a directory and make that new directory
    result_dir += next(str(i) for i in itertools.count()
                       if not os.path.isdir(result_dir + str(i)))
    os.makedirs(result_dir)

    # return the directory name for the results to go in
    return result_dir


def initialise(args):
    ''' This is responsible for setting up the output directory,
        checking the git repo isn't dirty and setting up logging.'''
    # Check version
    if sys.version_info.major != 3:
        sys.exit("Requires python 3.")

    # load experiment file
    settings = yaml.load(args.experiment, Loader=yaml.CSafeLoader)

    # create result directory
    result_dir = create_result_dir(args.result_dir, settings['name'])

    # copy experiment config into results directory
    shutil.copy(args.experiment.name, result_dir)

    return settings, result_dir


def run_tasks(tasks, num_processes, out_stream, batch_mode):
    out_stream.write('[')
    if num_processes == 1:
        run_sequential(tasks, out_stream, batch_mode)
    elif num_processes < 1:
        run_scooped(tasks, out_stream, batch_mode)
    else:
        # Run the actual learning as a parallel process
        run_parallel(tasks, num_processes, out_stream, batch_mode)
    out_stream.write(']')


def run_parallel(tasks, num_processes, out_stream, batch_mode):
    ''' runs the given configurations '''
    suffix_fmt = 'completed: %(index)d/%(max)d | elapsed: %(elapsed)ds | eta: %(eta)ds'
    with Pool(processes=num_processes) as pool:
        if not batch_mode:
            bar = Bar('Parallelised ({})'.format(num_processes),
                      max=len(tasks), suffix=suffix_fmt)
            bar.update()
        # uses unordered map to ensure results are dumped as soon as available
        for i, result in enumerate(pool.imap_unordered(learn_bool_net, tasks)):
            cfg.dump_results_partial(result, out_stream, i == 0)
            if not batch_mode:
                bar.next()
        if not batch_mode:
            bar.finish()


def scoop_worker_wrapper(*args, **kwargs):
    import traceback
    try:
        # call to actual worker code
        return learn_bool_net(*args, **kwargs)
    except:
        type, value, tb = sys.exc_info()
        lines = traceback.format_exception(type, value, tb)
        print(''.join(lines))
        raise


def run_scooped(tasks, out_stream, batch_mode):
    ''' runs the given configurations '''
    suffix_fmt = 'completed: %(index)d/%(max)d | elapsed: %(elapsed)ds | eta: %(eta)ds'
    if not batch_mode:
        bar = Bar('Scooped', max=len(tasks), suffix=suffix_fmt)
        bar.update()
    # uses unordered map to ensure results are dumped as soon as available
    for i, result in enumerate(scoop.futures.map_as_completed(
            scoop_worker_wrapper, tasks)):
        cfg.dump_results_partial(result, out_stream, i == 0)
        if not batch_mode:
            bar.next()
    if not batch_mode:
        bar.finish()


def run_sequential(tasks, out_stream, batch_mode):
    ''' runs the given configurations '''
    suffix_fmt = 'completed: %(index)d/%(max)d | elapsed: %(elapsed)ds | eta: %(eta)ds'
    if not batch_mode:
        bar = Bar('Sequential', max=len(tasks), suffix=suffix_fmt)
        bar.update()
    # map gives an iterator so results are dumped as soon as available
    for i, result in enumerate(map(learn_bool_net, tasks)):
        cfg.dump_results_partial(result, out_stream, i == 0)
        if not batch_mode:
            bar.next()
    if not batch_mode:
        bar.finish()


# ############################## MAIN ####################################### #
def main():
    start_time = time()

    args = parse_arguments()

    notifier = initialise_notifications(args)

    settings, result_dir = initialise(args)

    # initialise logging
    initialise_logging(settings, result_dir)

    try:
        print('Directories initialised.')
        print('Results in: ' + result_dir + '\n')

        # generate learning tasks
        try:
            configurations = cfg.generate_configurations(settings,
                                                         args.batch_mode)
            print('Done: {} configurations generated.'.format(
                len(configurations)))
            tasks = cfg.generate_tasks(configurations, args.batch_mode)
            print('Done: {} tasks generated.\n'.format(len(tasks)))
        except cfg.ValidationError as err:
            print(err)
            print('\nExperiment aborted.')
            print('Result directory will still exist: {}'.format(result_dir))
            logging.shutdown()
            return

        result_filename = os.path.join(result_dir, 'results.json')
        with open(result_filename, 'w') as results_stream:
            # Run the actual learning as a parallel process
            run_tasks(tasks, args.numprocs, results_stream, args.batch_mode)

        total_time = time() - start_time

        print('Runs completed in {}s.'.format(total_time))

        notify(notifier, settings, total_time)

    finally:
        logging.shutdown()

if __name__ == '__main__':
    main()
