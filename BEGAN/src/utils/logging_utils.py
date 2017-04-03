from colorama import init
from colorama import Fore, Back, Style
import tensorflow as tf
from terminaltables import SingleTable
from natsort import natsorted


def print_table(TABLE_DATA):

    table_instance = SingleTable(TABLE_DATA, "")
    table_instance.justify_columns[2] = 'right'
    print(table_instance.table)
    print


def print_bright(s):

    init()
    print(Style.BRIGHT + s + Style.RESET_ALL)


def print_green(info, value=""):

    print(Fore.GREEN + "[%s] " % info + Style.RESET_ALL + str(value))


def print_red(info, value=""):

    print(Fore.RED + "[%s] " % info + Style.RESET_ALL + str(value))


def print_session(session_type):

    FLAGS = tf.app.flags.FLAGS

    print_bright("\nSetting up TF %s session:" % session_type)
    for key in natsorted(FLAGS.__dict__["__flags"].keys()):
        if "dir" not in key:
            print_green(key, FLAGS.__dict__["__flags"][key])


def print_directories(list_delete, list_create):

    print_bright("\nConfiguring directories:")

    for d in list_delete:
        print_red("Deleting", d)

    for d in list_create:
        print_green("Creating", d)


def print_initialize():

    print_bright("\nInitialization:")
    print_green("Created session saver")
    print_green("Ran init ops")


def print_summaries():

    print_bright("\nSummaries:")
    list_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    for t in list_summaries:
        print_green(t.name)


def print_queues():

    print_bright("\nQueues:")
    print_green("Created coordinator")
    print_green("Started queue runner")


def print_check_data(out, list_data):

    print
    TABLE_DATA = (('Variable Name', 'Shape', "Min value", "Max value"),)
    for o, t in zip(out, list_data):
        TABLE_DATA += (tuple([t.name, str(o.shape), "%.3g" % o.min(), "%.3g" % o.max()]),)
    print_table(TABLE_DATA)


def print_checkpoint(checkpoint):

    print_bright("\nCheckpointing:")
    print_green(checkpoint.model_checkpoint_path)


def print_meta_graph(meta_graph_path):

    print_bright("\nMetagraph:")
    print_green("Loaded " + meta_graph_path)


def print_restore():

    print_bright("\nRestored session")
