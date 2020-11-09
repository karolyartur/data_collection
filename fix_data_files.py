## @package data_collection
#  Fix errors of data files
#
#  This module can be used to fix the errors appearing in the data files containing the robot states
#  It can be used by importing the package, or this module in a python module, or from the command line.
#
#  For command line usage call python fix_data_files.py -h for help


## Parse command line
#
#  This function parses the command line arguments
def parse_args():

    import argparse

    parser = argparse.ArgumentParser(
        description = 'Fix errors of data files')
    parser.add_argument(
        '-ifilename', dest='ifilename',
        help='Name of the dataset file to fix. Default is: data',
        default='data', type=str)
    parser.add_argument(
        '-ofilename', dest='ofilename',
        help='Name of the output (fixed) dataset file. Default is: data_correct',
        default='data_correct', type=str)

    args = parser.parse_args()
    return args


## Create dataset
#
#  This function is used for fixing data files
#  @param args Object for passing the options of the data files.
def fix_data_files(args):
    print('Reading file: ' + args.ifilename + '.txt')
    file_content = ''
    with open(args.ifilename + '.txt', 'r') as f:
        data = f.readlines()
        for line in data:
            if line.find(']') == -1:
                file_content += (' '.join(line.split()) + ' ').replace('[ ', '[')
            else:
                file_content += (' '.join(line.split()) + '\n').replace('[ ', '[').replace(' ]', ']')

    with open(args.ofilename + '.txt', 'w') as f:
        print('Saving file: ' + args.ofilename + '.txt')
        f.write(file_content)


if __name__ == "__main__":
    args = parse_args()
    fix_data_files(args)