import argparse
import sys
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    processors = dict()
    processors['recognition'] = import_class('processor.rec_parsing.REC_Processor')

    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    arg = parser.parse_args()
    Processor = processors[arg.processor] # arg.processor: recognition
    p = Processor(sys.argv[2:]) # sys.argv[2:]: '-c', 'config/ntu120_xsub/train.yaml'

    # Processor = processors['recognition']
    # p = Processor(['-c', 'config/ntu120_xsub/train.yaml']) 
    p.start()