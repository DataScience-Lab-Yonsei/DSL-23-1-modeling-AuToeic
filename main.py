from text2img import Text2Img
from img2toeic import CreateToeic
from clip import Clipevaluate
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("-c", "--CAPTIONS", dest="CAPTIONS", help="Single caption to generate for or filepath for .txt "
                                                            "file of captions to generate for", default=None, type=str)
args = parser.parse_args()
if args.CAPTIONS is None:
    print("\nNo caption supplied - using the default of \"a happy dog\".\n")
    captions = ['a happy dog']
elif not args.CAPTIONS.endswith(".txt"):
    captions = [args.CAPTIONS]
elif args.CAPTIONS.endswith(".txt"):
    with open(args.CAPTIONS, 'r') as f:
        lines = f.readlines()
    captions = [line[:-1] if line.endswith('\n') else line for line in lines]
else:
    raise ValueError("Please input a valid argument for --CAPTIONS")

Text2Img(captions)
CreateToeic()
Clipevaluate()