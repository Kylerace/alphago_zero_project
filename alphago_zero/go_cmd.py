import argparse

# Arguments
parser = argparse.ArgumentParser(prog = "KylerGo", description='work with go models')
subparsers = parser.add_subparsers()

#training_group = parser.add_mutually_exclusive_group(required=True)
#parser.add_argument("--train", action="store_true")
training_parser = subparsers.add_parser("train", dest = "train")
training_parser.add_argument("-r", "--resume")
training_parser.add_argument("-s", "--status")

#parser.add_argument("--match", action="store_true")
match_parser = subparsers.add_parser("match", dest="match")
match_parser.add_argument("-k", '--komi', dest="komi", type=float, default=0)
match_parser.add_argument("-s", '--size', dest="size", type=int, default=7)
match_parser.add_argument("-bp", "--bplayer", dest="black_player", choices=["human", "random", "model"], default = "human") 
match_parser.add_argument("-wp", "--wplayer", dest="white_player", choices=["human", "random", "model"], default = "random") 
#0 = no human, 1 = human playing black, 2 = human playing white, 3 = human playing both white and black
#match_group.add_argument("-p", "--players")

#parser.add_argument("--file", action="store_true")
file_parser = subparsers.add_parser("file", dest="file")
#file_group.add_argument("")

#TODOKYLER: THIS SHIT DONT WORK. figure out argparse

test_input = ["match"]

args = parser.parse_args(test_input)
print(args)
#print(match_parser.parse_args(arg_input))

if args.match == True:
    from go_match import *
    size = args.size
    komi = args.komi

    bplayer = args.bplayer
    wplayer = args.wplayer
    
    if bplayer == "human":
        bplayer = HumanPlayer()
    elif bplayer == "random":
        bplayer = RandomPlayer()
    elif bplayer == "model":
        print("unimplemented!")

    if wplayer == "human":
        wplayer = HumanPlayer()
    elif wplayer == "random":
        wplayer = RandomPlayer()
    elif wplayer == "model":
        print("unimplemented!")

    players = (bplayer, wplayer)
    match_args = MatchArgs(boardsize = size, komi = komi, players = players, time_controls=(None, None))

if args.train == True:
    from go_train import *
    print("unimplemented!")
    
if args.file == True:
    from model_save import *
    print("unimplemented!")