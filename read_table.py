import sys
import pickle


if __name__ == "__main__":
    argv = sys.argv
    i = int(argv[1])
    I = int(argv[2])
    d = int(argv[3])
    K = int(argv[4])
    h = float(argv[5])
    tau = float(argv[6])
    
    with open("idun_files/table.pickle", "rb") as pickle_file:
        table = pickle.load(pickle_file)
    
    print(table[(i, I, d, K, h, tau)])
