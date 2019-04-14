from word_suggestion import *

#TODO take in CLI argument for the user

def main():
    model_metadata = Suggest_Util.load_dict()
    suggest = Word_Suggestion(*model_metadata.values())

    

if __name__ == "__main__":
    main()