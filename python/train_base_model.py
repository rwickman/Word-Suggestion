from word_suggestion import *

def main():
    data, max_sequence_length = Suggest_Util.parse_conversation_json("../data/conversation.json", 100)
    data = Suggest_Util.clean_data(data)
    word_to_id, id_to_word = Suggest_Util.words_to_id(data)
    vocab_size = len(word_to_id)
    word_suggest = Word_Suggestion(vocab_size, max_sequence_length, word_to_id, id_to_word)
    word_suggest.build_model()
    
    word_suggest.train(data,True, 45, True)

    model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word}
    Suggest_Util.save_dict(model_metadata)  
    
    # print(word_suggest.generate_seq("they", 20))
    # print(word_suggest.predict(""))
    # print(word_suggest.predict("they do"))
    # print(word_suggest.predict("let's")) 
    # print(word_suggest.predict("i hope")) #so
    # print(word_suggest.predict("you're gonna"))
    
if __name__ == "__main__":
    main()