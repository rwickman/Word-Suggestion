from word_suggestion import *

def main():
    data, max_sequence_length = Suggest_Util.parse_conversation_json("../data/conversation.json", 5000)
    word_to_id, id_to_word = Suggest_Util.data_to_id(data)
    vocab_size = len(word_to_id)

    word_suggest = Word_Suggestion(vocab_size, max_sequence_length, word_to_id, id_to_word)
    word_suggest.build_model()
    model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word}
    Suggest_Util.save_dict(model_metadata)
    word_suggest.train(data,True, 20, True)
    #print(word_suggest.generate_seq("They", 20))
    model_metadata = {"vocab_size" : vocab_size, "max_sequence_length" : max_sequence_length, "word_to_id" : word_to_id, "id_to_word" : id_to_word}
    Suggest_Util.save_dict(model_metadata)

if __name__ == "__main__":
    main()